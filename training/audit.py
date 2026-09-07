"""The spec-level invariants, I1-I9 (docs/pipelines/05 §1).

🔴 These exist because of a measured pattern in this repo: six defects were
found in one review session, **none by a green test suite**, and two shapes
recurred -- a test asserting an *adjacent* quantity, and a component inherited
from a source recipe that was correct in its regime and silently wrong in ours.

⚠️ Every check here measures an **empirically drawn stream**, never the config
it was drawn from. The constraint arithmetic can be right while the sampler
implementing it is wrong, and only the stream catches that. They need no
corpus, no model and no GPU, so they can run before the corpus exists.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from training.sampler import C1_BOUNDS, Sampler
from training.spec import (CELL_TABLE, SampleSpec, cell_labels, is_fake_cell,
                           stratum_of)

__all__ = ["AuditReport", "SHORTCUT_AUC_GATE", "audit_specs", "run_audit"]

#: ★ docs/data/07 E-S2. Above this, neutralise before training anything.
SHORTCUT_AUC_GATE = 0.60


@dataclass
class AuditReport:
    """One row per invariant, with the measured quantity that decided it."""

    results: dict[str, tuple[bool, str]]

    @property
    def ok(self) -> bool:
        return all(passed for passed, _ in self.results.values())

    @property
    def failures(self) -> dict[str, str]:
        return {k: why for k, (passed, why) in self.results.items() if not passed}

    def raise_for_status(self) -> None:
        if not self.ok:
            lines = "\n".join(f"  {k}: {why}" for k, why in self.failures.items())
            raise AssertionError(f"spec audit failed:\n{lines}")

    def __str__(self) -> str:
        return "\n".join(f"{'PASS' if p else 'FAIL'}  {k}: {why}"
                         for k, (p, why) in sorted(self.results.items()))


def _fraction(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _feature_frame(specs: Sequence[SampleSpec]) -> tuple[np.ndarray, list[str]]:
    """Every structural knob the sampler controls, as a numeric matrix.

    🔴 Includes ``transforms`` **parameters** and the whole ``normalize`` draw,
    not just transform names. A review found that a name-only frequency table is
    blind to a leak carried in a parameter -- ``rawboost(strength=0.9 if fake
    else 0.1)`` puts the *name* on both sides at exactly equal rate, so the
    frequency check reports 0.0000 while the parameter separates the labels at
    AUC 1.000. ``normalize`` was read by nothing at all, and it is the A-S1
    test-chain draw: the single highest-leverage stage in the pipeline.
    """
    rows: list[dict[str, float]] = []
    for s in specs:
        gains = [c.gain_db for c in s.components]
        feat: dict[str, float] = {
            "duration_s": s.duration_s,
            "is_composed": float(s.render_mode == "composed"),
            "is_sequential": float(s.structure == "sequential"),
            "n_components": float(len(s.components)),
            "mean_gain_db": float(np.mean(gains)),
            "gain_spread_db": float(max(gains) - min(gains)),
            "crossfade_ms": s.crossfade_ms,
        }
        # ⚠️ `source_offset_s` is deliberately NOT a feature. It is where inside
        # the source file we start reading -- the model sees rendered audio and
        # cannot observe it, so it cannot be a shortcut. It IS correlated with
        # the source pool's duration statistics, which differ between the fake
        # and real whole-file pools by chance when those pools are small: with
        # 12 real and 9 fake instrumental whole-file rows it reached AUC 0.764
        # inside the music-only stratum and failed this gate on a corpus with no
        # actual leak. Source-duration imbalance is a manifest property worth
        # knowing about, but it is not a spec-level shortcut.
        for name, params in s.transforms:
            feat[f"t:{name}"] = 1.0
            for k, v in params.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    feat[f"t:{name}:{k}={v}"] = 1.0
                else:
                    feat[f"t:{name}:{k}"] = float(v)
        for k, v in s.normalize.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                feat[f"n:{k}={v}"] = 1.0
            else:
                feat[f"n:{k}"] = float(v)
        rows.append(feat)

    # Keep columns seen often enough to mean something; a one-per-sample column
    # would let the model memorise rather than find a shortcut.
    counts = Counter(k for r in rows for k in r)
    keep = sorted(k for k, c in counts.items() if c >= max(10, 0.01 * len(rows)))
    X = np.zeros((len(rows), len(keep)), dtype=float)
    for i, r in enumerate(rows):
        for j, k in enumerate(keep):
            X[i, j] = r.get(k, 0.0)
    return X, keep


def _auc_or_none(X: np.ndarray, y: np.ndarray) -> float | None:
    """Cross-validated AUC. `None` when the sample cannot support an estimate.

    ⚠️ Cross-validated, not in-sample: the feature matrix one-hots categorical
    transform and normalize values, and an in-sample fit over many sparse
    columns inflates AUC and manufactures false alarms. A real leak survives
    cross-validation trivially -- the trapped stream this was built against
    scores 1.000 either way.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(y) < 200 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 50:
        return None
    if X.shape[1] == 0 or not np.isfinite(X).all():
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    from sklearn.metrics import roc_auc_score
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _metadata_shortcut(specs: Sequence[SampleSpec]) -> tuple[bool, str]:
    """Can structural metadata alone predict FILE_FAKE? (E-S2 / VG2 at spec level.)

    ⚠️ ``stratum`` is not a feature. Presence is a legitimate signal the model is
    *asked* to predict, so including it would measure the label space rather than
    our corpus.

    🔴 But the probe runs **per stratum as well as pooled, and takes the worst**.
    Because the model is trained to predict presence, it effectively knows the
    stratum for free -- so a leak that cancels marginally while pointing opposite
    ways inside two strata is fully available to it. A review built exactly that:
    a duration shift of opposite sign in mixed vs non-mixed files scored 0.4986
    marginally (invisible) and 0.7719 with the interaction.

    🔴 **Cell 9 is excluded**, for the same reason it is excluded from the
    mixedness balance: it is always REAL *and* never composed, contributing an
    unfixable correlation. Including it, the strict policy (``f8 = 1``) scored
    0.6003 and failed this gate -- a false alarm blocking a policy we support.
    """
    specs = [s for s in specs if s.cell != 9]
    if not specs:
        return True, "skipped: no cells 1-8 in the stream"

    y_all = np.array([s.file_fake for s in specs])
    X_all, names = _feature_frame(specs)
    scored: list[tuple[float, str]] = []

    pooled = _auc_or_none(X_all, y_all)
    if pooled is not None:
        scored.append((pooled, "pooled"))
    for stratum in ("voice-only", "music-only", "mixed"):
        idx = [i for i, s in enumerate(specs) if s.stratum == stratum]
        if len(idx) < 200:
            continue
        auc = _auc_or_none(X_all[idx], y_all[idx])
        if auc is not None:
            scored.append((auc, stratum))

    if not scored:
        return True, "skipped: not enough samples to estimate an AUC"
    worst, where = max(scored)
    detail = ", ".join(f"{w}={a:.4f}" for a, w in sorted(scored, key=lambda t: -t[0]))
    return (worst < SHORTCUT_AUC_GATE,
            f"worst metadata-only CV AUC = {worst:.4f} in {where!r} "
            f"over {len(names)} feature(s); gate < {SHORTCUT_AUC_GATE} [{detail}]")


def audit_specs(specs: Sequence[SampleSpec], manifest: pd.DataFrame | None = None,
                slice_: str | None = None, tol: float = 0.02,
                batch_size: int = 32, min_present: int = 2) -> AuditReport:
    """Run I1-I9 over a drawn stream."""
    if not specs:
        raise ValueError("audit_specs needs at least one spec")
    r: dict[str, tuple[bool, str]] = {}
    n = len(specs)
    fake = [s for s in specs if s.file_fake == 1]
    real = [s for s in specs if s.file_fake == 0]

    # -- I1: P(T | L) = P(T) for every transform NAME ------------------------ #
    # ⚠️ Names only, and deliberately so -- it is cheap and interpretable, and it
    # localises which transform is skewed. It is NOT sufficient on its own: a
    # review showed `rawboost(strength=0.9 if fake else 0.1)` puts the name on
    # both sides at exactly equal rate, so this reports 0.0000 while the
    # parameter separates the labels at AUC 1.000. I1b covers parameters.
    by_label: dict[int, Counter] = {0: Counter(), 1: Counter()}
    for s in specs:
        for name, _ in s.transforms:
            by_label[s.file_fake][name] += 1
    names = set(by_label[0]) | set(by_label[1])
    worst, worst_name = 0.0, "-"
    for name in names:
        p1 = _fraction(by_label[1][name], len(fake))
        p0 = _fraction(by_label[0][name], len(real))
        if abs(p1 - p0) > worst:
            worst, worst_name = abs(p1 - p0), name
    r["I1_transform_name_independence"] = (
        worst <= tol,
        f"worst |P(T|FAKE) - P(T|REAL)| = {worst:.4f} on {worst_name!r} "
        f"over {len(names)} transform NAME(s), tol {tol}. "
        f"⚠️ names only -- parameter-level leaks are I1b's job")

    # -- I1b: the metadata shortcut audit, E-S2 at spec level ---------------- #
    # 🔴 The joint check. Each balance above can hold individually while a
    # *combination* of structural features still separates the labels -- and
    # nothing else here would see it. This is docs/data/07 E-S2 ("logistic
    # regression on metadata-only features, gate AUC < 0.60"), promoted to VG2,
    # run over specs instead of over decoded audio.
    r["I1b_metadata_shortcut_auc"] = _metadata_shortcut(specs)

    # -- I2: composedness label-independent WITHIN each presence stratum ----- #
    # ⚠️ The stratified form. A marginal balance can hold while composedness
    # predicts the label inside a stratum (docs/pipelines/02 §3).
    strat: dict[tuple[str, int], list[int]] = defaultdict(list)
    for s in specs:
        strat[(s.stratum, s.file_fake)].append(int(s.render_mode == "composed"))
    worst, worst_where = 0.0, "-"
    for stratum in ("voice-only", "music-only", "mixed"):
        f1, f0 = strat.get((stratum, 1)), strat.get((stratum, 0))
        if not f1 or not f0:
            continue
        gap = abs(sum(f1) / len(f1) - sum(f0) / len(f0))
        if gap > worst:
            worst, worst_where = gap, stratum
    r["I2_stratified_composedness"] = (
        worst <= tol,
        f"worst within-stratum |P(composed|FAKE) - P(composed|REAL)| = {worst:.4f} "
        f"in {worst_where!r}, tol {tol}")

    # -- I2b: mixedness label-independent, over cells 1-8 -------------------- #
    # Cell 9 excluded: a file with no components cannot be fake, so the
    # "neither" stratum can never be label-balanced.
    c8 = [s for s in specs if s.cell != 9]
    f8_, r8_ = [s for s in c8 if s.file_fake], [s for s in c8 if not s.file_fake]
    pm_f = _fraction(sum(s.stratum == "mixed" for s in f8_), len(f8_))
    pm_r = _fraction(sum(s.stratum == "mixed" for s in r8_), len(r8_))
    r["I2b_mixedness_balance"] = (
        abs(pm_f - pm_r) <= tol,
        f"P(mixed|FAKE) = {pm_f:.4f} vs P(mixed|REAL) = {pm_r:.4f} "
        f"(cells 1-8), tol {tol}")

    # -- I3: real component files appear on both sides of FILE_FAKE ---------- #
    sides: dict[str, set[int]] = defaultdict(set)
    draw_counts: Counter = Counter()
    for s in specs:
        for c in s.components:
            sides[c.file_id].add(s.file_fake)
            draw_counts[c.file_id] += 1
    if manifest is not None:
        real_ids = set(manifest.loc[
            manifest.pool.isin(["A", "C"]), "file_id"].astype(str))
        # ⚠️ Only files drawn at least twice can testify. A file drawn once
        # cannot appear on both sides, and counting it measures the draw budget
        # rather than the sampler -- at 1,500 draws over a 400-file real pool
        # this reported a failure on a corpus with nothing wrong with it.
        drawn_real = {fid for fid in sides if fid in real_ids}
        eligible = {fid for fid in drawn_real if draw_counts[fid] >= 2}
        both = {fid for fid in eligible if sides[fid] == {0, 1}}
        frac = _fraction(len(both), len(eligible))
        if len(eligible) < 20:
            r["I3_real_components_on_both_sides"] = (
                True,
                f"skipped: only {len(eligible)} real component(s) drawn twice or more "
                f"out of {len(drawn_real)}; draw more specs to test this")
        else:
            r["I3_real_components_on_both_sides"] = (
                frac >= 0.5,
                f"{len(both)}/{len(eligible)} = {frac:.3f} of real components drawn "
                f"2+ times appear with both FILE_FAKE=0 and =1 "
                f"({len(drawn_real)} distinct real components drawn)")
    else:
        r["I3_real_components_on_both_sides"] = (True, "skipped: no manifest given")

    # -- I4: file_fake agrees with metrics.dacon ---------------------------- #
    bad = [s.sample_id for s in specs
           if s.file_fake != cell_labels(s.cell)["file_fake"]]
    r["I4_file_fake_matches_metrics"] = (
        not bad, f"{len(bad)} spec(s) disagree with metrics.dacon.file_fake_label")

    # -- I5: no drawn file lies outside the active slice --------------------- #
    if manifest is not None and slice_ is not None:
        allowed = set(manifest.loc[manifest["slice"] == slice_, "file_id"].astype(str))
        leaked = {c.file_id for s in specs for c in s.components
                  if c.file_id not in allowed}
        r["I5_split_safety"] = (
            not leaked,
            f"{len(leaked)} file(s) drawn from outside slice={slice_!r}"
            + (f": {sorted(leaked)[:3]}" if leaked else ""))
    else:
        r["I5_split_safety"] = (True, "skipped: no manifest/slice given")

    # -- I6: labels are a function of the cell alone ------------------------- #
    mismatched = [s.sample_id for s in specs
                  if (s.voice_present, s.music_present, s.voice_fake, s.music_fake)
                  != CELL_TABLE[s.cell]]
    r["I6_labels_come_from_the_cell"] = (
        not mismatched, f"{len(mismatched)} spec(s) whose labels disagree with the cell")

    # -- I7: cells 6 and 7 are never rendered as whole files ----------------- #
    scraped = [s.sample_id for s in specs
               if s.cell in (6, 7) and s.render_mode != "composed"]
    r["I7_cells_6_7_always_composed"] = (
        not scraped, f"{len(scraped)} spec(s) in cells 6/7 not composed")

    # -- I8 (C1): per-head positive rate in [0.2, 0.8], after masking -------- #
    lo, hi = C1_BOUNDS
    pools = {
        "file": (specs, lambda s: s.file_fake == 1),
        "voice": ([s for s in specs if s.voice_present], lambda s: s.voice_fake == 1),
        "music": ([s for s in specs if s.music_present], lambda s: s.music_fake == 1),
        "v_pres": (specs, lambda s: s.voice_present == 1),
        "m_pres": (specs, lambda s: s.music_present == 1),
    }
    rates = {h: _fraction(sum(1 for s in pool if pred(s)), len(pool))
             for h, (pool, pred) in pools.items()}
    out = {h: v for h, v in rates.items() if not lo <= v <= hi}
    r["I8_C1_positive_rates"] = (
        not out,
        "measured " + ", ".join(f"{h}={v:.3f}" for h, v in rates.items())
        + f"; bounds [{lo}, {hi}]" + (f"; OUT: {out}" if out else ""))

    # -- I9 (C2): per-head present-count floor per batch --------------------- #
    starved: Counter = Counter()
    n_batches = 0
    for start in range(0, n - batch_size + 1, batch_size):
        batch = specs[start:start + batch_size]
        n_batches += 1
        for head, key in (("voice", "voice_present"), ("music", "music_present")):
            if sum(getattr(s, key) for s in batch) < min_present:
                starved[head] += 1
    r["I9_C2_present_count_floor"] = (
        not starved,
        f"{sum(starved.values())}/{n_batches} batch(es) below {min_present} present "
        f"samples for a masked head" + (f": {dict(starved)}" if starved else ""))
    return AuditReport(r)


def run_audit(sampler: Sampler, n: int = 20_000, epoch: int = 0, seed: int = 0,
              manifest: pd.DataFrame | None = None, **kw) -> AuditReport:
    """Draw `n` specs and audit them. Cheap: no audio is decoded."""
    specs = list(sampler.epoch_specs(n, epoch=epoch, seed=seed))
    return audit_specs(specs, manifest=manifest, slice_=sampler.slice_, **kw)
