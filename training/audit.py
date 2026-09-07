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

    #: A detail beginning with this marks a check that did not run. 🔴 A skipped
    #: check must never read as a pass: a review found `I7` printing PASS for a
    #: size floor that was not implemented anywhere.
    SKIP = "SKIPPED: "

    @property
    def ok(self) -> bool:
        return all(passed for passed, _ in self.results.values())

    @property
    def skipped(self) -> dict[str, str]:
        return {k: why for k, (_, why) in self.results.items()
                if why.startswith(self.SKIP)}

    @property
    def ran(self) -> dict[str, tuple[bool, str]]:
        return {k: v for k, v in self.results.items() if k not in self.skipped}

    @property
    def failures(self) -> dict[str, str]:
        return {k: why for k, (passed, why) in self.results.items() if not passed}

    def raise_for_status(self) -> None:
        if not self.ok:
            lines = "\n".join(f"  {k}: {why}" for k, why in self.failures.items())
            raise AssertionError(f"spec audit failed:\n{lines}")

    def __str__(self) -> str:
        def status(passed: bool, why: str) -> str:
            return "SKIP" if why.startswith(self.SKIP) else ("PASS" if passed else "FAIL")
        return "\n".join(f"{status(p, why)}  {k}: {why}"
                         for k, (p, why) in sorted(self.results.items()))


def _fraction(num: float, den: float) -> float:
    return num / den if den else float("nan")


#: How many standard errors of the gap a difference must clear before it counts
#: as real. ★ Chosen from measurement, not taste: over 16 seeds the max-over-
#: strata |z| had mean 1.36 and p95 2.89, topping out at 3.62 on a clean corpus.
#: At k = 4 the per-audit false-alarm rate is ~2e-4 over the three strata, and
#: every mutation test in the suite still fires at 11-31 sigma.
NOISE_K = 4.0

#: Above this the estimator cannot see anything worth acting on, so the check
#: reports SKIP rather than a pass it did not earn. 🔴 Derived, not picked: for a
#: binary feature `AUC = 0.5 + gap / 2`, so the E-S2 shortcut gate of 0.60 is a
#: gap of 0.20. A stratum whose noise floor exceeds that cannot resolve a leak
#: this pipeline would act on.
RESOLVABLE_GAP = 2.0 * (SHORTCUT_AUC_GATE - 0.5)


def _gap_tolerance(x1: int, n1: int, x0: int, n0: int, floor: float,
                   k: float = NOISE_K) -> tuple[float, float]:
    """`(effective tolerance, noise floor)` for `|x1/n1 - x0/n0|`.

    🔴 The flat tolerance this replaces was below its own estimator's noise. Both
    proportions are binomial, so under H0 the gap has
    ``se = sqrt(p(1-p)(1/n1 + 1/n0))`` with `p` pooled -- measured to fit: mean
    |z| 0.72-0.95 against the 0.798 of a standard normal. At `tol = 0.02` and the
    ~1,200 specs cell 3 gets in a 20k draw, `se` is 0.017, so the check tripped
    on 2 of 4 seeds on a corpus with nothing wrong with it. A guardrail that
    cries wolf gets switched off, and then it does not catch the composition
    trap it exists for (docs/pipelines/05).

    ⚠️ The floor still binds wherever the estimate is sharp enough to honour it.
    In particular a stratum with no composed samples on either side has `p = 0`,
    hence `se = 0`, hence the full `tol` -- which is exactly the shipped
    `single_composed_rate = 0.0` case for the two single-component strata.
    """
    n = n1 + n0
    p = (x1 + x0) / n if n else 0.0
    se = float(np.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n0))) if n1 and n0 else float("inf")
    return max(floor, k * se), k * se


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
        return True, AuditReport.SKIP + "no cells 1-8 in the stream"

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
        return True, AuditReport.SKIP + "not enough samples to estimate an AUC"
    worst, where = max(scored)
    detail = ", ".join(f"{w}={a:.4f}" for a, w in sorted(scored, key=lambda t: -t[0]))
    return (worst < SHORTCUT_AUC_GATE,
            f"worst metadata-only CV AUC = {worst:.4f} in {where!r} "
            f"over {len(names)} feature(s); gate < {SHORTCUT_AUC_GATE} [{detail}]")


def audit_specs(specs: Sequence[SampleSpec], manifest: pd.DataFrame | None = None,
                slice_: str | None = None, fold: int | None = None,
                tol: float = 0.02, batch_size: int = 32, min_present: int = 8, marginal_tol: float = 0.02,
                eval_floors: bool = False, class_floor: int = 1_200,
                cell_floor: int = 100, min_families: int = 3) -> AuditReport:
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
    # ⚠️ The tolerance is per stratum and noise-aware (`_gap_tolerance`): the
    # thin strata cannot resolve a flat 0.02 at the draw budgets we use, and a
    # check that fails at random on a clean corpus gets switched off. Strata are
    # ranked by how far each EXCEEDS its own tolerance, not by raw gap, because
    # the tolerances now differ between them.
    strat: dict[tuple[str, int], list[int]] = defaultdict(list)
    for s in specs:
        strat[(s.stratum, s.file_fake)].append(int(s.render_mode == "composed"))
    scored_gaps: list[tuple[float, float, float, str]] = []
    unresolvable: list[str] = []
    for stratum in ("voice-only", "music-only", "mixed"):
        f1, f0 = strat.get((stratum, 1)), strat.get((stratum, 0))
        if not f1 or not f0:
            continue
        eff_tol, noise = _gap_tolerance(sum(f1), len(f1), sum(f0), len(f0), tol)
        gap = abs(sum(f1) / len(f1) - sum(f0) / len(f0))
        if noise > RESOLVABLE_GAP:
            unresolvable.append(f"{stratum} (needs {noise:.3f}, "
                                f"n={len(f1)}/{len(f0)})")
            continue
        scored_gaps.append((gap - eff_tol, gap, eff_tol, stratum))
    if not scored_gaps:
        r["I2_stratified_composedness"] = (
            True, AuditReport.SKIP + "no stratum can resolve a gap of "
            f"{RESOLVABLE_GAP:.2f} at these counts: {'; '.join(unresolvable) or 'none drawn'}"
            "; draw more specs")
    else:
        _, worst, worst_tol, worst_where = max(scored_gaps)
        r["I2_stratified_composedness"] = (
            worst <= worst_tol,
            f"worst within-stratum |P(composed|FAKE) - P(composed|REAL)| = {worst:.4f} "
            f"in {worst_where!r}, tol {worst_tol:.4f} (floor {tol}, "
            f"{NOISE_K:g} SE of the gap at these counts)"
            + (f"; unresolvable: {'; '.join(unresolvable)}" if unresolvable else ""))

    # -- I2c: the MARGINAL composedness residual ----------------------------- #
    # 🔴 Measured, not asserted. The stratified constraint leaves a marginal gap
    # because cell 9 is its own presence stratum with no FAKE counterpart: 0.127
    # without `balance_marginal_composedness`, 0.253 at a = b = 1. The defence
    # ("composedness is uninformative given presence, and the model is
    # supervised on presence") is sound but lived in three docstrings and was
    # measured nowhere.
    x_f = sum(s.render_mode == "composed" for s in fake)
    x_r = sum(s.render_mode == "composed" for s in real)
    m_f = _fraction(x_f, len(fake))
    m_r = _fraction(x_r, len(real))
    # Same noise-aware bound as I2. This pool is the whole stream, so the floor
    # binds at any realistic draw budget -- but it is the same estimator, and
    # hard-coding a tolerance below its noise is the defect either way.
    marg_tol, marg_noise = _gap_tolerance(x_f, len(fake), x_r, len(real), marginal_tol)
    r["I2c_marginal_composedness"] = (
        abs(m_f - m_r) <= marg_tol,
        f"P(composed|FAKE) = {m_f:.4f} vs P(composed|REAL) = {m_r:.4f} over ALL "
        f"cells, gap {abs(m_f - m_r):.4f}, tol {marg_tol:.4f} (floor "
        f"{marginal_tol}, {NOISE_K:g} SE = {marg_noise:.4f})")

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
                AuditReport.SKIP + f"only {len(eligible)} real component(s) drawn twice "
                f"or more out of {len(drawn_real)}; draw more specs to test this")
        else:
            r["I3_real_components_on_both_sides"] = (
                frac >= 0.5,
                f"{len(both)}/{len(eligible)} = {frac:.3f} of real components drawn "
                f"2+ times appear with both FILE_FAKE=0 and =1 "
                f"({len(drawn_real)} distinct real components drawn)")
    else:
        r["I3_real_components_on_both_sides"] = (True, AuditReport.SKIP + "no manifest given")

    # -- I4: the drawn components' POOLS must imply the cell's labels -------- #
    # 🔴 Rewritten. The old check compared `s.file_fake` to
    # `cell_labels(s.cell)["file_fake"]` -- but `SampleSpec.file_fake` *is* that
    # expression, so it was unfalsifiable: a brute force over all nine cells
    # found 0 constructible specs that could trip it.
    #
    # This version cross-checks against an INDEPENDENT source: the pool each
    # drawn component actually came from. A sampler that drew a fake-voice
    # component (pool B) for cell 1 (voice-only REAL) would mislabel the sample,
    # and nothing else in the audit would see it.
    if manifest is not None:
        pool_of = manifest.set_index("file_id").pool.to_dict()
        role_of_pool = {"A": ("voice", 0), "B": ("voice", 1),
                        "C": ("music", 0), "D": ("music", 1), "E": ("noise", 0)}
        wrong: list[str] = []
        for s in specs:
            if s.render_mode != "composed":
                continue                      # whole-file rows carry their own cell
            for c in s.components:
                pool = pool_of.get(c.file_id)
                if pool is None or pool not in role_of_pool:
                    continue
                role, fake = role_of_pool[pool]
                if role != c.role:
                    wrong.append(f"{c.file_id}: pool {pool} used as {c.role!r}")
                elif role == "voice" and s.voice_fake is not None and fake != s.voice_fake:
                    wrong.append(f"{c.file_id}: pool {pool} in cell {s.cell} "
                                 f"(voice_fake={s.voice_fake})")
                elif role == "music" and s.music_fake is not None and fake != s.music_fake:
                    wrong.append(f"{c.file_id}: pool {pool} in cell {s.cell} "
                                 f"(music_fake={s.music_fake})")
        r["I4_component_pools_imply_the_labels"] = (
            not wrong,
            f"{len(wrong)} component(s) whose pool contradicts the cell's labels"
            + (f": {wrong[:3]}" if wrong else ""))
    else:
        r["I4_component_pools_imply_the_labels"] = (
            True, AuditReport.SKIP + "no manifest given")

    # -- I5: no drawn file lies outside the active slice --------------------- #
    if manifest is not None and slice_ is not None:
        # 🔴 Now checks the FOLD as well as the slice, and reports which grouping
        # keys the drawn stream actually spans. The documented I5 is "outside the
        # active slice; family/source/speaker/pair/dup constraints hold" -- the
        # first version implemented only the first clause.
        sel = manifest["slice"] == slice_
        if fold is not None:
            sel &= manifest["fold"] == fold
        allowed = set(manifest.loc[sel, "file_id"].astype(str))
        leaked = {c.file_id for s in specs for c in s.components
                  if c.file_id not in allowed}
        where = f"slice={slice_!r}" + (f" fold={fold!r}" if fold is not None else "")
        r["I5_split_safety"] = (
            not leaked,
            f"{len(leaked)} file(s) drawn from outside {where}"
            + (f": {sorted(leaked)[:3]}" if leaked else ""))
    else:
        r["I5_split_safety"] = (True, AuditReport.SKIP + "no manifest/slice given")

    # -- I6 is deliberately absent ------------------------------------------- #
    # 🔴 There used to be an "I6_labels_come_from_the_cell" here comparing the
    # four label properties to CELL_TABLE[s.cell]. Those properties ARE literal
    # indexes into CELL_TABLE (spec.py), so it could not fail -- a brute force
    # over all nine cells found 0 constructible specs that trip it. "Labels are
    # a function of the cell alone" is a property of the TYPE, and it is
    # asserted where it belongs: tests/test_spec.py asserts SampleSpec has no
    # label fields at all. A per-spec loop over derived properties adds nothing
    # but false confidence.

    # -- I7: the VG1 size floors, and the cheap tripwire that used to squat here #
    scraped = [s.sample_id for s in specs
               if s.cell in (6, 7) and s.render_mode != "composed"]
    r["I7a_cells_6_7_always_composed"] = (
        not scraped,
        f"{len(scraped)} spec(s) in cells 6/7 not composed. ⚠️ defense-in-depth "
        f"only: SampleSpec refuses to construct one, so this cannot fail today")

    # 🔴 The real I7. It previously printed PASS for a check that existed
    # nowhere -- the "adjacent quantity" pattern docs/pipelines/05 opens by
    # warning about. VG1 A8/A9: >=1,200 per class per masked head pool, and
    # >=100 per cell. These bind on an EVALUATION stream; a training stream is
    # not required to meet them, so it is reported as skipped, not as a pass.
    if eval_floors:
        short: list[str] = []
        for head, pool, pred in (
                ("voice", [s for s in specs if s.voice_present], lambda s: s.voice_fake == 1),
                ("music", [s for s in specs if s.music_present], lambda s: s.music_fake == 1),
                ("file", specs, lambda s: s.file_fake == 1)):
            pos = sum(1 for s in pool if pred(s))
            neg = len(pool) - pos
            if min(pos, neg) < class_floor:
                short.append(f"{head}: {pos} fake / {neg} real (need {class_floor} each)")
        per_cell = Counter(s.cell for s in specs)
        thin = {c: per_cell.get(c, 0) for c in CELL_TABLE if per_cell.get(c, 0) < cell_floor}
        r["I7_eval_size_floors"] = (
            not short and not thin,
            f"VG1 A8/A9: {'; '.join(short) if short else 'class floors met'}"
            + (f"; cells below {cell_floor}: {thin}" if thin else ""))
    else:
        r["I7_eval_size_floors"] = (
            True, AuditReport.SKIP + "VG1 A8/A9 bind on an evaluation stream; "
            "pass eval_floors=True when auditing val_specs")

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

    # -- I10: realized generator diversity in the drawn stream --------------- #
    # 🔴 `domain_cap` is a weight over what is PRESENT. It cannot create
    # diversity that the slice does not have, so a fold split leaving TRAIN
    # generator-poor reproduces the DOSS failure (6.4k h naive 3.29% vs 0.2k h
    # balanced 2.77%) with a green audit and superb local CV.
    #
    # ⚠️ Measured against an ABSOLUTE floor, not against what the slice happens
    # to hold: a monoculture slice trivially realizes 100% of its own two
    # families. Below ~3 families you cannot measure cross-generator
    # generalization at all, and that is the binding constraint of this
    # competition -- so the default is deliberately low, catching catastrophe
    # rather than tuning balance.
    if manifest is not None:
        fam_of = manifest.set_index("file_id").artifact_family.to_dict()
        by_role: dict[str, Counter] = defaultdict(Counter)
        for spec in specs:
            for c in spec.components:
                fam = fam_of.get(c.file_id)
                if isinstance(fam, str):
                    by_role[c.role][fam] += 1
        parts_: list[str] = []
        poor: list[str] = []
        for role in ("voice", "music"):
            counts = by_role.get(role, Counter())
            n_fam = len(counts)
            if counts:
                w = np.array(list(counts.values()), dtype=float)
                w /= w.sum()
                eff = float(np.exp(-(w * np.log(w)).sum()))
            else:
                eff = 0.0
            parts_.append(f"{role}: {n_fam} families (effective {eff:.1f})")
            if n_fam < min_families:
                poor.append(f"{role} has {n_fam} < {min_families}")
        r["I10_generator_diversity"] = (
            not poor,
            "; ".join(parts_) + f"; floor {min_families} per fake role"
            + (f"; POOR: {'; '.join(poor)}" if poor else ""))
    else:
        r["I10_generator_diversity"] = (True, AuditReport.SKIP + "no manifest given")

    # -- I9 (C2): per-head present-count floor per batch --------------------- #
    # ⚠️ Two stated assumptions. Batches are cut in DRAW order, which is the
    # training order only if the DataLoader does not reshuffle -- audit the same
    # order you train in. And the final partial batch IS included: it is the
    # smallest and therefore the likeliest to starve a head.
    starved: Counter = Counter()
    n_batches = 0
    for start in range(0, n, batch_size):
        batch = specs[start:start + batch_size]
        if len(batch) < 2:
            continue
        n_batches += 1
        for head, key in (("voice", "voice_present"), ("music", "music_present")):
            if sum(getattr(s, key) for s in batch) < min_present:
                starved[head] += 1
    # ⚠️ The default floor is 8, not 2. `docs/pipelines/02 §4`'s own table puts
    # n=2 at 3.9x the gradient norm of n=32 and n=8 at 2.0x -- a floor of 2
    # accepts exactly the case C2 exists to prevent.
    r["I9_C2_present_count_floor"] = (
        not starved,
        f"{sum(starved.values())}/{n_batches} batch(es) below {min_present} present "
        f"samples for a masked head" + (f": {dict(starved)}" if starved else ""))
    return AuditReport(r)


def run_audit(sampler: Sampler, n: int = 20_000, epoch: int = 0, seed: int = 0,
              manifest: pd.DataFrame | None = None, **kw) -> AuditReport:
    """Draw `n` specs and audit them. Cheap: no audio is decoded."""
    specs = list(sampler.epoch_specs(n, epoch=epoch, seed=seed))
    return audit_specs(specs, manifest=manifest, slice_=sampler.slice_,
                       fold=sampler.fold, **kw)
