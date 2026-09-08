"""The spec-level invariants, I1-I9 and I21 (docs/pipelines/05 §1).

Critical: these exist because of a measured pattern in this repo: six defects
were found in one review session, **none by a green test suite**, and two
shapes recurred -- a test asserting an *adjacent* quantity, and a component
inherited from a source recipe that was correct in its regime and silently
wrong in ours.

Caveat: every check here measures an **empirically drawn stream**, never the
config it was drawn from. The constraint arithmetic can be right while the
sampler implementing it is wrong, and only the stream catches that. They need
no corpus, no model and no GPU, so they can run before the corpus exists.
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

#: Strong evidence (docs/data/07 E-S2). Above this, neutralise before training
#: anything.
SHORTCUT_AUC_GATE = 0.60


@dataclass
class AuditReport:
    """One row per invariant, with the measured quantity that decided it."""

    results: dict[str, tuple[bool, str]]

    #: A detail beginning with this marks a check that did not run. Critical: a
    #: skipped check must never read as a pass: a review found `I7` printing
    #: PASS for a size floor that was not implemented anywhere.
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
#: as real. Strongly evidenced, and chosen from measurement rather than taste:
#: over 16 seeds the max-over-strata |z| had mean 1.36 and p95 2.89, topping out
#: at 3.62 on a clean corpus. At k = 4 the per-audit false-alarm rate is ~2e-4
#: over the three strata, and every mutation test in the suite still fires at
#: 11-31 sigma.
NOISE_K = 4.0

#: Above this the estimator cannot see anything worth acting on, so the check
#: reports SKIP rather than a pass it did not earn. Critical, and derived
#: rather than picked: for a binary feature `AUC = 0.5 + gap / 2`, so the E-S2
#: shortcut gate of 0.60 is a gap of 0.20. A stratum whose noise floor exceeds
#: that cannot resolve a leak this pipeline would act on.
RESOLVABLE_GAP = 2.0 * (SHORTCUT_AUC_GATE - 0.5)


def _gap_tolerance(x1: int, n1: int, x0: int, n0: int, floor: float,
                   k: float = NOISE_K) -> tuple[float, float]:
    """`(effective tolerance, noise floor)` for `|x1/n1 - x0/n0|`.

    Critical: the flat tolerance this replaces was below its own estimator's
    noise. Both proportions are binomial, so under H0 the gap has ``se =
    sqrt(p(1-p)(1/n1 + 1/n0))`` with `p` pooled -- measured to fit: mean
    |z| 0.72-0.95 against the 0.798 of a standard normal. At `tol = 0.02` and the
    ~1,200 specs cell 3 gets in a 20k draw, `se` is 0.017, so the check tripped
    on 2 of 4 seeds on a corpus with nothing wrong with it. A guardrail that
    cries wolf gets switched off, and then it does not catch the composition
    trap it exists for (docs/pipelines/05).

    Caveat: the floor still binds wherever the estimate is sharp enough to
    honour it. In particular a stratum with no composed samples on either side
    has `p = 0`, hence `se = 0`, hence the full `tol` -- which is exactly the
    shipped `single_composed_rate = 0.0` case for the two single-component
    strata.
    """
    n = n1 + n0
    p = (x1 + x0) / n if n else 0.0
    se = float(np.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n0))) if n1 and n0 else float("inf")
    return max(floor, k * se), k * se


def _feature_frame(specs: Sequence[SampleSpec]) -> tuple[np.ndarray, list[str]]:
    """Every structural knob the sampler controls, as a numeric matrix.

    Critical: includes ``transforms`` **parameters** and the whole
    ``normalize`` draw, not just transform names. A review found that a
    name-only frequency table is blind to a leak carried in a parameter --
    ``rawboost(strength=0.9 if fake else 0.1)`` puts the *name* on both sides
    at exactly equal rate, so the frequency check reports 0.0000 while the
    parameter separates the labels at AUC 1.000. ``normalize`` was read by
    nothing at all, and it is the A-S1 test-chain draw: the single
    highest-leverage stage in the pipeline.
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
        # Caveat: `source_offset_s` is deliberately NOT a feature. It is where
        # inside the source file we start reading -- the model sees rendered
        # audio and cannot observe it, so it cannot be a shortcut. It IS
        # correlated with the source pool's duration statistics, which differ
        # between the fake and real whole-file pools by chance when those pools
        # are small: with 12 real and 9 fake instrumental whole-file rows it
        # reached AUC 0.764 inside the music-only stratum and failed this gate
        # on a corpus with no actual leak. Source-duration imbalance is a
        # manifest property worth knowing about, but it is not a spec-level
        # shortcut.
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

    Caveat: cross-validated, not in-sample: the feature matrix one-hots
    categorical transform and normalize values, and an in-sample fit over many
    sparse columns inflates AUC and manufactures false alarms. A real leak
    survives cross-validation trivially -- the trapped stream this was built
    against scores 1.000 either way.
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


def _auc_se(n1: int, n0: int) -> float:
    """Standard error of an AUC under H0 (A = 0.5): `sqrt((n1+n0+1)/(12 n1 n0))`."""
    if n1 < 1 or n0 < 1:
        return float("inf")
    return float(np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)))


def _metadata_shortcut(specs: Sequence[SampleSpec]) -> tuple[bool, str]:
    """Can structural metadata alone predict FILE_FAKE? (E-S2 / VG2 at spec level.)

    Caveat: ``stratum`` is not a feature. Presence is a legitimate signal the
    model is *asked* to predict, so including it would measure the label space
    rather than our corpus.

    Critical: but the probe runs **per stratum as well as pooled, and takes the
    worst**. Because the model is trained to predict presence, it effectively
    knows the stratum for free -- so a leak that cancels marginally while
    pointing opposite ways inside two strata is fully available to it. A review
    built exactly that: a duration shift of opposite sign in mixed vs non-mixed
    files scored 0.4986 marginally (invisible) and 0.7719 with the interaction.

    Critical: **cell 9 is excluded**, for the same reason it is excluded from
    the mixedness balance: it is always REAL *and* never composed, contributing
    an unfixable correlation. Including it, the strict policy (``f8 = 1``)
    scored 0.6003 and failed this gate -- a false alarm blocking a policy we
    support.
    """
    specs = [s for s in specs if s.cell != 9]
    if not specs:
        return True, AuditReport.SKIP + "no cells 1-8 in the stream"

    y_all = np.array([s.file_fake for s in specs])
    X_all, names = _feature_frame(specs)
    scored: list[tuple[float, str]] = []

    counts: dict[str, tuple[int, int]] = {}
    pooled = _auc_or_none(X_all, y_all)
    if pooled is not None:
        scored.append((pooled, "pooled"))
        counts["pooled"] = (int((y_all == 1).sum()), int((y_all == 0).sum()))
    for stratum in ("voice-only", "music-only", "mixed"):
        idx = [i for i, s in enumerate(specs) if s.stratum == stratum]
        if len(idx) < 200:
            continue
        auc = _auc_or_none(X_all[idx], y_all[idx])
        if auc is not None:
            scored.append((auc, stratum))
            ys = y_all[idx]
            counts[stratum] = (int((ys == 1).sum()), int((ys == 0).sum()))

    if not scored:
        return True, AuditReport.SKIP + "not enough samples to estimate an AUC"
    worst, where = max(scored)

    # Critical: noise-aware, for the same reason I2 is. A fixed 0.60 against a
    # per-probe AUC whose own standard error is ~0.033 in a thin stratum is a
    # false-alarm generator, and taking the MAX over four probes inflates the
    # tail further. A seed sweep caught it: voice-only scored 0.6023 on a clean
    # corpus at n=1500. Under H0 (A = 0.5) the AUC's se is
    # sqrt((n1+n0+1)/(12*n1*n0)).
    n_worst = counts[where]
    se = _auc_se(*n_worst)
    gate = max(SHORTCUT_AUC_GATE, 0.5 + NOISE_K * se)
    detail = ", ".join(f"{w}={a:.4f}" for a, w in sorted(scored, key=lambda t: -t[0]))
    return (worst < gate,
            f"worst metadata-only CV AUC = {worst:.4f} in {where!r} over "
            f"{len(names)} feature(s); gate < {gate:.4f} "
            f"(floor {SHORTCUT_AUC_GATE}, {NOISE_K:g} SE = {NOISE_K * se:.4f} at "
            f"n={n_worst}) [{detail}]")


def audit_specs(specs: Sequence[SampleSpec], manifest: pd.DataFrame | None = None,
                slice_: str | None = None, fold: int | None = None,
                tol: float = 0.02, batch_size: int = 32, min_present: int = 8, marginal_tol: float = 0.02,
                eval_floors: bool = False, class_floor: int = 1_200,
                cell_floor: int = 100, min_families: int = 3,
                batches: Sequence[Sequence[int]] | None = None) -> AuditReport:
    """Run I1-I9 over a drawn stream."""
    if not specs:
        raise ValueError("audit_specs needs at least one spec")
    r: dict[str, tuple[bool, str]] = {}
    n = len(specs)
    fake = [s for s in specs if s.file_fake == 1]
    real = [s for s in specs if s.file_fake == 0]

    # -- I1: P(T | L) = P(T) for every transform NAME ------------------------ #
    # Caveat: names only, and deliberately so -- it is cheap and interpretable,
    # and it localises which transform is skewed. It is NOT sufficient on its
    # own: a review showed `rawboost(strength=0.9 if fake else 0.1)` puts the
    # name on both sides at exactly equal rate, so this reports 0.0000 while
    # the parameter separates the labels at AUC 1.000. I1b covers parameters.
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
        f"caveat: names only -- parameter-level leaks are I1b's job")

    # -- I1b: the metadata shortcut audit, E-S2 at spec level ---------------- #
    # Critical: the joint check. Each balance above can hold individually while
    # a *combination* of structural features still separates the labels -- and
    # nothing else here would see it. This is docs/data/07 E-S2 ("logistic
    # regression on metadata-only features, gate AUC < 0.60"), promoted to VG2,
    # run over specs instead of over decoded audio.
    r["I1b_metadata_shortcut_auc"] = _metadata_shortcut(specs)

    # -- I2: composedness label-independent WITHIN each presence stratum ----- #
    # Caveat: the stratified form. A marginal balance can hold while
    # composedness predicts the label inside a stratum (docs/pipelines/02 §3).
    # Caveat: the tolerance is per stratum and noise-aware (`_gap_tolerance`):
    # the thin strata cannot resolve a flat 0.02 at the draw budgets we use,
    # and a check that fails at random on a clean corpus gets switched off.
    # Strata are ranked by how far each EXCEEDS its own tolerance, not by raw
    # gap, because the tolerances now differ between them.
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
    # Critical: measured, not asserted. The stratified constraint leaves a
    # marginal gap because cell 9 is its own presence stratum with no FAKE
    # counterpart: 0.127 without `balance_marginal_composedness`, 0.253 at a =
    # b = 1. The defence ("composedness is uninformative given presence, and
    # the model is supervised on presence") is sound but lived in three
    # docstrings and was measured nowhere.
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
    # Caveat: noise-aware, like I2/I2c. This is the same
    # difference-of-binomials estimator, and it was left on a flat 0.02 when
    # they were fixed -- a seed sweep caught it immediately at n=1500, seed 1:
    # gap 0.0220 against 0.02, a false alarm on a clean corpus.
    mix_f = sum(1 for s in f8_ if s.stratum == "mixed")
    mix_r = sum(1 for s in r8_ if s.stratum == "mixed")
    mix_tol, mix_noise = _gap_tolerance(mix_f, len(f8_), mix_r, len(r8_), tol)
    r["I2b_mixedness_balance"] = (
        abs(pm_f - pm_r) <= mix_tol,
        f"P(mixed|FAKE) = {pm_f:.4f} vs P(mixed|REAL) = {pm_r:.4f} (cells 1-8), "
        f"gap {abs(pm_f - pm_r):.4f}, tol {mix_tol:.4f} "
        f"(floor {tol}, {NOISE_K:g} SE = {mix_noise:.4f})")

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
        # Caveat: only files drawn at least twice can testify. A file drawn
        # once cannot appear on both sides, and counting it measures the draw
        # budget rather than the sampler -- at 1,500 draws over a 400-file real
        # pool this reported a failure on a corpus with nothing wrong with it.
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
    # Critical: rewritten. The old check compared `s.file_fake` to
    # `cell_labels(s.cell)["file_fake"]` -- but `SampleSpec.file_fake` *is*
    # that expression, so it was unfalsifiable: a brute force over all nine
    # cells found 0 constructible specs that could trip it.
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
        # Critical: now checks the FOLD as well as the slice, and reports which
        # grouping keys the drawn stream actually spans. The documented I5 is
        # "outside the active slice; family/source/speaker/pair/dup constraints
        # hold" -- the first version implemented only the first clause.
        sel = manifest["slice"] == slice_
        if fold is not None:
            sel &= manifest["fold"] == fold
        allowed = set(manifest.loc[sel, "file_id"].astype(str))
        leaked = {c.file_id for s in specs for c in s.components
                  if c.file_id not in allowed}
        where = f"slice={slice_!r}" + (f" fold={fold!r}" if fold is not None else "")

        # Critical: I5 is a TWO-clause statement ("outside the active slice
        # *and* fold"), so a run with `fold=None` must not report a bare PASS
        # for the half it did not evaluate -- `AuditReport.SKIP` exists for
        # exactly this and a review found the frozen evaluation set, the
        # instrument this project trusts over the leaderboard, passing that
        # way.
        #
        # Caveat: whether the fold clause is live is MEASURED, not assumed.
        # After `folds.apply_folds(..., fold=k)` the fold is baked into `slice`
        # and the frame's `fold` column is uniformly `k`, so the clause has
        # nothing left to say and PASS is honest. On an unresolved frame the
        # column spans several folds, the clause is real, and skipping it
        # silently is the failure mode.
        spanned = sorted(manifest.loc[sel, "fold"].dropna().unique().tolist()) \
            if "fold" in manifest.columns else []
        if fold is None and len(spanned) > 1:
            r["I5_split_safety"] = (True, AuditReport.SKIP + (
                f"slice={slice_!r} checked and clean ({len(leaked)} leak(s)), but "
                f"no fold given for a manifest spanning folds {spanned}: the fold "
                f"half of I5 did not run"))
        else:
            resolved = "" if fold is not None else \
                f"; fold clause vacuous (the frame is one fold: {spanned})"
            r["I5_split_safety"] = (
                not leaked,
                f"{len(leaked)} file(s) drawn from outside {where}"
                + (f": {sorted(leaked)[:3]}" if leaked else "") + resolved)
    else:
        r["I5_split_safety"] = (True, AuditReport.SKIP + "no manifest/slice given")

    # -- I6 is deliberately absent ------------------------------------------- #
    # Critical: there used to be an "I6_labels_come_from_the_cell" here
    # comparing the four label properties to CELL_TABLE[s.cell]. Those
    # properties ARE literal indexes into CELL_TABLE (spec.py), so it could not
    # fail -- a brute force over all nine cells found 0 constructible specs
    # that trip it. "Labels are a function of the cell alone" is a property of
    # the TYPE, and it is asserted where it belongs: tests/test_spec.py asserts
    # SampleSpec has no label fields at all. A per-spec loop over derived
    # properties adds nothing but false confidence.

    # -- I7: the VG1 size floors, and the cheap tripwire that used to squat here #
    scraped = [s.sample_id for s in specs
               if s.cell in (6, 7) and s.render_mode != "composed"]
    r["I7a_cells_6_7_always_composed"] = (
        not scraped,
        f"{len(scraped)} spec(s) in cells 6/7 not composed. Caveat: defense-in-depth "
        f"only: SampleSpec refuses to construct one, so this cannot fail today")

    # Critical: the real I7. It previously printed PASS for a check that
    # existed nowhere -- the "adjacent quantity" pattern docs/pipelines/05
    # opens by warning about. VG1 A8/A9: >=1,200 per class per masked head
    # pool, and >=100 per cell. These bind on an EVALUATION stream; a training
    # stream is not required to meet them, so it is reported as skipped, not as
    # a pass.
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

    # -- I21: realized generator diversity in the drawn stream --------------- #
    # Critical: `domain_cap` is a weight over what is PRESENT. It cannot create
    # diversity that the slice does not have, so a fold split leaving TRAIN
    # generator-poor reproduces the DOSS failure (6.4k h naive 3.29% vs 0.2k h
    # balanced 2.77%) with a green audit and superb local CV.
    #
    # Caveat: measured against an ABSOLUTE floor, not against what the slice
    # happens to hold: a monoculture slice trivially realizes 100% of its own
    # two families. Below ~3 families you cannot measure cross-generator
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
        r["I21_generator_diversity"] = (
            not poor,
            "; ".join(parts_) + f"; floor {min_families} per fake role"
            + (f"; POOR: {'; '.join(poor)}" if poor else ""))
    else:
        r["I21_generator_diversity"] = (True, AuditReport.SKIP + "no manifest given")

    # -- I9 (C2): per-head present-count floor per batch --------------------- #
    # Critical: "audit the same order you train in" is the whole assumption,
    # and M4 changed the training order: `training.dataset.training_batches`
    # buckets by duration, so contiguous draw-order slices are batches the
    # optimiser never sees. Pass `batches=` -- the index plan itself -- and
    # this measures what is actually stepped on. Without it the check still
    # cuts draw order, which is correct for an unbucketed loader and is what
    # the eval path uses.
    #
    # Caveat: the trailing partial batch is EXCLUDED, and that is a reversal.
    # It was included on the argument that it is smallest and likeliest to
    # starve a head -- but C2's floor is absolute (the gradient norm scales as
    # 1/sqrt(n), not with the batch fraction), so a short final batch can never
    # satisfy it: a seed sweep found 1/188 batches failing on a clean corpus
    # for exactly this reason. Training uses `drop_last=True`, so that batch is
    # never stepped on; auditing it measures a batch the optimiser will not
    # see.
    starved: Counter = Counter()
    # Caveat: the two branches drop rows for different reasons, so they must
    # not share one phrase. In draw order the remainder is what `drop_last`
    # eats; in a supplied plan it is whatever the plan never indexes -- which
    # is the same thing for a `bucket_batches` plan and NOT the same thing for
    # a partial or overlapping one. Reporting both as "excluded (drop_last)"
    # would be a number that quietly means something else, so each says what it
    # counted.
    if batches is None:
        groups = [specs[start:start + batch_size]
                  for start in range(0, n - batch_size + 1, batch_size)]
        dropped = n % batch_size
        dropped_note = f"{dropped} trailing spec(s) excluded (drop_last)"
        plan = f"draw order, batch_size={batch_size}"
    else:
        bad = [i for b in batches for i in b if not 0 <= i < n]
        if bad:
            raise ValueError(f"batches index outside the spec list: {bad[:3]}")
        groups = [[specs[i] for i in b] for b in batches]
        covered = {i for b in batches for i in b}
        dropped = n - len(covered)
        repeated = sum(len(b) for b in batches) - len(covered)
        dropped_note = f"{dropped} spec(s) the plan never batches"
        if repeated:
            # A spec in two batches is stepped on twice per epoch. C2 is still
            # measured correctly per batch, but the reader must not take the
            # dropped count for a `drop_last` remainder.
            dropped_note += f"; {repeated} spec(s) batched more than once"
        sizes = sorted({len(b) for b in batches})
        plan = f"supplied plan, {len(batches)} batch(es) of {sizes}"
    for batch in groups:
        for head, key in (("voice", "voice_present"), ("music", "music_present")):
            if sum(getattr(s, key) for s in batch) < min_present:
                starved[head] += 1
    # Caveat: the default floor is 8, not 2. `docs/pipelines/02 §4`'s own table
    # puts n=2 at 3.9x the gradient norm of n=32 and n=8 at 2.0x -- a floor of
    # 2 accepts exactly the case C2 exists to prevent.
    r["I9_C2_present_count_floor"] = (
        not starved,
        f"{sum(starved.values())}/{len(groups)} batch(es) below {min_present} present "
        f"samples for a masked head ({plan})"
        + (f": {dict(starved)}" if starved else "")
        + (f"; {dropped_note}" if dropped else ""))
    return AuditReport(r)


def run_audit(sampler: Sampler, n: int = 20_000, epoch: int = 0, seed: int = 0,
              manifest: pd.DataFrame | None = None, **kw) -> AuditReport:
    """Draw `n` specs and audit them. Cheap: no audio is decoded."""
    specs = list(sampler.epoch_specs(n, epoch=epoch, seed=seed))
    return audit_specs(specs, manifest=manifest, slice_=sampler.slice_,
                       fold=sampler.fold, **kw)
