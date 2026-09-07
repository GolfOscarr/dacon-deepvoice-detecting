#!/usr/bin/env python3
"""Verify the properties of the DACON 236749 metric asserted in docs/validation/.

Every number in docs/validation/02-metric-harness.md and 05-lb-probe-plan.md is produced here.
Run:  .venv/bin/python scripts/verify_metric.py
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score

W = dict(file=0.45, music=0.27, voice=0.18, vp=0.05, mp=0.05)


def eer(y_true, y_score):
    """The official estimator, transcribed from the competition page."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def total(eer_f, eer_v, eer_m, auc_vp, auc_mp):
    ads = 0.5 * (1 - eer_f) + 0.2 * (1 - eer_v) + 0.3 * (1 - eer_m)
    cps = 0.5 * auc_vp + 0.5 * auc_mp
    return 0.9 * ads + 0.1 * cps


def gaussian_scores(n_pos, n_neg, target_eer, rng):
    """Scores whose population EER equals `target_eer`."""
    d = 2 * norm.isf(target_eer)
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    s = np.r_[rng.normal(d, 1, n_pos), rng.normal(0, 1, n_neg)]
    return y, s


def check_constant_is_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1200)
    for c in (0.0, 0.123, 0.5, 1.0):
        s = np.full(1200, c)
        assert eer(y, s) == 0.5, c
        assert roc_auc_score(y, s) == 0.5, c
    assert total(0.5, 0.5, 0.5, 0.5, 0.5) == 0.5
    print("[OK] all-constant submission scores exactly 0.5000 (any constant, any column)")


def check_decomposition():
    """4 submissions recover 3 EERs + CPS exactly. docs/validation/05."""
    eF, eV, eM, aV, aM = 0.132, 0.181, 0.294, 0.973, 0.961
    s_f = total(eF, .5, .5, .5, .5)
    s_v = total(.5, eV, .5, .5, .5)
    s_m = total(.5, .5, eM, .5, .5)
    s_full = total(eF, eV, eM, aV, aM)

    r_f = 0.5 - (s_f - 0.5) / W["file"]
    r_v = 0.5 - (s_v - 0.5) / W["voice"]
    r_m = 0.5 - (s_m - 0.5) / W["music"]
    ads = 0.5 * (1 - r_f) + 0.2 * (1 - r_v) + 0.3 * (1 - r_m)
    r_cps = (s_full - 0.9 * ads) / 0.1

    for got, want in ((r_f, eF), (r_v, eV), (r_m, eM), (r_cps, 0.5 * (aV + aM))):
        assert abs(got - want) < 1e-12, (got, want)
    print(f"[OK] decomposition exact: EER {r_f:.3f}/{r_v:.3f}/{r_m:.3f}  CPS {r_cps:.3f}")


def check_monotone_invariance():
    rng = np.random.default_rng(2)
    y, s = gaussian_scores(500, 500, 0.12, rng)
    p = norm.cdf(s)
    base = eer(y, p)
    for name, f in (("x^3", lambda x: x ** 3),
                    ("logit", lambda x: np.log(np.clip(x, 1e-12, 1 - 1e-12) / (1 - np.clip(x, 1e-12, 1 - 1e-12)))),
                    ("affine", lambda x: 3 * x + 7)):
        assert abs(eer(y, f(p)) - base) < 1e-12, name
    print("[OK] EER invariant under strictly monotone transforms")


def check_aggregation():
    """Mean-of-folds vs pooled-OOF. docs/validation/02 section 4."""
    rng = np.random.default_rng(11)
    d = 2 * norm.isf(0.10)
    fold_eers, ys, raw, ranked = [], [], [], []
    for _ in range(5):
        yk = np.r_[np.ones(400), np.zeros(400)]
        sk = np.r_[rng.normal(d, 1, 400), rng.normal(0, 1, 400)]
        a, b = rng.uniform(0.5, 2.0), rng.uniform(-2, 2)   # harmless per-fold monotone rescale
        obs = a * sk + b
        fold_eers.append(eer(yk, obs))
        ys.append(yk); raw.append(obs)
        ranked.append((np.argsort(np.argsort(obs)) + 0.5) / len(obs))
    y = np.concatenate(ys)
    m_fold = float(np.mean(fold_eers))
    m_raw = eer(y, np.concatenate(raw))
    m_rank = eer(y, np.concatenate(ranked))
    assert abs(m_fold - 0.10) < 0.02, m_fold
    assert m_raw > m_fold + 0.05, (m_raw, m_fold)          # pooling raw scores is badly wrong
    assert abs(m_rank - m_fold) < 0.02, (m_rank, m_fold)
    print(f"[OK] aggregation: mean-of-folds {m_fold:.4f} (sd {np.std(fold_eers):.4f}) | "
          f"pooled-raw {m_raw:.4f} WRONG | pooled-rank {m_rank:.4f}")


def check_rollup_linearity():
    """Score is linear in its components, so roll-up order is irrelevant."""
    rng = np.random.default_rng(5)
    F = rng.uniform(0.05, 0.35, (5, 5))
    per_fold = [total(*F[i]) for i in range(5)]
    assert np.isclose(np.mean(per_fold), total(*F.mean(axis=0)), atol=1e-12)
    print("[OK] mean(Score per fold) == Score(mean of components)")


def check_invariances():
    """What EER is and is not invariant to. docs/validation/02 section 5."""
    rng = np.random.default_rng(13)
    n = 4000
    y, s = gaussian_scores(n, n, 0.10, rng)
    base = eer(y, s)
    for keep in (0.5, 0.2):                                 # class prevalence
        idx = np.r_[np.arange(n), n + rng.choice(n, int(n * keep), replace=False)]
        assert abs(eer(y[idx], s[idx]) - base) < 0.02, keep
    print(f"[OK] EER invariant to class prevalence (1:1 -> 1:0.2, base {base:.4f})")

    neg = rng.normal(0, 1, 3000)                            # within-class composition
    easy = rng.normal(2 * norm.isf(0.03), 1, 3000)
    hard = rng.normal(2 * norm.isf(0.30), 1, 3000)
    yy = np.r_[np.ones(3000), np.zeros(3000)]
    out = []
    for f_hard in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = int(3000 * f_hard)
        out.append(eer(yy, np.r_[np.r_[hard[:k], easy[:3000 - k]], neg]))
    assert out[-1] > out[0] + 0.2, out                      # composition dominates
    print("[OK] EER NOT invariant to cell composition within a class: "
          + " -> ".join(f"{v:.4f}" for v in out))


def table_sampling_noise(reps=1000):
    print("\nEER sampling noise — 95% CI half-width in percentage points "
          f"({reps} reps, balanced classes)")
    print(f"{'n/class':>9} {'EER=0.20':>9} {'0.10':>7} {'0.05':>7}")
    rng = np.random.default_rng(7)
    for n in (150, 300, 600, 1200, 3000):
        row = []
        for t in (0.20, 0.10, 0.05):
            v = np.array([eer(*gaussian_scores(n, n, t, rng)) for _ in range(reps)])
            assert abs(v.mean() - t) < 0.002, (n, t, v.mean())   # estimator is unbiased
            row.append(1.96 * v.std() * 100)
        print(f"{n:>9} {row[0]:>8.1f}  {row[1]:>6.1f}  {row[2]:>6.1f}")


def table_paired_power(reps=400):
    """Why comparisons must be paired. docs/validation/02."""
    print(f"\nResolving a true 1.0-point gap (EER 0.100 vs 0.090), {reps} reps")
    print(f"{'n/class':>9} {'rho':>6} {'sd(diff)':>10} {'P(correct sign)':>17}")
    rng = np.random.default_rng(3)
    for n in (600, 1200, 3000):
        for rho in (0.0, 0.8, 0.95):
            d = []
            for _ in range(reps):
                zc = rng.normal(0, 1, 2 * n)
                a = np.sqrt(rho) * zc + np.sqrt(1 - rho) * rng.normal(0, 1, 2 * n)
                b = np.sqrt(rho) * zc + np.sqrt(1 - rho) * rng.normal(0, 1, 2 * n)
                y = np.r_[np.ones(n), np.zeros(n)]
                a += np.r_[np.full(n, 2 * norm.isf(0.10)), np.zeros(n)]
                b += np.r_[np.full(n, 2 * norm.isf(0.09)), np.zeros(n)]
                d.append(eer(y, b) - eer(y, a))
            d = np.array(d)
            print(f"{n:>9} {rho:>6.2f} {d.std():>10.4f} {(d < 0).mean():>16.0%}")


def table_saturation():
    """Rounding is harmless; saturation at the operating point is not. docs/validation/02."""
    rng = np.random.default_rng(1)
    y, s = gaussian_scores(600, 600, 0.10, rng)
    p = norm.cdf(s - norm.isf(0.10))
    print(f"\nRounding the submitted probabilities (baseline EER {eer(y, p):.4f})")
    for dp in (6, 4, 3, 2, 1):
        print(f"   round to {dp} dp -> EER {eer(y, np.round(p, dp)):.4f}")
    print("\nSaturating the extremes (ties destroy ranking near the operating point)")
    for frac in (0.0, 0.1, 0.3, 0.5, 0.8):
        q = p.copy()
        k = int(frac * len(p))
        if k:
            q[np.argsort(-p)[:k]] = 1.0
            q[np.argsort(p)[:k]] = 0.0
        print(f"   top/bottom {frac:>4.0%} saturated -> EER {eer(y, q):.4f}")


if __name__ == "__main__":
    check_constant_is_half()
    check_decomposition()
    check_monotone_invariance()
    check_aggregation()
    check_rollup_linearity()
    check_invariances()
    table_sampling_noise()
    table_paired_power()
    table_saturation()
