#!/usr/bin/env python3
"""The scalar threads of docs/processing/01 §3 that need no draw and no decode.

  S1  B1 pair tests (§3.2 / §3.3 / §3.4): same utterance, real vs vocoded --
      does the vocoder change level, crest, leading silence, DC, sub-1 kHz energy?
  S2  DC / high-pass (§3.3): dc_offset and band 0 across the corpus, within CFAD,
      per CFAD generator; the vocoder difference energy above each cutoff.
  S3  Vectors (§3.8): LTAS / mel moments per head under both holdouts; per-source
      identifiability (adversarial validation within a label) -> aug_strength.
  S4  Families (§3.6): the WaveFake correlation clusters; music-family count;
      hours per domain_key; Korean.
  S5  Manifest actions (§3.5): hours moved per action; the 4 s floor per pool.
  S6  Codec census (§3.7): container / bit_rate per pool; chain bandwidth per source.

    V=/data/project/private/dacon-venvs/dacon311/bin/python
    $V scripts/strategy/scalar_threads.py            # writes eda/out/_strategy/
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eda.analyze.screens import generator_key                      # noqa: E402
from eda.analyze.shortcut import head_labels, univariate_auc       # noqa: E402
from eda.config import load_eda_config                             # noqa: E402
from eda.driver import load_files, load_signal, load_vectors       # noqa: E402
from eda.extract.vectors import MEL_HZ_MAX, MEL_HZ_MIN, N_MELS, _hz_to_mel, _mel_to_hz  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

OUT = ROOT / "eda" / "out" / "_strategy"
SEED = 0


def cv_auc(x, y, groups=None, k=5):
    minority = int(np.bincount(y).min())
    k = max(2, min(k, minority))
    if groups is not None:
        ng = len(np.unique(groups))
        if ng < 2:
            return float("nan")
        splitter, args = StratifiedGroupKFold(min(k, ng) if ng >= 5 else 2), (x, y, groups)
    else:
        splitter, args = StratifiedKFold(k, shuffle=True, random_state=SEED), (x, y)
    scores = []
    for tr, te in splitter.split(*args):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
        m.fit(x[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(x[te])[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# --------------------------------------------------------------------------- #
# S1 -- the B1 pair tests
# --------------------------------------------------------------------------- #

def s1_pair_tests(shared: Path) -> pd.DataFrame:
    section("S1  B1 pair tests: same utterance, real vs vocoded (chain plane)")
    sig = pd.read_parquet(shared / "b1" / "b1_signal.parquet")
    pairs = pd.read_parquet(shared / "b1" / "b1_pairs.parquet")
    df = pairs.merge(sig, on="file_id", suffixes=("", "_sig"))
    real = df[df.role == "real"].set_index("utterance")
    cols = {"rms_dbfs_chain": "rms", "crest_factor_db_chain": "crest",
            "peak_dbfs_chain": "peak", "lead_silence_s_chain": "lead_silence",
            "tail_silence_s_chain": "tail_silence", "dc_offset_chain": "dc_abs",
            "band_energy_0_chain": "band0_0-1kHz", "clipping_ratio_chain": "clipping"}
    rows = []
    for voc, g in df[df.role != "real"].groupby("role"):
        g = g.set_index("utterance")
        r = real.reindex(g.index)
        for col, name in cols.items():
            a, b = g[col].to_numpy(dtype=float), r[col].to_numpy(dtype=float)
            if name == "dc_abs":
                a, b = np.abs(a), np.abs(b)
            d = a - b
            ok = np.isfinite(d)
            d = d[ok]
            # paired AUC: fraction of pairs where fake > real (ties half)
            pauc = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))
            rows.append({"vocoder": voc.replace("ljspeech_", ""), "feature": name, "n": int(len(d)),
                         "delta_median": float(np.median(d)),
                         "delta_p25": float(np.percentile(d, 25)),
                         "delta_p75": float(np.percentile(d, 75)),
                         "paired_auc": max(pauc, 1 - pauc)})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index="vocoder", columns="feature", values="paired_auc").round(3)
    print("paired AUC (|fraction of same-utterance pairs where fake > real|, 0.5 = no effect)")
    print(piv.to_string())
    print("\nmedian delta fake - real")
    print(out.pivot_table(index="vocoder", columns="feature", values="delta_median").round(3).to_string())
    out.to_parquet(OUT / "b1_pair_tests.parquet", index=False)
    return out


# --------------------------------------------------------------------------- #
# S2 -- DC and the high-pass
# --------------------------------------------------------------------------- #

def s2_dc_highpass(signal: pd.DataFrame, shared: Path) -> None:
    section("S2  DC offset and band 0 (0-1 kHz): fingerprint or cue?")
    v = signal[signal.pool.isin(["A", "B"])].copy()
    y = head_labels(v, "voice_fake")
    v = v[y.notna()]; y = y[y.notna()].astype(int).to_numpy()
    feats = {"|dc_offset|": v["dc_offset_chain"].abs().to_numpy(),
             "band_energy_0 (0-1 kHz share)": v["band_energy_0_chain"].to_numpy(),
             "crest_factor_db": v["crest_factor_db_chain"].to_numpy(),
             "rms_dbfs": v["rms_dbfs_chain"].to_numpy()}
    groups = v["source_name"].to_numpy()
    rows = []
    for name, x in feats.items():
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        rows.append({"feature": name, "scope": "corpus A vs B", "n": len(y),
                     "auc_univariate": univariate_auc(x, y),
                     "auc_source_grouped": cv_auc(x[:, None], y, groups)})
        cf = v["source_name"].isin(["cfad-real", "cfad-fake"]).to_numpy()
        rows.append({"feature": name, "scope": "within CFAD (real vs fake)", "n": int(cf.sum()),
                     "auc_univariate": univariate_auc(x[cf], y[cf]), "auc_source_grouped": np.nan})
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))

    # per CFAD generator vs cfad-real, on |dc| and band0
    gen = generator_key(v)
    cf = v[v.source_name.isin(["cfad-real", "cfad-fake"])].copy()
    cf["gen"] = gen[cf.index].fillna("cfad-real")
    real = cf[cf.gen == "cfad-real"]
    rows = []
    for g, sub in cf[cf.gen != "cfad-real"].groupby("gen"):
        for feat in ("dc_offset_chain", "band_energy_0_chain", "crest_factor_db_chain"):
            x = np.concatenate([np.abs(sub[feat].to_numpy(float)), np.abs(real[feat].to_numpy(float))])
            yy = np.concatenate([np.ones(len(sub), int), np.zeros(len(real), int)])
            rows.append({"generator": g, "n_fake": len(sub), "feature": feat,
                         "auc_vs_cfad_real": univariate_auc(np.nan_to_num(x), yy)})
    per = pd.DataFrame(rows).pivot_table(index="generator", columns="feature", values="auc_vs_cfad_real").round(3)
    print("\nper CFAD generator vs cfad-real (same publisher, real vs one vocoder):")
    print(per.to_string())
    per.to_parquet(OUT / "dc_per_cfad_generator.parquet")

    # the vocoder difference energy above each cutoff, from the B1 difference image
    diff = pd.read_parquet(shared / "b1" / "b1_difference_chain.parquet")
    edges = _mel_to_hz(np.linspace(_hz_to_mel(MEL_HZ_MIN), _hz_to_mel(MEL_HZ_MAX), N_MELS + 2))
    centres = edges[1:-1]
    rows = []
    for voc in diff.columns:
        d2 = diff[voc].to_numpy(float) ** 2
        total = d2.sum()
        for cut in (0, 20, 40, 72, 100, 150, 300):
            rows.append({"vocoder": voc.replace("ljspeech_", ""), "cutoff_hz": cut,
                         "difference_energy_above_cutoff": float(d2[centres >= cut].sum() / total)})
    hp = pd.DataFrame(rows).pivot_table(index="vocoder", columns="cutoff_hz",
                                       values="difference_energy_above_cutoff").round(3)
    print("\nfraction of each vocoder's real-fake LTAS difference energy that lies ABOVE a cutoff"
          f" (band centres: first bands at {np.round(centres[:4], 1)} Hz):")
    print(hp.to_string())
    hp.to_parquet(OUT / "highpass_surviving_difference.parquet")


# --------------------------------------------------------------------------- #
# S3 -- the vectors
# --------------------------------------------------------------------------- #

def s3_vectors(cfg, signal: pd.DataFrame) -> None:
    section("S3  LTAS / mel-moment vectors: per head under both holdouts; per-source identifiability")
    parts = ["A", "B", "C", "D", "E", "cell8"]
    blocks, ids = [], []
    for p in parts:
        v = load_vectors(cfg, p)
        blocks.append(np.hstack([v["ltas_chain"], v["mel_band_skew_chain"], v["mel_band_kurt_chain"]]))
        ids.append(np.asarray(v["file_id"], dtype=object))
    X = np.vstack(blocks); fid = np.concatenate(ids)
    pos = pd.Series(np.arange(len(fid)), index=fid)
    sig = signal[signal.file_id.isin(pos.index)].copy()
    Xs = X[pos[sig.file_id].to_numpy()]
    ok = np.isfinite(Xs).all(axis=1)
    sig, Xs = sig[ok], Xs[ok]
    Xl = Xs[:, :N_MELS]
    rows = []
    for head in ("music_fake", "voice_fake", "music_present", "voice_present"):
        y = head_labels(sig, head)
        m = y.notna().to_numpy()
        yy = y[m].astype(int).to_numpy(); groups = sig.loc[m, "source_name"].to_numpy()
        for name, xx in (("ltas[128]", Xl[m]), ("ltas+skew+kurt[384]", Xs[m])):
            rows.append({"head": head, "features": name, "n": int(m.sum()),
                         "auc": cv_auc(xx, yy), "auc_source_grouped": cv_auc(xx, yy, groups),
                         "n_sources": int(len(np.unique(groups)))})
    tab = pd.DataFrame(rows); print(tab.round(3).to_string(index=False))
    tab.to_parquet(OUT / "vector_audit.parquet", index=False)

    # per-source identifiability within a label (adversarial validation within a pool)
    rows = []
    for pool, g in sig.groupby("pool"):
        if g.source_name.nunique() < 2:
            continue
        xx = Xl[g.index.map(lambda i: sig.index.get_loc(i))]
        for src in sorted(g.source_name.unique()):
            yy = (g.source_name == src).astype(int).to_numpy()
            rows.append({"pool": pool, "source": src, "n": int(yy.sum()),
                         "auc_one_vs_rest_in_pool": cv_auc(xx, yy)})
    ident = pd.DataFrame(rows)
    print("\nper-source one-vs-rest AUC on LTAS within its pool (1.0 = the archive is nameable from 128 numbers):")
    print(ident.round(3).to_string(index=False))
    ident.to_parquet(OUT / "source_identifiability.parquet", index=False)


# --------------------------------------------------------------------------- #
# S4 -- families and domains
# --------------------------------------------------------------------------- #

def s4_families(files: pd.DataFrame, shared: Path) -> None:
    section("S4  artifact families and domain hours")
    for plane in ("native", "chain"):
        corr = pd.read_parquet(shared / "b1" / f"b1_family_correlation_{plane}.parquet")
        names = [c.replace("ljspeech_", "") for c in corr.columns]
        d = 1.0 - corr.to_numpy(float)
        np.fill_diagonal(d, 0.0); d = np.clip((d + d.T) / 2, 0, None)
        Z = linkage(squareform(d, checks=False), method="average")
        print(f"\nWaveFake vocoder clusters, {plane} plane, average linkage on 1 - r:")
        for thr in (0.4, 0.5, 0.6):
            lab = fcluster(Z, t=thr, criterion="distance")
            clusters = {}
            for n, l in zip(names, lab):
                clusters.setdefault(int(l), []).append(n)
            print(f"  cut at r >= {1 - thr:.1f}: {sorted(clusters.values(), key=len, reverse=True)}")

    gen = generator_key(files)
    f = files.copy(); f["generator"] = gen
    f["domain_key"] = f["source_name"] + "|" + f["generator"].fillna(f["group_key"]).astype(str)
    f["hours"] = f["duration_s"] / 3600
    dom = f.groupby(["pool", "source_name", "domain_key"], dropna=False).agg(
        files=("file_id", "size"), hours=("hours", "sum")).reset_index()
    print("\ndomains (source x generator) with > 500 files -- where domain_cap=500 bites:")
    print(dom[dom.files > 500].sort_values("hours", ascending=False).round(2).to_string(index=False))
    print(f"\ndomains total: {len(dom)}; over 500 files: {int((dom.files > 500).sum())}")
    dom.to_parquet(OUT / "domain_hours.parquet", index=False)

    music = f[f.pool.eq("D") | f.cell.astype(str).eq("8")]
    mf = music.groupby(["source_name", "domain_key"]).agg(files=("file_id", "size"), hours=("hours", "sum"))
    print("\nmusic-fake domains (pool D + cell 8) -- the candidate music families:")
    print(mf.round(1).to_string())

    ko = f[f.source_name.eq("zeroth-korean") | f.path.str.contains("/ko/", regex=False)]
    print("\nKorean: " + "; ".join(f"{s}: {n} files / {h:.2f} h" for s, n, h in
          ko.groupby("source_name").agg(n=("file_id", "size"), h=("hours", "sum")).itertuples()))


# --------------------------------------------------------------------------- #
# S5 -- the manifest actions and the 4 s floor
# --------------------------------------------------------------------------- #

def s5_manifest(files: pd.DataFrame, signal: pd.DataFrame, shared: Path) -> None:
    section("S5  G-EDA6 actions in hours; the 4 s floor per pool")
    wl = pd.read_parquet(shared / "reassignment_worklist.parquet")
    wl = wl.merge(files[["file_id", "duration_s"]], on="file_id", how="left", suffixes=("", "_m"))
    act = wl.groupby(["action", "source_name", "pool"]).agg(
        files=("file_id", "size"), hours=("duration_s_m", lambda s: s.sum() / 3600)).reset_index()
    print(act.round(2).to_string(index=False))
    measured = signal.groupby("source_name").size()
    census = files.groupby("source_name").size()
    cov = pd.DataFrame({"measured": measured, "census": census}).fillna(0).astype(int)
    cov["coverage"] = (cov.measured / cov.census).round(3)
    print("\nVAD coverage per source (rows measured / rows in census):")
    print(cov.to_string())

    f = files.copy(); f["hours"] = f["duration_s"] / 3600
    part = f["pool"].fillna("cell" + f["cell"].astype(str))
    floor = f.groupby(part).apply(lambda g: pd.Series({
        "files": len(g), "hours": g.hours.sum(),
        "files_under_4s": int((g.duration_s < 4).sum()),
        "frac_under_4s": float((g.duration_s < 4).mean()),
        "hours_under_4s": g.hours[g.duration_s < 4].sum(),
        "files_2_to_4s": int(((g.duration_s >= 2) & (g.duration_s < 4)).sum()),
        "hours_2_to_4s": g.hours[(g.duration_s >= 2) & (g.duration_s < 4)].sum(),
    }))
    print("\nthe 4 s floor at census scale (H3), and what a 2 s component floor would recover:")
    print(floor.round(3).to_string())
    floor.to_parquet(OUT / "four_second_floor.parquet")


# --------------------------------------------------------------------------- #
# S6 -- codec census
# --------------------------------------------------------------------------- #

def s6_codec(files: pd.DataFrame, signal: pd.DataFrame) -> None:
    section("S6  container / bit rate per pool; chain-plane bandwidth per source")
    f = files.copy()
    part = f["pool"].fillna("cell" + f["cell"].astype(str))
    tab = f.groupby([part, "container"]).agg(
        files=("file_id", "size"), bit_rate_median_kbps=("bit_rate", lambda s: s.median() / 1000),
        orig_sr_mode=("orig_sr", lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)).reset_index()
    print(tab.round(1).to_string(index=False))
    bw = signal.groupby(["pool", "source_name"], dropna=False).agg(
        bandwidth_chain_median=("effective_bandwidth_hz_chain", "median"),
        near_nyquist_chain_median=("near_nyquist_ratio_chain", "median"),
        bandwidth_native_median=("effective_bandwidth_hz_native", "median")).round(3)
    print("\nchain-plane effective bandwidth per source (what survives the 16 kHz chain):")
    print(bw.to_string())
    bw.to_parquet(OUT / "bandwidth_per_source.parquet")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_eda_config(ROOT / "configs" / "eda.yaml")
    shared = cfg.out / "_shared"
    files = load_files(cfg)
    signal = load_signal(cfg)
    s1_pair_tests(shared)
    s2_dc_highpass(signal, shared)
    s3_vectors(cfg, signal)
    s4_families(files, shared)
    s5_manifest(files, signal, shared)
    s6_codec(files, signal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
