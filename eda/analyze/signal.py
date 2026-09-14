"""The S tier, read: what 58,885 decoded files say about processing.

docs/EDA/09 step 6. The S tier is measured and until now nothing consumed it --
`eda analyze` reads `files.parquet` and stops. This module is the other side:
every table here exists to settle one processing decision, and each says which.

🔴 **Everything is paired.** `load_planes` decodes each file once and measures
it twice -- as published (`native`) and after the 16 kHz chain (`chain`) -- so
every row is its own control. A native-minus-chain difference is what the
competition's resample *removes*, measured on identical content rather than
inferred from two populations that differ in content too. That is the design
[00 section 4c](../../docs/EDA/00-harness.md) established on pool E; this runs
it on the whole corpus.

⚠️ **`nyquist_hz` and `orig_sr` are not acoustics.** The chain plane's Nyquist
is 8000 for every row by construction, and the native plane's is the published
sample rate divided by two -- which is a metadata column wearing a signal
column's name. Including it in a chain-plane audit would re-measure X1's
confound and report it as an acoustic finding. `CHAIN_EXCLUDED` names them and
`plane_features` drops them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda.analyze.shortcut import METADATA_FEATURES
from eda.planes import CHAIN, NATIVE, PLANES

__all__ = ["CHAIN_EXCLUDED", "SIGNAL_SCALARS", "TEST_MAX_S", "TEST_MIN_S",
           "DUPLICATE_COSINE", "bandwidth_report", "content_fingerprints",
           "content_similarity_profile", "duration_report", "level_report",
           "near_duplicates", "plane_features", "paired", "signal_audit"]

#: The competition's test set: 1,200 files, **4-60 s, 16 kHz**
#: (docs/competition/01). Every duration statistic here is read against it,
#: because a pool whose files are all 30 s is a pool the model will never see in
#: that form.
TEST_MIN_S = 4.0
TEST_MAX_S = 60.0

#: Every scalar the S tier emits per plane, without its `_native` / `_chain`
#: suffix. Grouped by the extractor that owns it so a column added there is
#: obvious here.
SIGNAL_SCALARS = {
    "level": ("peak_dbfs", "rms_dbfs", "crest_factor_db", "dc_offset",
              "clipping_ratio", "stat_t_rms", "stat_t_std", "stat_t_var",
              "stat_t_pwr"),
    "timing": ("duration_s_decoded", "silence_ratio", "lead_silence_s",
               "tail_silence_s", "n_valid_spans_ge_4s", "longest_valid_span_s"),
    "spectral": ("effective_bandwidth_hz", "near_nyquist_ratio", "hf_ratio_8k",
                 "spectral_flatness", "spectral_centroid_hz", "nyquist_hz",
                 *(f"band_energy_{i}" for i in range(8))),
    "vector": ("mel_bands_flat",),
}

#: 🔴 Columns that are the file's *published rate* in disguise, and therefore
#: must not enter an audit that asks what the **chain** leaks.
#:
#: `nyquist_hz_chain` is 8000.0 on every row -- a constant, harmless but useless.
#: `nyquist_hz_native` is `orig_sr / 2`, which X1b already showed separates the
#: corpus on its own. An audit that included it would rediscover the metadata
#: confound and report it as acoustic evidence.
CHAIN_EXCLUDED = ("nyquist_hz",)


def plane_features(plane: str) -> dict[str, tuple[str, ...]]:
    """The feature spec for one plane, in `build_design`'s shape.

    Numeric only: every S-tier scalar is a float, and the `*_ok` flags are not
    features -- a row whose extractor failed is a row with no measurement, not
    a row with an informative measurement.
    """
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; have {list(PLANES)}")
    numeric = tuple(f"{c}_{plane}"
                    for group in SIGNAL_SCALARS.values() for c in group
                    if c not in CHAIN_EXCLUDED)
    return {"numeric": numeric, "boolean": (), "categorical": ()}


def paired(signal: pd.DataFrame, column: str) -> pd.DataFrame:
    """`native`, `chain` and their difference for one base column.

    ⚠️ Rows where either plane failed are dropped, and the count is on the
    frame's `attrs`. Keeping them would put a NaN difference beside a real one
    and make the median quietly a median of whatever survived.
    """
    native, chain = f"{column}_{NATIVE}", f"{column}_{CHAIN}"
    missing = [c for c in (native, chain) if c not in signal.columns]
    if missing:
        raise KeyError(
            f"{missing} not in the signal table. Its paired columns are "
            f"{sorted({c.rsplit('_', 1)[0] for c in signal.columns})}")
    out = pd.DataFrame({
        "file_id": signal["file_id"], "source_name": signal["source_name"],
        "partition": signal["partition"],
        NATIVE: pd.to_numeric(signal[native], errors="coerce"),
        CHAIN: pd.to_numeric(signal[chain], errors="coerce"),
    })
    before = len(out)
    out = out.dropna(subset=[NATIVE, CHAIN])
    out["delta"] = out[NATIVE] - out[CHAIN]
    out.attrs["dropped"] = before - len(out)
    return out


def _q(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else float("nan")


def bandwidth_report(signal: pd.DataFrame) -> pd.DataFrame:
    """Per source: what the 16 kHz chain removes. R1's question, measured.

    **Decides**: whether a resampler-skirt transform is worth one cheap
    operation, and whether any above-8 kHz feature can survive into the model.

    `hf_ratio_8k_native` is the fraction of a file's energy the chain throws
    away; it is ~0 on the chain plane by construction, so the native column is
    the whole statement. `bandwidth_lost_hz` is paired -- the same file before
    and after -- so it is a property of the transform, not of the population.
    """
    rows = []
    bw = paired(signal, "effective_bandwidth_hz")
    hf = paired(signal, "hf_ratio_8k").set_index("file_id")[NATIVE]
    cen = paired(signal, "spectral_centroid_hz")
    for source, g in bw.groupby("source_name"):
        cg = cen[cen["source_name"] == source]
        rows.append({
            "source_name": source, "n": int(len(g)),
            "bandwidth_native_hz": _q(g[NATIVE], 0.5),
            "bandwidth_chain_hz": _q(g[CHAIN], 0.5),
            "bandwidth_lost_hz": _q(g["delta"], 0.5),
            "hf_ratio_8k_native": _q(hf.reindex(g["file_id"]).dropna(), 0.5),
            "centroid_native_hz": _q(cg[NATIVE], 0.5),
            "centroid_chain_hz": _q(cg[CHAIN], 0.5),
        })
    return pd.DataFrame(rows).sort_values("source_name").reset_index(drop=True)


def duration_report(signal: pd.DataFrame) -> pd.DataFrame:
    """Per partition: duration and silence, read against the test set's window.

    **Decides**: the crop/pad policy -- and this is the table to read first.
    [06 X1c](../../docs/EDA/06-cross-pool.md) found duration holding **0.933
    grouped AUC** on `music_present` when every other head collapsed to chance,
    because every music source we hold is long and every voice source is short
    while the test set is 4-60 s for both. The `in_test_window` column is that
    gap in one number per pool.

    ⚠️ Measured on the **chain** plane. `duration_s_decoded` is identical on
    both by construction -- resampling does not change how long a file is --
    and reading it from the chain plane makes that explicit rather than lucky.
    """
    dur = pd.to_numeric(signal[f"duration_s_decoded_{CHAIN}"], errors="coerce")
    sil = pd.to_numeric(signal[f"silence_ratio_{CHAIN}"], errors="coerce")
    lead = pd.to_numeric(signal[f"lead_silence_s_{CHAIN}"], errors="coerce")
    tail = pd.to_numeric(signal[f"tail_silence_s_{CHAIN}"], errors="coerce")
    spans = pd.to_numeric(signal[f"n_valid_spans_ge_4s_{CHAIN}"], errors="coerce")
    frame = pd.DataFrame({"partition": signal["partition"], "duration_s": dur,
                          "silence_ratio": sil, "lead_s": lead, "tail_s": tail,
                          "spans": spans})
    rows = []
    for partition, g in frame.groupby("partition"):
        d = g["duration_s"].dropna()
        rows.append({
            "partition": partition, "n": int(len(g)),
            "dur_p05": _q(d, 0.05), "dur_median": _q(d, 0.5),
            "dur_p95": _q(d, 0.95),
            "in_test_window": float(((d >= TEST_MIN_S) & (d <= TEST_MAX_S)).mean())
            if len(d) else float("nan"),
            "under_4s": float((d < TEST_MIN_S).mean()) if len(d) else float("nan"),
            "over_60s": float((d > TEST_MAX_S).mean()) if len(d) else float("nan"),
            "silence_median": _q(g["silence_ratio"].dropna(), 0.5),
            "lead_median_s": _q(g["lead_s"].dropna(), 0.5),
            "tail_median_s": _q(g["tail_s"].dropna(), 0.5),
            "has_4s_span": float((g["spans"].dropna() >= 1).mean())
            if g["spans"].notna().any() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("partition").reset_index(drop=True)


def level_report(signal: pd.DataFrame) -> pd.DataFrame:
    """Per source: loudness, headroom, DC and clipping, on the chain plane.

    **Decides**: whether to normalise, and at which stage. A corpus whose
    sources differ by 20 dB in median RMS hands the model a level cue that has
    nothing to do with whether audio is generated -- the same shape as the
    metadata confound, in a column no metadata audit would catch.

    ⚠️ Chain plane. Level after the chain is the level the model sees, and it
    is the only one a normalisation decision can act on.
    """
    rows = []
    cols = {c: pd.to_numeric(signal[f"{c}_{CHAIN}"], errors="coerce")
            for c in ("peak_dbfs", "rms_dbfs", "crest_factor_db", "dc_offset",
                      "clipping_ratio")}
    frame = pd.DataFrame({"source_name": signal["source_name"], **cols})
    for source, g in frame.groupby("source_name"):
        rows.append({
            "source_name": source, "n": int(len(g)),
            "peak_dbfs_median": _q(g["peak_dbfs"].dropna(), 0.5),
            "rms_dbfs_median": _q(g["rms_dbfs"].dropna(), 0.5),
            "rms_dbfs_p05": _q(g["rms_dbfs"].dropna(), 0.05),
            "rms_dbfs_p95": _q(g["rms_dbfs"].dropna(), 0.95),
            "crest_db_median": _q(g["crest_factor_db"].dropna(), 0.5),
            "dc_offset_max": float(g["dc_offset"].abs().max())
            if g["dc_offset"].notna().any() else float("nan"),
            "frac_clipping": float((g["clipping_ratio"].fillna(0) > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("source_name").reset_index(drop=True)


def signal_audit(signal: pd.DataFrame, cfg=None, *, plane: str = CHAIN,
                 group_column: str = "source_name") -> pd.DataFrame:
    """X1, re-run on what the decoder actually produced. **The decisive number.**

    X1 asks whether the label is predictable from metadata; every fix for that
    is a render-time transform, and [06 X1b](../../docs/EDA/06-cross-pool.md)
    settled it -- tags stripped, one identical encode, the confound is gone from
    the file. This asks the harder version: **is the label still predictable
    from the audio itself, after the chain the competition mandates?**

    🔴 Run on the `chain` plane by default, because that is the only plane a
    model ever sees. A native-plane result is interesting for R1 and is not a
    statement about leakage the model can exploit.

    ⚠️ A high AUC here is *not* automatically a defect. Real audio and vocoded
    audio genuinely differ, and a detector is supposed to find that. What makes
    a number here a **shortcut** is surviving the *archive* holdout: duration
    separating pool C from pool D at 30 s against 10 s is a property of how the
    two archives were built, not of whether music is generated, and the test
    set is 4-60 s for both.

    ⚠️ `group_column` decides which question is being asked, and
    `shortcut_audit`\'s docstring has the difference. `source_name` holds out an
    archive; `group_key` holds out a clip or a speaker and leaves the archive in
    training, so it scores **higher**. Measured on the chain plane:
    `music_present` is 0.852 source-grouped and 0.894 group-keyed, and only the
    first is evidence about publisher generalisation.
    """
    from eda.analyze.shortcut import shortcut_audit

    return shortcut_audit(signal, cfg, features=plane_features(plane),
                          group_column=group_column)


# --------------------------------------------------------------------------- #
# E1 pass 2: duplicates by content rather than by bytes
# --------------------------------------------------------------------------- #

#: Cosine similarity above which two files are reported as the same content.
#: 🔴 Calibrated, not guessed -- `content_similarity_profile` reports the
#: distribution this sits in, and docs/EDA/05 E1b records what it found there.
#: Deliberately severe: the sweep exists to *name* suspects for a human to
#: confirm, and a permissive threshold over 58,885 files produces a list nobody
#: reads.
DUPLICATE_COSINE = 0.999


def content_fingerprints(vectors: dict[str, np.ndarray], meta: pd.DataFrame
                         ) -> tuple[np.ndarray, pd.DataFrame]:
    """Chain-plane LTAS as a content fingerprint: z-scored, then L2-normalised.

    🔴 **Chain plane, and z-scored per row.** E1 pass 1 compares sha256 and so
    cannot see a file that was re-encoded, re-containered or published at a
    second sample rate. The chain plane puts every file at 16 kHz, so a rate
    difference stops being a difference; z-scoring each row removes the overall
    level, so a re-mastered copy still matches its original.

    ⚠️ **It is a timbre fingerprint, not an identity one, and the difference was
    measured rather than assumed.** The median file's best match among 58,883
    others is already **cosine 0.965**, and at a 0.999 threshold the top hits
    include `bus-helsinki-20-789-a_0` against `bus-vienna-38-1134-a_0` -- two
    different recordings of a bus, whose long-term spectra agree because buses
    sound like buses. Decoded and compared sample by sample, those pairs differ
    by up to 0.61 in amplitude.

    So this **shortlists candidates**; it does not confirm duplicates. A hit is
    a pair worth looking at, and what confirmed the real finding in
    [05 E1b](../../docs/EDA/05-pool-e-noise.md) was the publisher\'s own
    filenames, not this number.

    ⚠️ Rows whose vectors failed are dropped and counted on the frame. An
    all-NaN row L2-normalises to NaN and would match nothing while silently
    shrinking every denominator.
    """
    ltas = np.asarray(vectors[f"ltas_{CHAIN}"], dtype=np.float64)
    ids = np.asarray(vectors["file_id"], dtype=object)
    if len(ltas) != len(meta):
        raise ValueError(
            f"{len(ltas)} vector row(s) against {len(meta)} metadata row(s); "
            f"they are joined by position and must be the same table")

    good = ~np.isnan(ltas).any(axis=1)
    ltas, ids, kept = ltas[good], ids[good], meta[good].reset_index(drop=True)

    centred = ltas - ltas.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    # A row with no spectral variation at all -- digital silence -- would divide
    # by zero and then match every other silent file at cosine 1.0. It is a real
    # finding, but it is `mel_bands_flat`'s finding, not a duplicate.
    flat = (scale.ravel() == 0.0)
    scale[flat] = 1.0
    unit = centred / scale
    norm = np.linalg.norm(unit, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    out = (unit / norm).astype(np.float32)

    kept = kept.assign(file_id=ids)
    kept.attrs["dropped"] = int((~good).sum())
    kept.attrs["flat"] = int(flat.sum())
    return out, kept


def near_duplicates(fingerprints: np.ndarray, meta: pd.DataFrame, *,
                    threshold: float = DUPLICATE_COSINE, block: int = 1024
                    ) -> pd.DataFrame:
    """Every pair above `threshold`, with both sides' source and partition.

    Blocked matrix multiply rather than a neighbour index: the fingerprint is
    128-dimensional, so the full 58,885-square product is ~4e11 flops and runs
    in seconds, while a tree index degrades badly at that dimension and an
    approximate one would make the answer depend on a random seed.

    Pairs are upper-triangular -- `i < j` -- so a pair is reported once.
    """
    n = len(fingerprints)
    if n != len(meta):
        raise ValueError(f"{n} fingerprint(s) against {len(meta)} row(s)")
    source = meta["source_name"].to_numpy()
    partition = meta["partition"].to_numpy()
    ids = meta["file_id"].to_numpy()
    # 🔴 The column this sweep exists for. A near-duplicate pair inside one
    # group is the grouping working -- ten microphones on one room, two takes by
    # one speaker -- and costs nothing. A pair that **straddles** two groups is
    # a fold leak that `group_key` does not catch, because the fold builder will
    # happily put one side in train and the other in validation.
    has_group = "group_key" in meta.columns
    group = (meta["group_key"].to_numpy() if has_group
             else np.full(len(meta), None, dtype=object))
    rows = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = fingerprints[start:stop] @ fingerprints.T
        # Upper triangle only: column index must exceed the absolute row index.
        cols = np.arange(n)[None, :]
        sim[cols <= np.arange(start, stop)[:, None]] = -np.inf
        hit_r, hit_c = np.nonzero(sim >= threshold)
        for r, c in zip(hit_r, hit_c):
            i = start + int(r)
            rows.append({"file_id_a": ids[i], "file_id_b": ids[c],
                         "source_a": source[i], "source_b": source[c],
                         "partition_a": partition[i], "partition_b": partition[c],
                         "cosine": float(sim[r, c]),
                         "group_a": group[i], "group_b": group[c],
                         "cross_source": source[i] != source[c],
                         "cross_partition": partition[i] != partition[c],
                         "cross_group": (group[i] != group[c]) if has_group
                         else None})
    return pd.DataFrame(rows, columns=["file_id_a", "file_id_b", "source_a",
                                       "source_b", "partition_a", "partition_b",
                                       "group_a", "group_b", "cosine",
                                       "cross_source", "cross_partition",
                                       "cross_group"])


def content_similarity_profile(fingerprints: np.ndarray, *, block: int = 1024,
                               quantiles=(0.5, 0.9, 0.99, 0.999, 1.0)
                               ) -> pd.DataFrame:
    """The distribution of each file's **best** match to any other file.

    🔴 This is what makes `DUPLICATE_COSINE` a calibration rather than a guess.
    A threshold picked without knowing where the bulk of top-1 similarities sit
    is a number that either reports everything or nothing, and both read as a
    clean corpus.
    """
    n = len(fingerprints)
    best = np.full(n, -np.inf)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = fingerprints[start:stop] @ fingerprints.T
        np.fill_diagonal(sim[:, start:stop], -np.inf)
        best[start:stop] = sim.max(axis=1)
    series = pd.Series(best)
    return pd.DataFrame({"quantile": list(quantiles),
                         "top1_cosine": [float(series.quantile(q))
                                         for q in quantiles]})
