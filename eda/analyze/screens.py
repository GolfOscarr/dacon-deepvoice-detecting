"""Step 2 of docs/EDA/10: four questions answered over columns already on disk.

B4/D6 (degenerate output), B6 (cross-generator separability), E5 (cell-9
viability) and X2 (the metadata leak). None of them decodes -- that is the point
of grouping them, and it is why the package docstring's no-audio rule holds here
too.

🔴 **The symmetry rule governs the whole module.**
[04 D6](../../docs/EDA/04-pool-d-fake-instrumental.md) states it for the
degenerate screen and it applies to every filter this file proposes: *"any
threshold that flags pool C at a materially different rate than pool D is an
asymmetric filter (R2) and is manufacturing a cue."*
[02 B4](../../docs/EDA/02-pool-b-fake-voice.md) puts the same rule the other way
round -- *"do not simply drop silent fakes without checking pool A's silent
rate"*. So `degenerate_screen` does not return a drop list. It returns the
**per-pool flag rate under one threshold**, because the rate *is* the finding:
a screen that fires at 3% on fake and 0.1% on real has not found broken files,
it has built a label.

⚠️ **One half of B4/D6 is not computable here and is not silently omitted.**
Both specs ask for a looping/babble detector -- the maximum of the mel-envelope
autocorrelation at lags between 0.5 s and half the file length. That needs the
envelope *over time*, and the V tier stores per-band statistics over time
(`eda.extract.vectors`), not the series itself. It needs a decode.
`LOOPING_NOT_COMPUTED` records that rather than letting a table of five screens
read as if it were the six that were asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda.analyze.shortcut import HEADS, head_labels, univariate_auc
from eda.analyze.signal import plane_features
from eda.extract.vectors import N_MELS
from eda.planes import CHAIN, NATIVE

__all__ = ["CLIPPED_RATIO", "DURATION_OUTLIER", "LOOPING_NOT_COMPUTED",
           "MIN_MEASURABLE_S", "SILENT_PEAK_DBFS", "SILENCE_RATIO",
           "SCREENS", "cell9_viability", "degenerate_flags",
           "degenerate_screen", "metadata_leak", "separability"]

#: Peak below this is digital silence for the screen's purposes. -60 dBFS is
#: three orders of magnitude under full scale; nothing audible lives there and
#: a legitimately quiet music tail still peaks far above it.
SILENT_PEAK_DBFS = -60.0

#: The share of samples at full scale that counts as clipped output. 1% is well
#: past what mastering produces and into what a broken generator produces.
CLIPPED_RATIO = 0.01

#: Files shorter than this cannot be measured meaningfully -- and the test set
#: starts at 4 s, so anything here is unusable regardless of why.
MIN_MEASURABLE_S = 0.5

#: Almost entirely silent by the timing extractor's own definition.
#: ⚠️ Deliberately not a "mostly silent" threshold: [04 D6] warns that long
#: near-silent stretches are **legitimate** in music in a way they are not in
#: speech, so this fires only on files that are essentially nothing.
SILENCE_RATIO = 0.99

#: A duration this many times off its own group's median. Relative to the group
#: rather than absolute, because pool D is uniformly 10.000 s and pool C is
#: 30.003 s -- an absolute window would flag one pool wholesale.
DURATION_OUTLIER = 4.0

#: The sub-part of B4/D6 that a decode-free pass cannot deliver. Named so it
#: appears in the output rather than being absent from it.
LOOPING_NOT_COMPUTED = (
    "mel-envelope autocorrelation (looping/babble) needs the envelope over "
    "time, which the V tier does not store -- it needs a decode")

#: Screen name -> what it means. The flag columns `degenerate_flags` emits.
SCREENS = {
    "digital_silence": f"peak <= {SILENT_PEAK_DBFS} dBFS",
    "all_bands_flat": f"all {N_MELS} mel bands constant over time",
    "clipped": f"more than {CLIPPED_RATIO:.0%} of samples at full scale",
    "too_short": f"under {MIN_MEASURABLE_S} s",
    "all_silence": f"silence_ratio >= {SILENCE_RATIO}",
    "duration_outlier": f"{DURATION_OUTLIER}x off its own group's median",
}


def degenerate_flags(signal: pd.DataFrame, plane: str = CHAIN) -> pd.DataFrame:
    """One boolean column per screen, plus `any_flag`. One row per input row.

    Measured on the **chain** plane by default: a file is degenerate if what
    reaches the model is degenerate. A native-plane flag on a file the chain
    repairs would be a finding about the archive, not about the training data.
    """
    if plane not in (NATIVE, CHAIN):
        raise ValueError(f"unknown plane {plane!r}")

    def col(name: str) -> pd.Series:
        full = f"{name}_{plane}"
        if full not in signal.columns:
            raise KeyError(
                f"{full!r} is not in the signal table; the screen cannot be "
                f"run on the {plane!r} plane without it")
        return pd.to_numeric(signal[full], errors="coerce")

    duration = col("duration_s_decoded")
    # 🔴 The group median, and `group_key` is the right grain: it is the
    # publisher's own atom (`eda.groupkeys`), so a generator directory in pool D
    # is compared against itself rather than against pool C's 30 s excerpts.
    if "group_key" in signal.columns:
        median = duration.groupby(signal["group_key"]).transform("median")
    else:
        median = pd.Series(duration.median(), index=signal.index)
    ratio = duration / median.replace(0.0, np.nan)
    outlier = (ratio > DURATION_OUTLIER) | (ratio < 1.0 / DURATION_OUTLIER)

    flags = pd.DataFrame({
        "digital_silence": col("peak_dbfs") <= SILENT_PEAK_DBFS,
        "all_bands_flat": col("mel_bands_flat") >= N_MELS,
        "clipped": col("clipping_ratio") > CLIPPED_RATIO,
        "too_short": duration < MIN_MEASURABLE_S,
        "all_silence": col("silence_ratio") >= SILENCE_RATIO,
        "duration_outlier": outlier.fillna(False),
    }, index=signal.index).fillna(False)
    flags["any_flag"] = flags.any(axis=1)
    return flags


def degenerate_screen(signal: pd.DataFrame, plane: str = CHAIN) -> pd.DataFrame:
    """B4 / D6: the per-partition flag rate under **one** threshold.

    🔴 Returns rates, not a drop list, and the rates are the finding. A screen
    that fires at a materially different rate on the fake pools than on the real
    ones is an asymmetric filter (R2) -- it would manufacture "silence means
    fake" rather than remove broken files. The comparison the reader must make
    is B against A, and D against C, **down the same column**.
    """
    flags = degenerate_flags(signal, plane)
    by = signal["partition"] if "partition" in signal.columns else pd.Series(
        "all", index=signal.index)
    rows = []
    for partition, idx in flags.groupby(by).groups.items():
        block = flags.loc[idx]
        row = {"partition": partition, "n": len(block)}
        for screen in SCREENS:
            row[screen] = float(block[screen].mean())
        row["any_flag"] = float(block["any_flag"].mean())
        row["n_flagged"] = int(block["any_flag"].sum())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("partition").reset_index(drop=True)
    out.attrs["plane"] = plane
    out.attrs["not_computed"] = LOOPING_NOT_COMPUTED
    return out


#: `source_name -> the index of the path component naming the generator`, counted
#: from the source root. Explicit per source, in `eda.groupkeys`' style, because
#: a guessed depth silently returns a clip id or an archive name.
#:
#: 🔴 **This is the stratification axis, and it is not `group_key`.**
#: [02 B0b](../../docs/EDA/02-pool-b-fake-voice.md) states the distinction for
#: WaveFake -- *"the vocoder is a stratification axis; the voice is the grouping
#: one"* -- and pool D is where confusing them bites hardest: `group_key` there
#: is the FakeMusicCaps parent clip, giving **5,521 groups of ~5 rows**, so a
#: separability run keyed on it measures nothing and reports nothing. Measured:
#: it returned zero rows.
#: ⚠️ Every depth here was counted against a real path and checked against the
#: resulting group count. Three of the first four guesses were wrong and each
#: failed *quietly* -- returning the CFAD split (`fake_clean`), the MLAAD
#: **language** (`bg`, 54 of them, which looks exactly like a plausible
#: generator count), and WaveFake's archive directory. `generator_key` therefore
#: refuses a depth that resolves to one constant value per source rather than
#: trusting this table.
GENERATOR_DEPTH = {
    # fakemusiccaps/zenodo-15063698/<Generator>/<clip>.wav
    "fakemusiccaps": 1,
    # cfad/<archive>/CFAD/clean_version/<split>/fake_clean/<generator>/<file>
    "cfad-fake": 5,
    # mlaad/v9/payload/fake/<language>/<generator>/<file>
    "mlaad": 4,
    # wavefake/<archive>/generated_audio/<vocoder>/<file>
    "wavefake": 2,
}


def generator_key(signal: pd.DataFrame) -> pd.Series:
    """The generator directory per row, or NA for a source without one.

    Serves B6 here and the D2 / D3 per-generator censuses in step 3 -- which is
    why it is one function rather than three path splits that can disagree.

    ⚠️ NA rather than a fallback. A source with no registered generator depth is
    one nobody has looked at, and inventing a component for it would put an
    archive name or a clip id in a column labelled `generator`.
    """
    if "path" not in signal.columns:
        raise KeyError("no `path` column; the generator is a path component")
    out = pd.Series(pd.NA, index=signal.index, dtype="object")
    for source, depth in GENERATOR_DEPTH.items():
        rows = signal["source_name"] == source
        if not rows.any():
            continue
        # The path is `<source root>/<...>`; the root's own depth varies per
        # source, so count from the first component after the source name.
        parts = signal.loc[rows, "path"].str.split("/")
        keys = parts.map(
            lambda p, d=depth: f"{p[0]}/{p[d + 1]}" if len(p) > d + 1 else pd.NA)
        # 🔴 A depth that lands on a directory every file shares resolves to one
        # value and reads as "this source has a single generator", which is a
        # plausible sentence and was wrong three times out of four while this
        # table was being written. A source with genuinely one generator does
        # not belong in `GENERATOR_DEPTH` at all.
        distinct = int(keys.nunique(dropna=True))
        if distinct <= 1 and rows.sum() > 1:
            raise ValueError(
                f"GENERATOR_DEPTH[{source!r}] = {depth} resolves to "
                f"{distinct} distinct value(s) over {int(rows.sum())} rows "
                f"({keys.dropna().iloc[0] if distinct else 'nothing'!r}). That "
                f"is a shared parent directory, not a generator -- recount the "
                f"depth against an actual path")
        out.loc[rows] = keys
    return out


def separability(signal: pd.DataFrame, partition: str, *,
                 by: str = "group_key", min_rows: int = 50) -> pd.DataFrame:
    """B6: how identifiable is each generator from the S tier alone?

    One row per `group_key` in `partition`: the best single chain-plane feature
    at telling that group from the rest of its own partition, one-vs-rest.

    🔴 Read as a **leakage** measurement, not a capability one. A generator that
    a single scalar separates at AUC 0.95 from its pool-mates is a generator the
    model can memorise, and `group_key` is the fold-disjointness axis -- so a
    high number here says a fold split will *look* like generator transfer while
    measuring generator recognition. It is B1's family question
    (`eda.analyze.pairs`) asked of the whole pool with a cruder instrument.

    ⚠️ Univariate by design. A multivariate model would score higher on every
    row and say less: the point is to name the **column** that carries the
    identity, because that is the one a render-time transform has to neutralise.
    """
    sub = signal[signal["partition"] == partition]
    if by == "generator":
        axis = generator_key(sub)
    elif by in sub.columns:
        axis = sub[by]
    else:
        raise KeyError(
            f"no {by!r} column; `group_key` reaches the signal table through "
            f"`load_signal`'s join, and `generator` is built by `generator_key`")
    spec = plane_features(CHAIN)
    columns = [c for c in spec["numeric"] if c in sub.columns]
    if not columns:
        raise KeyError(f"no chain-plane feature columns in the {partition!r} rows")

    matrix = sub[columns].apply(pd.to_numeric, errors="coerce")
    # A column that is constant over the partition carries no information and
    # its AUC is an arbitrary 0.5; dropping it keeps `top_feature` meaningful.
    matrix = matrix.loc[:, matrix.nunique(dropna=True) > 1]
    rows = []
    n_groups = int(axis.nunique(dropna=True))
    for key, idx in sub.groupby(axis.rename("axis")).groups.items():
        if len(idx) < min_rows or len(idx) == len(sub):
            continue
        y = sub.index.isin(idx).astype(int)
        scores = {c: univariate_auc(matrix[c].to_numpy(dtype=float), y)
                  for c in matrix.columns}
        scores = {c: v for c, v in scores.items() if not np.isnan(v)}
        if not scores:
            continue
        best = max(scores, key=scores.get)
        rows.append({"partition": partition, "axis": by, "key": key,
                     "n": len(idx), "best_auc": float(scores[best]),
                     "top_feature": best,
                     "median_auc": float(np.median(list(scores.values())))})
    # 🔴 An empty result is an answer -- "no group in this partition is large
    # enough to measure" -- and it must come back as a table with the declared
    # columns. Measured: returning a bare `DataFrame([])` made `sort_values`
    # raise `KeyError: 'best_auc'`, which reads as a code fault rather than as
    # the finding that pool D's `group_key` is the clip and averages 5 rows.
    out = pd.DataFrame(rows, columns=["partition", "axis", "key", "n",
                                      "best_auc", "top_feature", "median_auc"])
    out = out.sort_values("best_auc", ascending=False).reset_index(drop=True)
    out.attrs["n_groups"] = n_groups
    out.attrs["n_measurable"] = len(out)
    out.attrs["min_rows"] = min_rows
    return out


def cell9_viability(signal: pd.DataFrame, *, partition: str = "E",
                    min_duration_s: float = 4.0,
                    speech_ratio: float = 0.0) -> pd.DataFrame:
    """E5: how much of pool E is *defensibly* cell 9, at the sampler's floor?

    Cell 9 is `(0, 0, None, None)` -- neither voice nor music present. A file
    asserting that while carrying audible speech is wrong at **any** level:
    #417333 A3 confirms `PRESENT = 1` at any duration.

    🔴 `speech_ratio = 0.0` means *any* speech evidence disqualifies, and that
    is deliberate rather than conservative. The competition's own answer makes
    this a threshold-free question, so a tolerance here would be us choosing a
    level of audibility that the rules say does not exist.

    ⚠️ **This is a one-sided screen and the table says so.** Silero is a speech
    VAD; it cannot evidence *music*, so `music_present = 0` goes unchecked here.
    [09 §7](../../docs/EDA/09-next-steps.md) records why PANNs was declined, and
    [RESULTS §5.1](../../docs/EDA/RESULTS_FOR_ANALYSIS.md) records the
    misreading that followed from forgetting it. Every count below is therefore
    an **upper bound** on cell-9 viability.
    """
    sub = signal[signal["partition"] == partition]
    if "vad_ok" not in sub.columns:
        raise KeyError("no `vad_ok` column; the content tier has not run")
    duration = pd.to_numeric(sub[f"duration_s_decoded_{CHAIN}"], errors="coerce")
    measured = sub["vad_ok"].fillna(False)
    long_enough = duration >= min_duration_s

    rows = []
    for source, idx in sub.groupby("source_name").groups.items():
        block = sub.loc[idx]
        ok = measured.loc[idx]
        keep = long_enough.loc[idx]
        # 🔴 Three states, never two. A file whose VAD did not run is **not**
        # evidence of absent speech, and folding it into the viable count is the
        # `na`-read-as-`pass` error the gates' tri-state exists to refuse.
        ratio = pd.to_numeric(block["vad_speech_ratio_50"], errors="coerce")
        silent = ok & (ratio <= speech_ratio)
        rows.append({
            "source_name": source, "n": len(block),
            "n_ge_min_duration": int(keep.sum()),
            "n_unmeasured": int((~ok).sum()),
            "n_voice_evidenced": int((ok & (ratio > speech_ratio)).sum()),
            "n_viable": int((silent & keep).sum()),
            "viable_share": float((silent & keep).mean()),
        })
    out = pd.DataFrame(rows).sort_values("source_name").reset_index(drop=True)
    out.attrs["upper_bound"] = (
        "music is unchecked: Silero is a speech VAD and PANNs was declined")
    out.attrs["min_duration_s"] = min_duration_s
    return out


def metadata_leak(files: pd.DataFrame, signal: pd.DataFrame, *,
                  cfg=None, group_column: str = "source_name") -> pd.DataFrame:
    """X2: is the metadata shortcut *exploitable*, or only a mirage?

    Runs the audit twice over the same rows -- once on M-tier metadata, once on
    chain-plane acoustics -- and reports the drop per head.

    🔴 **X1 and X2 are the same measurement read in opposite directions**
    ([06 X2](../../docs/EDA/06-cross-pool.md)). X1 wants the metadata AUC low;
    X2 asks whether a high one is worth shipping. They cannot both be satisfied:
    *"a metadata AUC high enough to be worth shipping is, by construction, a
    shortcut that will not generalize."*

    ⚠️ **The evidence that would resolve it in the other direction does not
    exist.** X2 says to resolve in X1's direction -- neutralise -- *unless* the
    organizers' dummy files show the test chain preserves the signal. Those
    files (`TEST_0000-0002.wav`) are X4's blocker and are not on disk, in the
    repo or in S3. So the conditional cannot fire, and this function reports the
    two AUCs and the drop; the decision they support is recorded in the docs,
    not computed here.
    """
    from eda.analyze.shortcut import shortcut_audit

    joined = signal
    meta = shortcut_audit(joined, cfg, group_column=group_column)
    chain = shortcut_audit(joined, cfg, features=plane_features(CHAIN),
                           group_column=group_column)
    out = meta[["head", "n", "auc", "auc_source_grouped", "top_features"]].merge(
        chain[["head", "auc", "auc_source_grouped", "top_features"]],
        on="head", suffixes=("_metadata", "_chain"))
    out["drop_ungrouped"] = out["auc_metadata"] - out["auc_chain"]
    out.attrs["blocked"] = (
        "the dummy-file range check needs TEST_0000-0002.wav, which is X4's "
        "blocker and is not on disk, in the repo or in S3")
    return out
