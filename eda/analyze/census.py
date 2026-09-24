"""The six censuses that were measured and never tabulated (docs/EDA/10 §2.3).

A1 and B3 (native format per source), B2 (MLAAD language x generator), C6 (codec
provenance), D2 (per-generator) and D3 (near-Nyquist per generator, both planes).
Each was marked 📊 in [10 §1](../../docs/EDA/10-final-plan.md) -- *"measured and
on disk; no write-up. A `groupby` away, no decode"* -- and this is that groupby.

⚠️ **A1 and B3 are one function on purpose.** B3's question is *"the format
census against pool A"*, which is A1's table read across the pool boundary. Two
functions would be two groupbys that could disagree about, say, how a null
`encoder` is counted -- and the comparison is the entire point of B3.

🔴 **D2 and D3 were blocked on a key nobody had built**, not on data. Both ask
for a per-*generator* breakdown and the census carries only `source_name` and
`group_key`; in pool D `group_key` is the parent clip
([02 B6b](../../docs/EDA/02-pool-b-fake-voice.md)). They are unblocked by
`eda.analyze.screens.generator_key`, which is why that lives in `screens` and is
imported here rather than reimplemented.
"""

from __future__ import annotations

import pandas as pd

from eda.analyze.screens import generator_key
from eda.planes import CHAIN, NATIVE

__all__ = ["codec_provenance", "format_census", "generator_census",
           "mlaad_inventory", "near_nyquist_by_generator"]

#: 🔴 Declared column lists, one per census. They exist so an **empty** result
#: is still a table with the right shape: `pd.DataFrame([])` has no columns and
#: the `sort_values` that follows raises `KeyError` on the sort key, which reads
#: as a code fault rather than as the finding that nothing matched. The same
#: defect hit `screens.separability` first, where the empty result *was* the
#: finding -- pool D's `group_key` is the parent clip and no group is
#: measurable.
FORMAT_COLUMNS = ("source_name", "pool", "n", "hours", "container",
                  "n_containers", "codec", "sample_fmt", "orig_sr_mode",
                  "n_rates", "channels_mode", "encoder_present", "n_encoders")
MLAAD_COLUMNS = ("language", "generator", "n", "hours", "orig_sr_mode",
                 "container")
CODEC_COLUMNS = ("source_name", "container", "n", "lossy", "codec",
                 "bit_rate_median", "bit_rate_p05", "bit_rate_p95",
                 "xing_present", "lame_present")
GENERATOR_COLUMNS = ("generator", "source_name", "pool", "n", "hours",
                     "orig_sr_mode", "container", "duration_s_median",
                     "duration_s_std", "duration_constant")
NEAR_NYQUIST_COLUMNS = ("generator", "n", "source_name",
                        *(f"{stat}_{plane}"
                          for plane in (NATIVE, CHAIN)
                          for stat in ("near_nyquist_ratio",
                                       "effective_bandwidth_hz",
                                       "hf_ratio_8k")))


def _mode(values: pd.Series):
    """The most common value, or NA. `Series.mode()` returns a frame-shaped
    result and an empty one for an all-null column; both need handling."""
    found = values.dropna()
    return found.mode().iloc[0] if len(found) else pd.NA


def format_census(files: pd.DataFrame) -> pd.DataFrame:
    """**A1 + B3**: the native-format census, one row per source.

    A1 asks for it over pool A; B3 asks for pool B's *against* pool A's. The
    table carries `pool` so the comparison is a sort, not a second query.
    """
    rows = []
    for source, block in files.groupby("source_name"):
        rates = block["orig_sr"].dropna()
        rows.append({
            "source_name": source,
            "pool": _mode(block["pool"]) if "pool" in block else pd.NA,
            "n": len(block),
            "hours": float(block["duration_s"].sum() / 3600),
            "container": _mode(block["container"]),
            "n_containers": int(block["container"].nunique(dropna=True)),
            "codec": _mode(block["codec_name"]),
            "sample_fmt": _mode(block["sample_fmt"]),
            "orig_sr_mode": int(_mode(rates)) if len(rates) else -1,
            "n_rates": int(rates.nunique()),
            "channels_mode": _mode(block["orig_channels"]),
            # 🔴 Missingness, not just the value. `encoder` is null for every
            # wav and present for every mp3, so *whether it is there* is the
            # archive fingerprint -- which is what `build_design`'s `_isna`
            # indicators encode and what X2 found inverts under a holdout.
            "encoder_present": float(block["encoder"].notna().mean()),
            "n_encoders": int(block["encoder"].nunique(dropna=True)),
        })
    return (pd.DataFrame(rows, columns=FORMAT_COLUMNS)
            .sort_values(["pool", "source_name"]).reset_index(drop=True))


def mlaad_inventory(files: pd.DataFrame) -> pd.DataFrame:
    """**B2**: MLAAD's `language x generator` leaf inventory.

    B2's headline question is *"whether Korean fake voice exists in our pool at
    all"* -- the competition is Korean-hosted and the Korean slice size is an
    open decision with no evidence attached. This table is that evidence.
    """
    mlaad = files[files["source_name"] == "mlaad"].copy()
    if mlaad.empty:
        return pd.DataFrame(columns=list(MLAAD_COLUMNS))
    # `payload/fake/<language>/<generator>/<file>`; the generator half is
    # `generator_key`'s, so the two cannot disagree about the depth.
    parts = mlaad["path"].str.split("/")
    mlaad["language"] = parts.map(lambda p: p[4] if len(p) > 4 else pd.NA)
    mlaad["generator"] = generator_key(mlaad).str.split("/").str[-1]
    rows = []
    for (language, generator), block in mlaad.groupby(["language", "generator"]):
        rates = block["orig_sr"].dropna()
        rows.append({"language": language, "generator": generator,
                     "n": len(block),
                     "hours": float(block["duration_s"].sum() / 3600),
                     "orig_sr_mode": int(_mode(rates)) if len(rates) else -1,
                     "container": _mode(block["container"])})
    return (pd.DataFrame(rows, columns=MLAAD_COLUMNS)
            .sort_values(["language", "generator"]).reset_index(drop=True))


def codec_provenance(files: pd.DataFrame) -> pd.DataFrame:
    """**C6**: the mp3 question -- which sources are lossy, and how lossy.

    C6 exists because a lossy source and a lossless one differ above 16 kHz in a
    way the chain cannot undo, and because `bit_rate` is one of the metadata
    columns X2 condemned. The table is per `source x container`, so a source
    that mixes them is visible rather than averaged.
    """
    rows = []
    for (source, container), block in files.groupby(
            ["source_name", "container"], dropna=False):
        rate = pd.to_numeric(block["bit_rate"], errors="coerce").dropna()
        rows.append({
            "source_name": source, "container": container, "n": len(block),
            "lossy": bool(container in {"mp3", "ogg", "m4a", "aac"}),
            "codec": _mode(block["codec_name"]),
            "bit_rate_median": float(rate.median()) if len(rate) else float("nan"),
            "bit_rate_p05": float(rate.quantile(0.05)) if len(rate) else float("nan"),
            "bit_rate_p95": float(rate.quantile(0.95)) if len(rate) else float("nan"),
            "xing_present": float(block["xing_tag"].fillna(False).mean()),
            "lame_present": float(block["lame_tag"].fillna(False).mean()),
        })
    return (pd.DataFrame(rows, columns=CODEC_COLUMNS)
            .sort_values(["source_name", "container"]).reset_index(drop=True))


def generator_census(files: pd.DataFrame) -> pd.DataFrame:
    """**D2**: per-generator counts, hours and native format.

    Not per *source*: that already exists and is what made D2 look done.
    """
    frame = files.copy()
    frame["generator"] = generator_key(frame)
    frame = frame[frame["generator"].notna()]
    rows = []
    for generator, block in frame.groupby("generator"):
        rates = block["orig_sr"].dropna()
        rows.append({
            "generator": generator,
            "source_name": _mode(block["source_name"]),
            "pool": _mode(block["pool"]) if "pool" in block else pd.NA,
            "n": len(block),
            "hours": float(block["duration_s"].sum() / 3600),
            "orig_sr_mode": int(_mode(rates)) if len(rates) else -1,
            "container": _mode(block["container"]),
            "duration_s_median": float(block["duration_s"].median()),
            # 🔴 The column B6b turned on. Constant here means the length *is*
            # the generator's id.
            #
            # ⚠️ **Not `std == 0`.** Measured: `mustango` and
            # `MusicGen_medium` are each a single repeated value (10.242 and
            # 10.180) whose `std` comes back as `1.776357e-15`, not `0.0` --
            # so a zero test counts 3 of FakeMusicCaps' 5 constant generators
            # and the write-up says "3 of 231" when the answer is 5. `min ==
            # max` is exact and has no tolerance to tune.
            "duration_s_std": float(block["duration_s"].std(ddof=0)),
            "duration_constant": bool(
                block["duration_s"].min() == block["duration_s"].max()),
        })
    # 🔴 Declared columns, so an empty result is still a table. The identical
    # defect in `screens.separability` raised `KeyError: 'best_auc'` and read as
    # a code fault rather than as the finding that nothing matched.
    return (pd.DataFrame(rows, columns=GENERATOR_COLUMNS)
            .sort_values(["pool", "source_name", "generator"])
            .reset_index(drop=True))


def near_nyquist_by_generator(signal: pd.DataFrame) -> pd.DataFrame:
    """**D3**: near-Nyquist rolloff per generator, on **both** planes.

    D3's point is R1's: a generator whose rolloff is distinctive at `native` and
    not at `chain` has a fingerprint the competition's own resample destroys,
    and one that is distinctive at both has a fingerprint that survives. The two
    columns must be read as a pair -- reporting either alone answers neither
    question.
    """
    frame = signal.copy()
    frame["generator"] = generator_key(frame)
    frame = frame[frame["generator"].notna()]
    rows = []
    for generator, block in frame.groupby("generator"):
        row = {"generator": generator, "n": len(block),
               "source_name": _mode(block["source_name"])}
        for plane in (NATIVE, CHAIN):
            for stat in ("near_nyquist_ratio", "effective_bandwidth_hz",
                         "hf_ratio_8k"):
                column = f"{stat}_{plane}"
                values = pd.to_numeric(block.get(column), errors="coerce")
                row[f"{stat}_{plane}"] = (float(values.median())
                                          if values is not None and len(values)
                                          else float("nan"))
        rows.append(row)
    return (pd.DataFrame(rows, columns=NEAR_NYQUIST_COLUMNS)
            .sort_values(["source_name", "generator"]).reset_index(drop=True))
