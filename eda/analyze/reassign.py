"""G-EDA6's work list: what to do about each row whose audio disputes its label.

`G-EDA6` fails on **2,442** rows and [07](../../docs/EDA/07-order-and-gates.md)
says *"reassign (F-A1) before dropping"*. The gate names the count; this module
names the **rows and the action**, which is the thing somebody actually has to
do.

🔴 **The 2,442 are three different problems and one number.** Reading them as a
single work list is the error this module exists to prevent -- they need
opposite actions:

* **Type A, 2,374 rows** assert `voice_present = 0` and carry speech. F-A1:
  *"Reassign, don't drop -- it becomes a mixed-cell sample"*
  ([data/10](../../docs/data/10-preprocessing-and-filtering.md)).
* **Type B-short, 42 rows** assert voice and carry none, and are **under 4 s** --
  the test set's own floor. A VAD given less than four seconds has little to work
  with, so these are `na`: *not measured*, never *measured and found empty*.
* **Type B-degenerate, 23 rows** assert voice, carry none, are of **normal
  length for their source**, and come from a pool that *generates* voice. Those
  are generation failures, and they are the population
  [02 B4](../../docs/EDA/02-pool-b-fake-voice.md) went looking for and did not
  find -- because the right detector for "the TTS produced no speech" is a VAD,
  not a clipping or level threshold.
* **Type B-sparse, 3 rows** are the same shape from pool **A**, which generates
  nothing. F-S4 is *generation*-failure detection, so calling a real
  recording's quiet passage a generation failure is wrong by construction.

⚠️ **Pool E is not a reassignment, and treating it as one would be worse than
doing nothing.** Pool E is an *additive layer*, not a standalone sample: noise
is mixed **under** a composite whose labels come from the other components. A
noise clip carrying speech, mixed under a music-only composite, makes that
composite carry voice while it asserts `voice_present = 0` -- so one
contaminated noise file mislabels every sample it is ever mixed into. And the
obvious reassignment is no good either: a field recording with distant speech is
a terrible cell-1 *voice* sample. The action is to restrict where it may be
used, which is neither "reassign" nor "drop".
"""

from __future__ import annotations

import pandas as pd

from eda.analyze.signal import VOICE_EVIDENCE_RATIO
from eda.planes import CHAIN
from training.manifest import POOL_LABELS
from training.spec import CELL_TABLE

__all__ = ["ACTIONS", "MIN_JUDGEABLE_S", "TARGET_CELL", "contradiction_rows",
           "reassignment_plan"]

#: A file shorter than this cannot be judged for voice the way a longer one can,
#: and it is the test set's own floor (4-60 s, docs/competition/01). Below it,
#: "the VAD found no speech" is not evidence that there is none.
MIN_JUDGEABLE_S = 4.0

#: Pool -> the cell a contradicted row belongs in once its voice is admitted.
#:
#: 🔴 Derived against `CELL_TABLE` rather than written as bare integers, and
#: checked at import: a cell number that does not carry the labels the
#: reassignment claims is the whole defect this table could introduce.
#:
#: * **C** real music + the real voice the VAD heard -> cell 5 `(1, 1, 0, 0)`.
#: * **D** fake music + voice. FakeMusicCaps is text-to-music, so any vocal in
#:   its output was generated with the rest of the clip -> cell 8 `(1, 1, 1, 1)`.
TARGET_CELL = {"C": 5, "D": 8}

#: What each group needs done. The strings are the artifact's `action` column.
ACTIONS = {
    "reassign_cell": "F-A1: becomes a mixed-cell whole_file sample",
    "restrict_noise": "may not be an additive layer under a PRESENT=0 composite",
    "unevidenceable": "under the 4 s floor; `na`, not a contradiction",
    "degenerate": "F-S4: normal length for its source, and no speech at all",
    "sparse_real": "a real recording the VAD found little in -- review, not a "
                   "generation failure",
}

#: Index of `label_voice_fake` in a `POOL_LABELS` tuple. Named rather than
#: written as `2`, for the reason `eda.analyze.shortcut` gives: the column order
#: is `training.manifest._LABEL_COLUMNS`', and a literal here would silently
#: swap two heads if that order ever changed.
_VOICE_FAKE = 2


def _check_targets() -> None:
    """Every target cell must actually assert voice, and match its pool's music.

    Cheap, and it is the only thing standing between a typo and 1,006 files
    reassigned to a cell that asserts the opposite of what was intended.
    """
    for pool, cell in TARGET_CELL.items():
        voice_present, music_present, _, music_fake = CELL_TABLE[cell]
        pool_labels = POOL_LABELS[pool]
        if voice_present != 1:
            raise ValueError(
                f"TARGET_CELL[{pool!r}] = {cell} asserts voice_present="
                f"{voice_present}; the whole point of the reassignment is that "
                f"the row carries voice")
        if music_present != 1 or music_fake != pool_labels[3]:
            raise ValueError(
                f"TARGET_CELL[{pool!r}] = {cell} carries music_fake="
                f"{music_fake} and pool {pool} asserts {pool_labels[3]}; "
                f"reassignment must not change what the music is")


_check_targets()


def contradiction_rows(signal: pd.DataFrame,
                       threshold: float = VOICE_EVIDENCE_RATIO) -> pd.DataFrame:
    """One row per contradicted file, with the action it needs.

    Columns: `file_id`, `source_name`, `pool`, `asserts_voice`,
    `speech_ratio`, `duration_s`, `kind`, `action`, `target_cell`, `why`.

    ⚠️ Rows whose VAD did not run are **excluded entirely**, not counted as
    either kind. `vad_ok = False` means no measurement exists, and a row with no
    measurement cannot contradict anything -- the same `na`-is-not-`pass`
    distinction the gates use.
    """
    measured = signal[signal["vad_ok"].fillna(False)].copy()
    asserts = {pool: labels[0] for pool, labels in POOL_LABELS.items()}
    measured["asserts_voice"] = measured["pool"].map(asserts)
    # A whole_file row has no pool and asserts nothing through one; its labels
    # come from its cell, and G-EDA6's `unevidenceable` path already covers it.
    measured = measured[measured["asserts_voice"].notna()]

    ratio = pd.to_numeric(measured["vad_speech_ratio_50"], errors="coerce")
    duration = pd.to_numeric(measured[f"duration_s_decoded_{CHAIN}"],
                             errors="coerce")
    says_none = measured["asserts_voice"] == 0

    rows = []
    for idx in measured.index:
        r, d = ratio.get(idx), duration.get(idx)
        if pd.isna(r):
            continue
        if says_none.get(idx) and r >= threshold:
            pool = measured.at[idx, "pool"]
            if pool in TARGET_CELL:
                kind, action = "A", "reassign_cell"
                target = TARGET_CELL[pool]
            else:
                # 🔴 Pool E. See the module docstring: it is a layer, not a
                # sample, so there is no cell to move it to -- the damage is to
                # whatever it gets mixed into.
                kind, action, target = "A", "restrict_noise", None
        elif not says_none.get(idx) and r < threshold:
            if pd.isna(d) or d < MIN_JUDGEABLE_S:
                kind, action, target = "B", "unevidenceable", None
            elif POOL_LABELS[measured.at[idx, "pool"]][_VOICE_FAKE] == 1:
                kind, action, target = "B", "degenerate", None
            else:
                # 🔴 F-S4 is **generation**-failure detection, and a pool-A row
                # was not generated. Calling a real recording's quiet passage a
                # generation failure is wrong by construction -- measured: 3
                # `cfad-real` rows sat just under the 0.20 threshold (ratio
                # 0.192) and the first version of this function filed them as
                # degenerate output from a corpus that generates nothing.
                kind, action, target = "B", "sparse_real", None
        else:
            continue
        rows.append({
            "file_id": measured.at[idx, "file_id"],
            "source_name": measured.at[idx, "source_name"],
            "pool": measured.at[idx, "pool"],
            "asserts_voice": int(measured.at[idx, "asserts_voice"]),
            "speech_ratio": float(r),
            "duration_s": None if pd.isna(d) else float(d),
            "kind": kind, "action": action, "target_cell": target,
            "why": ACTIONS[action],
        })
    out = pd.DataFrame(rows, columns=["file_id", "source_name", "pool",
                                      "asserts_voice", "speech_ratio",
                                      "duration_s", "kind", "action",
                                      "target_cell", "why"])
    out.attrs["threshold"] = threshold
    return out.sort_values(["action", "source_name", "file_id"]).reset_index(drop=True)


def reassignment_plan(rows: pd.DataFrame) -> pd.DataFrame:
    """The work list rolled up: how many of each action, per source."""
    if rows.empty:
        return pd.DataFrame(columns=["action", "source_name", "pool",
                                     "target_cell", "n", "speech_ratio_median"])
    out = (rows.groupby(["action", "source_name"], dropna=False)
           .agg(pool=("pool", "first"),
                target_cell=("target_cell", "first"),
                n=("file_id", "size"),
                speech_ratio_median=("speech_ratio", "median"))
           .reset_index())
    return out.sort_values(["action", "n"], ascending=[True, False]
                           ).reset_index(drop=True)
