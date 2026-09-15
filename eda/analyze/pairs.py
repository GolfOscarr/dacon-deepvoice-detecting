"""B1: the WaveFake ↔ LJSpeech paired vocoder experiment (docs/EDA/02 B1).

**The only unconfounded real/fake comparison in this corpus.** Every other one
differs in speaker, text, recording chain or archive as well as in label. Here
the real file and its seven fakes are the *same utterance by the same speaker
from the same recording*, and the only thing that differs is the vocoder --
which is the paired-negative construction docs/data/05 calls the T3 twin
design.

This module is table work only: it resolves utterances to their eight files and
reduces per-file mel vectors to per-vocoder differences. It never opens audio,
which is the rule `eda.analyze` exists to keep (see the package docstring) --
the decode is `eda.driver.pair_planes`.

🔴 **WaveFake ships three filename conventions and nothing announces it.**

    ljspeech_full_band_melgan     LJ001-0001_gen.wav
    ljspeech_hifiGAN              LJ001-0001_generated.wav
    ljspeech_melgan               LJ001-0001_gen.wav
    ljspeech_melgan_large         LJ001-0001_gen.wav
    ljspeech_multi_band_melgan    LJ001-0001_gen.wav
    ljspeech_parallel_wavegan     LJ001-0001_gen.wav
    ljspeech_waveglow             LJ001-0001.wav

The natural join -- `f"{utterance}_gen.wav"` -- resolves five of the seven and
silently loses HiFi-GAN and WaveGlow, one of which is the strongest vocoder in
the set. It fails as a *thin result*, never as an error: five columns where the
plan asked for seven, in a table nobody counts. `SUFFIXES` is therefore keyed by
directory and `pair_table` **refuses an utterance that does not resolve to all
eight files** rather than emitting a short row.

⚠️ This is a *selection*, not a redraw of `sample.json`. The recorded S-tier
draw is part of the corpus definition and every number published from it must
stay reproducible; B1 needs utterances the draw happens not to contain (only
128 of 13,100 are incidentally pairable in it), so it writes its own artifact
and leaves the draw alone.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eda.ids import relpath_of

__all__ = ["FAKE_SOURCE", "REAL_ROLE", "REAL_SOURCE", "SUFFIXES", "VOCODERS",
           "IncompletePair", "difference_images", "draw_pairs",
           "family_correlation", "pair_table", "surviving_fraction",
           "utterance_of", "vocoder_of"]

#: Pool A's LJSpeech, and pool B's WaveFake. Named rather than inlined because
#: the join is between two *sources*, and a typo in either is a table with zero
#: rows and no error.
REAL_SOURCE = "ljspeech"
FAKE_SOURCE = "wavefake"

#: What the real file is called in the `role` column. Not a vocoder name, and
#: deliberately not "ljspeech" -- the source is already its own column.
REAL_ROLE = "real"

#: Directory -> the suffix WaveFake appends to the utterance id in it. See the
#: module docstring: these are not uniform and the difference is unannounced.
SUFFIXES: Mapping[str, str] = {
    "ljspeech_full_band_melgan": "_gen",
    "ljspeech_hifiGAN": "_generated",
    "ljspeech_melgan": "_gen",
    "ljspeech_melgan_large": "_gen",
    "ljspeech_multi_band_melgan": "_gen",
    "ljspeech_parallel_wavegan": "_gen",
    "ljspeech_waveglow": "",
}

#: The seven, in a fixed order so a correlation matrix's axes are stable across
#: runs and two of them can be diffed.
VOCODERS: tuple[str, ...] = tuple(SUFFIXES)

#: Every file in a complete pair: the real one plus all seven fakes.
PAIR_SIZE = len(VOCODERS) + 1

#: LJSpeech's utterance id, `LJ<chapter>-<line>`. WaveFake's filenames carry it
#: verbatim, which is what makes the join possible at all.
UTTERANCE = re.compile(r"(LJ\d{3}-\d{4})")


class IncompletePair(RuntimeError):
    """An utterance resolved to fewer than `PAIR_SIZE` files.

    Its own type because it is the failure this module is built to surface. A
    silently short pair is a per-vocoder mean computed over a different
    population than the one beside it in the table, which reads as a vocoder
    difference and is not one.
    """


def utterance_of(file_id: str) -> str | None:
    """The LJSpeech utterance id in a `file_id`, or None.

    Matches on the path half only. The source half cannot contain one, but
    saying so costs nothing and the whole-string search would silently start
    matching if a source were ever renamed to something LJ-shaped.
    """
    found = UTTERANCE.search(relpath_of(file_id))
    return found.group(1) if found else None


def vocoder_of(file_id: str) -> str | None:
    """The WaveFake generator directory a `file_id` sits in, or None.

    ⚠️ Matched against `SUFFIXES` rather than by taking a fixed path depth.
    WaveFake also ships `jsut_*` (Japanese, a different corpus) and a
    Common-Voice-prompt directory, and a depth-based split would hand those to
    the caller as if they were LJSpeech vocoders.
    """
    for part in relpath_of(file_id).split("/"):
        if part in SUFFIXES:
            return part
    return None


def _expected_name(utterance: str, vocoder: str) -> str:
    return f"{utterance}{SUFFIXES[vocoder]}.wav"


def pair_table(files: pd.DataFrame) -> pd.DataFrame:
    """Every utterance that resolves to all eight files, one row per file.

    Columns: `pair_id`, `utterance`, `role`, `file_id`, `source_name`, `path`.
    `role` is `REAL_ROLE` or one of `VOCODERS`; `pair_id` is the utterance id,
    which is already unique and already meaningful -- an opaque counter would
    make the artifact unreadable for no gain.

    ⚠️ Incomplete utterances are **dropped and counted**, in `df.attrs`, not
    raised on. A partially-fetched archive is an ordinary state of this corpus
    (docs/EDA/08 probes wave by wave) and R2 says a failure is a row. The raise
    is `draw_pairs`' job, where a thin population would corrupt a published
    number.
    """
    wanted = files[files["source_name"].isin([REAL_SOURCE, FAKE_SOURCE])].copy()
    wanted["utterance"] = [utterance_of(f) for f in wanted["file_id"]]
    wanted = wanted[wanted["utterance"].notna()]

    # 🔴 The role, and the two ways it can be wrong. A real row is any
    # `ljspeech` row; a fake row is one whose directory is in `SUFFIXES` **and**
    # whose filename matches that directory's convention. The second half is
    # what catches a suffix map that has drifted from the archive: without it a
    # renamed file would simply vanish from its vocoder's column.
    roles: list[str | None] = []
    for row in wanted.itertuples():
        if row.source_name == REAL_SOURCE:
            roles.append(REAL_ROLE)
            continue
        vocoder = vocoder_of(row.file_id)
        if vocoder is None:
            roles.append(None)
            continue
        name = row.path.rsplit("/", 1)[-1]
        roles.append(vocoder if name == _expected_name(row.utterance, vocoder)
                     else None)
    wanted["role"] = roles
    wanted = wanted[wanted["role"].notna()]

    # A duplicate role within one utterance means two files claim to be the
    # same vocoder's rendering of the same text, which the count below would
    # otherwise paper over by reaching eight the wrong way.
    dupes = wanted.duplicated(subset=["utterance", "role"]).sum()
    if dupes:
        raise IncompletePair(
            f"{dupes} (utterance, role) pair(s) appear more than once. Two "
            f"files claim to be the same vocoder's rendering of the same "
            f"utterance; the join key is wrong, not the archive")

    sizes = wanted.groupby("utterance")["role"].nunique()
    complete = set(sizes[sizes == PAIR_SIZE].index)
    out = wanted[wanted["utterance"].isin(complete)].copy()
    out["pair_id"] = out["utterance"]
    out = out[["pair_id", "utterance", "role", "file_id", "source_name", "path"]]
    out = out.sort_values(["utterance", "role"]).reset_index(drop=True)

    out.attrs["n_utterances"] = int(len(sizes))
    out.attrs["n_complete"] = len(complete)
    out.attrs["n_incomplete"] = int(len(sizes) - len(complete))
    return out


def draw_pairs(pairs: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """`n` complete utterances, seeded, all eight rows of each.

    🔴 Refuses rather than truncating. `n` sets the precision of every
    per-vocoder mean B1 publishes, so quietly returning 40 utterances because
    the archive was half-fetched would put a number in the document that reads
    as if it came from the requested population.
    """
    available = pairs["utterance"].nunique()
    if available < n:
        raise IncompletePair(
            f"asked for {n} complete pairs and only {available} exist. "
            f"Either the archive is partially fetched or SUFFIXES has drifted "
            f"from it -- check the per-vocoder counts before lowering n")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.sort(pairs["utterance"].unique()), size=n,
                        replace=False)
    out = pairs[pairs["utterance"].isin(set(chosen))].reset_index(drop=True)
    # Every drawn utterance contributes all eight rows, or the draw is not what
    # it says. Cheap, and it is the assertion that would have caught the
    # `_gen.wav` join.
    counts = out.groupby("utterance")["role"].nunique()
    if not (counts == PAIR_SIZE).all():
        raise IncompletePair(
            f"drew {n} utterances and {int((counts != PAIR_SIZE).sum())} of "
            f"them are short of {PAIR_SIZE} files")
    out.attrs["seed"] = seed
    out.attrs["n_pairs"] = n
    return out


def difference_images(vectors: Mapping[str, np.ndarray],
                      pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """`vocoder -> E[mel(fake) - mel(real)]`, the `[128]` difference image.

    `vectors` maps `file_id` to that file's `ltas`, for **one** plane. The mean
    is over pairs, and each pair contributes `fake - real` computed on its own
    two files -- not `mean(fake) - mean(real)` over the two populations. On a
    complete pairing the two agree arithmetically; they stop agreeing the
    moment a file is missing from one side, and the paired form is the one that
    degrades honestly.

    ⚠️ `ltas` is already in dB relative to each file's own peak
    (`eda.extract.vectors`), so this difference is free of the level offset
    between a vocoder's output and the original. That is deliberate: mastering
    level is `level.py`'s finding and would otherwise dominate the image.
    """
    by_utterance: dict[str, dict[str, np.ndarray]] = {}
    for row in pairs.itertuples():
        vec = vectors.get(row.file_id)
        if vec is None:
            continue
        by_utterance.setdefault(row.utterance, {})[row.role] = vec

    out: dict[str, np.ndarray] = {}
    for vocoder in VOCODERS:
        deltas = [np.asarray(roles[vocoder], dtype=np.float64)
                  - np.asarray(roles[REAL_ROLE], dtype=np.float64)
                  for roles in by_utterance.values()
                  if vocoder in roles and REAL_ROLE in roles]
        if not deltas:
            continue
        stack = np.vstack(deltas)
        # NaN bands are real -- a band that never moves has no statistic
        # (`eda.extract.vectors`) -- so they are skipped per band rather than
        # poisoning the whole image.
        with np.errstate(invalid="ignore"):
            out[vocoder] = np.nanmean(stack, axis=0)
    return out


def surviving_fraction(native: Mapping[str, np.ndarray],
                       chain: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Per vocoder: how much of the artifact is still there after 16 kHz.

    The ratio is of **squared** difference summed over bands -- the energy of
    the difference image -- at `chain` against `native`. Both banks are pinned
    to an absolute 0-8000 Hz (`eda.extract.vectors`), so the two images are
    band-for-band comparable and the ratio is a subtraction of the same thing.

    🔴 **A ratio near 1.0 does not mean the chain removed nothing.** It means
    the chain removed nothing *below 8 kHz*, which is all a 0-8000 Hz bank can
    see. The artifact above the cut is invisible here by construction and is
    the scalar half's job -- `hf_ratio_8k` on the native plane, which B1 reports
    beside this table for exactly that reason.
    """
    rows = []
    for vocoder in VOCODERS:
        a, b = native.get(vocoder), chain.get(vocoder)
        if a is None or b is None:
            continue
        with np.errstate(invalid="ignore"):
            e_native = float(np.nansum(np.asarray(a) ** 2))
            e_chain = float(np.nansum(np.asarray(b) ** 2))
        rows.append({
            "vocoder": vocoder,
            "energy_native": e_native,
            "energy_chain": e_chain,
            "surviving_fraction": e_chain / e_native if e_native > 0 else np.nan,
            "peak_band_native": int(np.nanargmax(np.abs(a))) if e_native > 0 else -1,
            "peak_band_chain": int(np.nanargmax(np.abs(b))) if e_chain > 0 else -1,
        })
    return pd.DataFrame(rows)


def family_correlation(images: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """The 7x7 correlation between difference images.

    This is the `artifact_family` question, and it is the reason B1 outranks
    every other item left: docs/validation/01 splits folds on **artifact
    family, not model name**, because model names over-count. Two vocoders
    whose difference images correlate at 0.99 are one family for the fold
    builder's purposes however differently they are named.
    """
    present = [v for v in VOCODERS if v in images]
    data = np.vstack([np.asarray(images[v], dtype=np.float64) for v in present])
    # Bands that are NaN for any vocoder are dropped for every vocoder, so all
    # 21 correlations are computed over one common support. Pairwise deletion
    # would make the matrix's entries incomparable with each other.
    keep = ~np.isnan(data).any(axis=0)
    if keep.sum() < 2:
        return pd.DataFrame(np.nan, index=present, columns=present)
    matrix = np.corrcoef(data[:, keep])
    return pd.DataFrame(matrix, index=present, columns=present)
