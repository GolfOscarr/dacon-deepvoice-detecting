"""B1's pairing and reduction: `eda.analyze.pairs`, docs/EDA/10 step 1.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda.analyze import pairs as P
from eda.analyze.pairs import (IncompletePair, PAIR_SIZE, REAL_ROLE, SUFFIXES,
                               VOCODERS, difference_images, draw_pairs,
                               family_correlation, pair_table,
                               surviving_fraction, utterance_of, vocoder_of)


def _files(utterances, *, drop=(), rename=None) -> pd.DataFrame:
    """A census-shaped frame: the real file plus all seven fakes per utterance.

    `drop` removes `(utterance, vocoder)` pairs; `rename` replaces one
    vocoder's suffix, which is how the archive actually differs from the naive
    expectation.
    """
    rows = []
    for utt in utterances:
        rows.append({"file_id": f"ljspeech:LJSpeech-1.1/wavs/{utt}.wav",
                     "source_name": "ljspeech",
                     "path": f"ljspeech/mdc-1.1/LJSpeech-1.1/wavs/{utt}.wav"})
        for vocoder in VOCODERS:
            if (utt, vocoder) in drop:
                continue
            suffix = (rename[vocoder] if rename and vocoder in rename
                      else SUFFIXES[vocoder])
            name = f"{utt}{suffix}.wav"
            rel = f"generated_audio/{vocoder}/{name}"
            rows.append({"file_id": f"wavefake:{rel}", "source_name": "wavefake",
                         "path": f"wavefake/zenodo-5642694/{rel}"})
    return pd.DataFrame(rows)


UTTS = [f"LJ001-{i:04d}" for i in range(1, 6)]


# --------------------------------------------------------------------------- #
# the three filename conventions
# --------------------------------------------------------------------------- #

def test_all_seven_vocoders_resolve_including_the_two_odd_ones():
    """🔴 The defect this module exists to prevent. WaveFake names its files
    `_gen.wav` in five directories, `_generated.wav` in `ljspeech_hifiGAN` and
    bare `.wav` in `ljspeech_waveglow`, and nothing in the archive announces it.

    A join written as `f"{utterance}_gen.wav"` resolves five of seven and drops
    HiFi-GAN and WaveGlow **silently** -- five columns where the plan asked for
    seven, in a table nobody counts.

    Mutation: `SUFFIXES` given a uniform `"_gen"`. Every utterance then falls
    two files short of `PAIR_SIZE` and `n_complete` goes to 0, which is a
    refusal rather than a thin table.
    """
    assert set(SUFFIXES.values()) == {"_gen", "_generated", ""}, (
        "three conventions, not one -- see the module docstring")
    table = pair_table(_files(UTTS))
    assert table.attrs["n_complete"] == len(UTTS)
    assert set(table["role"]) == {REAL_ROLE, *VOCODERS}
    assert (table.groupby("utterance")["role"].nunique() == PAIR_SIZE).all()


def test_a_suffix_that_has_drifted_from_the_archive_is_a_refusal_not_a_short_row():
    """⚠️ The failure mode if WaveFake ever repackages: the directory is still
    there and still full, but its filenames no longer match `SUFFIXES`. Without
    the per-file name check in `pair_table` those files keep their vocoder role
    by directory alone and the table looks complete.

    Mutation: the `name == _expected_name(...)` test replaced by `True`.
    """
    files = _files(UTTS, rename={"ljspeech_hifiGAN": "_v2"})
    table = pair_table(files)
    assert table.attrs["n_complete"] == 0
    assert table.attrs["n_incomplete"] == len(UTTS)


def test_an_incomplete_utterance_is_dropped_and_counted_never_averaged():
    """R2: a failure is a row. A half-fetched archive is an ordinary state here
    (docs/EDA/08 probes wave by wave), so the short utterance leaves the table
    and is counted in `attrs` rather than raising.

    Mutation: the `sizes == PAIR_SIZE` filter removed -- the short utterance
    then contributes to six vocoders' means and not to the seventh, and the
    per-vocoder table compares populations that differ in content.
    """
    files = _files(UTTS, drop={(UTTS[0], "ljspeech_melgan")})
    table = pair_table(files)
    assert table.attrs["n_complete"] == len(UTTS) - 1
    assert table.attrs["n_incomplete"] == 1
    assert UTTS[0] not in set(table["utterance"])


def test_the_jsut_and_common_voice_directories_are_not_ljspeech_vocoders():
    """⚠️ WaveFake also ships `jsut_*` (a different corpus, Japanese) and a
    Common-Voice-prompt directory. A vocoder resolved by path *depth* would
    hand those back as if they were LJSpeech vocoders.

    Mutation: `vocoder_of` made to return `relpath.split("/")[1]`.
    """
    assert vocoder_of("wavefake:generated_audio/jsut_parallel_wavegan/x.wav") is None
    assert vocoder_of("wavefake:generated_audio/ljspeech_melgan/x.wav") == "ljspeech_melgan"
    assert utterance_of("wavefake:generated_audio/ljspeech_melgan/LJ001-0001_gen.wav") == "LJ001-0001"
    assert utterance_of("ljspeech:LJSpeech-1.1/wavs/nope.wav") is None


def test_two_files_claiming_one_vocoder_of_one_utterance_is_fatal():
    """A duplicate role reaches `PAIR_SIZE` the wrong way, so the completeness
    count would pass while one vocoder's mean is computed over a doubled file.

    Mutation: the `duplicated` check removed.
    """
    files = _files(UTTS)
    files = pd.concat([files, files.iloc[[1]]], ignore_index=True)
    with pytest.raises(IncompletePair, match="more than once"):
        pair_table(files)


# --------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------- #

def test_the_draw_refuses_to_truncate():
    """🔴 `n` sets the precision of every per-vocoder mean B1 publishes.
    Returning 3 utterances because only 3 exist would put a number in the
    document that reads as if it came from the requested population.

    Mutation: the `available < n` guard removed -- `rng.choice` then raises a
    numpy error about sample size, which is the same refusal with a message
    that does not say what is actually wrong.
    """
    table = pair_table(_files(UTTS))
    with pytest.raises(IncompletePair, match="only 5 exist"):
        draw_pairs(table, 10)


def test_the_draw_is_seeded_and_carries_whole_pairs():
    table = pair_table(_files(UTTS))
    a, b = draw_pairs(table, 3, seed=1), draw_pairs(table, 3, seed=1)
    assert list(a["file_id"]) == list(b["file_id"])
    assert list(a["file_id"]) != list(draw_pairs(table, 3, seed=2)["file_id"])
    assert len(a) == 3 * PAIR_SIZE
    assert (a.groupby("utterance")["role"].nunique() == PAIR_SIZE).all()


# --------------------------------------------------------------------------- #
# the reduction
# --------------------------------------------------------------------------- #

def _vectors(table, fake_offset, width=8):
    """Real rows are a fixed spectrum; each fake is that plus its own offset."""
    out = {}
    real = np.linspace(-40.0, 0.0, width)
    for row in table.itertuples():
        out[row.file_id] = (real if row.role == REAL_ROLE
                            else real + fake_offset[row.role])
    return out


def test_the_difference_is_paired_per_utterance_not_between_populations():
    """🔴 `E[fake - real]` over pairs, not `mean(fake) - mean(real)` over two
    populations. They agree arithmetically on a complete pairing and stop
    agreeing the moment one side loses a file -- and the paired form is the one
    that degrades honestly, which is the whole premise of B1.

    Mutation: `difference_images` rewritten as the difference of the two means.
    Here the real files differ per utterance, so the unpaired form picks up the
    content variation and the paired form cancels it exactly.

    ⚠️ The per-utterance offsets must **vary**, and that is not decoration.
    Measured: with one offset per vocoder the mutant `deltas[:1]` -- average
    the first pair and call it the mean -- **survived**, because one pair and
    five pairs give the same answer when every pair is identical. Varying the
    offset makes the mean over pairs differ from any single pair, which is the
    only thing that distinguishes an average from a sample.
    """
    table = pair_table(_files(UTTS))
    width = 8
    base_offsets = {v: float(i + 1) for i, v in enumerate(VOCODERS)}
    #: Distinct per utterance and summing to a mean that is not any member.
    per_utterance = {utt: float(k) for k, utt in enumerate(UTTS)}
    vectors = {}
    rng = np.random.default_rng(0)
    for utt in UTTS:
        # Per-utterance content, which is exactly what pairing removes.
        base = rng.normal(size=width) * 10.0
        for row in table[table["utterance"] == utt].itertuples():
            vectors[row.file_id] = (
                base if row.role == REAL_ROLE
                else base + base_offsets[row.role] + per_utterance[utt])
    images = difference_images(vectors, table)
    spread = float(np.mean(list(per_utterance.values())))
    assert spread != per_utterance[UTTS[0]], (
        "the first pair must not equal the mean, or `deltas[:1]` survives")
    for vocoder, offset in base_offsets.items():
        assert np.allclose(images[vocoder], offset + spread), (
            "the per-utterance content must cancel exactly, and the result "
            "must be the mean over every pair rather than the first one")


def test_a_file_missing_from_the_vectors_drops_its_pair_not_its_vocoder():
    table = pair_table(_files(UTTS))
    vectors = _vectors(table, {v: 1.0 for v in VOCODERS})
    victim = table[(table["utterance"] == UTTS[0])
                   & (table["role"] == REAL_ROLE)]["file_id"].iloc[0]
    del vectors[victim]
    images = difference_images(vectors, table)
    assert set(images) == set(VOCODERS), "every vocoder still has an image"
    assert np.allclose(images["ljspeech_melgan"], 1.0)


def test_surviving_fraction_is_an_energy_ratio_over_a_shared_band_definition():
    """⚠️ The ratio is only a subtraction of the same thing because the mel bank
    is pinned to an absolute 0-8000 Hz on **both** planes
    (`eda.extract.vectors`). The docstring's warning is the point: a fraction
    near 1.0 means nothing was removed *below 8 kHz*, which is all this can see.

    Mutation: `energy_chain / energy_native` inverted.
    """
    native = {v: np.array([2.0, 0.0, 0.0, 0.0]) for v in VOCODERS}
    chain = {v: np.array([1.0, 0.0, 0.0, 0.0]) for v in VOCODERS}
    out = surviving_fraction(native, chain).set_index("vocoder")
    assert np.allclose(out["surviving_fraction"], 0.25), "energy, so (1/2)^2"
    assert (out["peak_band_native"] == 0).all()


def test_flat_bands_stay_nan_and_do_not_poison_the_image_or_the_matrix():
    """A band that never moves has no statistic and `eda.extract.vectors` leaves
    it NaN deliberately. The reduction must skip those bands, not propagate
    them over the whole `[128]`.

    Mutation: `np.nanmean` -> `np.mean` in `difference_images`; the image
    becomes all-NaN and `family_correlation` returns an all-NaN matrix.
    """
    table = pair_table(_files(UTTS))
    vectors = _vectors(table, {v: 1.0 for v in VOCODERS}, width=4)
    victim = table["file_id"].iloc[0]
    vectors[victim] = np.array([np.nan, 1.0, 2.0, 3.0])
    images = difference_images(vectors, table)
    assert np.isfinite(images["ljspeech_full_band_melgan"]).all(), (
        "one NaN in one file must not blank the vocoder's whole image")


def test_family_correlation_uses_one_common_support_for_every_entry():
    """⚠️ Pairwise deletion would compute each of the 21 correlations over a
    different set of bands, and the entries would not be comparable with each
    other -- which is the only thing a family assignment reads them for.

    Mutation: the `keep` mask dropped and `np.corrcoef` given the raw matrix --
    any NaN band then makes the whole matrix NaN.
    """
    width = 6
    base = np.linspace(0.0, 1.0, width)
    images = {v: base * (i + 1) for i, v in enumerate(VOCODERS)}
    images["ljspeech_waveglow"] = images["ljspeech_waveglow"].copy()
    images["ljspeech_waveglow"][0] = np.nan
    matrix = family_correlation(images)
    assert list(matrix.index) == list(VOCODERS), "stable, declared axis order"
    assert np.isfinite(matrix.to_numpy()).all()
    # Scalar multiples of one shape are one family, and the matrix must say so.
    assert np.allclose(matrix.to_numpy(), 1.0)


def test_the_axes_are_the_declared_order_not_whatever_the_dict_iterates():
    """A correlation matrix whose axes move between runs cannot be diffed, and
    `artifact_family` is read off exactly this table.

    Mutation: `present` built from `images` instead of `VOCODERS`.
    """
    width = 5
    rng = np.random.default_rng(3)
    images = {v: rng.normal(size=width) for v in reversed(VOCODERS)}
    assert list(family_correlation(images).index) == list(VOCODERS)
