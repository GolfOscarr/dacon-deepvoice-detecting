"""The S tier: the two planes, and the extractors that run on both.

Separate from `test_eda.py` because the population is different -- everything
here decodes audio, and the fixtures are signals with a known answer rather than
a corpus with a known shape.

Every invariant below is paired with a mutation that was **observed** to fail;
`scripts/mutate_eda.py` holds them. A green suite is not evidence that a check
can fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from eda.extract import SIGNAL, ExtractorError, run_signal
from eda.extract.level import DB_FLOOR, level_row
from eda.extract.spectral import spectral_row
from eda.extract.timing import SILENCE_DBFS, timing_row
from eda.planes import CHAIN, NATIVE, NativeRateError, Plane, load_planes, native_rate
from models.audio import prepare_waveform
from models.config import AudioConfig
from training.render import load_audio


def _tone(path, hz=440.0, seconds=2.0, sr=44100, amp=0.5, channels=1):
    t = np.arange(int(seconds * sr)) / sr
    one = (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    data = one if channels == 1 else np.stack([one] * channels, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr)
    return path


# --------------------------------------------------------------------------
# the two planes
# --------------------------------------------------------------------------

def test_the_chain_plane_is_identical_to_what_the_training_path_produces(tmp_path):
    """🔴 The whole value of the chain plane is that it is *the same signal the
    model receives*. Not similar -- the same. Decode both ways and compare the
    arrays, because an EDA that measures a different signal than the model gets
    is measuring nothing (docs/EDA/00 section 2).

    Mutation: resample with anything but `resample_poly_to`, or skip
    `prepare_waveform`, and this fails."""
    path = _tone(tmp_path / "a.wav", channels=2)
    planes = load_planes(path)

    ref = load_audio(path, sample_rate=16_000)
    ref_mono = prepare_waveform(torch.from_numpy(ref)[None], AudioConfig())[0].numpy()
    assert np.array_equal(planes[CHAIN].wav[0], ref_mono)


def test_the_native_plane_is_not_resampled(tmp_path):
    """The point of `native` is the file as published. 44.1 kHz in, 44.1 kHz
    out, both channels intact."""
    path = _tone(tmp_path / "a.wav", seconds=1.0, sr=44100, channels=2)
    planes = load_planes(path)
    assert planes[NATIVE].sample_rate == 44100
    assert planes[NATIVE].wav.shape == (2, 44100)
    assert planes[CHAIN].sample_rate == 16_000
    assert planes[CHAIN].wav.shape == (1, 16_000)


def test_a_declared_rate_that_disagrees_with_the_file_is_refused(tmp_path):
    """🔴 The native plane is produced by asking `load_audio` for the file's own
    rate, and `resample_poly_to` short-circuits only when the two match. Hand it
    a wrong rate and it resamples silently, and every `native` statistic is
    measured on an upsampled signal -- a whole plane that looks fine and answers
    a question nobody asked.

    Mutation: drop the comparison in `native_rate` and this passes while
    `load_planes(path, declared_sr=22050)` returns a 22.05 kHz 'native' plane
    for a 44.1 kHz file."""
    path = _tone(tmp_path / "a.wav", seconds=0.5)
    assert native_rate(path) == 44100
    assert native_rate(path, declared=44100) == 44100
    with pytest.raises(NativeRateError, match="44100 Hz and files.parquet records 22050"):
        native_rate(path, declared=22050)
    with pytest.raises(NativeRateError):
        load_planes(path, declared_sr=22050)


def test_a_file_nothing_can_read_is_a_finding_not_a_crash(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio at all")
    with pytest.raises(NativeRateError, match="could read a sample rate"):
        native_rate(path)


def test_the_channel_policy_comes_from_the_config_not_from_a_local_downmix(tmp_path):
    """`channels: mid_side` is a live option (09 B8). A local `.mean(0)` would
    silently ignore it, and the EDA would then describe a chain the model does
    not have."""
    sr, n = 16_000, 8_000
    left = np.full(n, 0.5, dtype=np.float32)
    right = np.full(n, -0.1, dtype=np.float32)
    path = tmp_path / "a.wav"
    sf.write(path, np.stack([left, right], axis=-1), sr)

    downmix = load_planes(path, audio=AudioConfig(channels="downmix"))[CHAIN]
    only_left = load_planes(path, audio=AudioConfig(channels="left"))[CHAIN]
    assert downmix.wav[0].mean() == pytest.approx(0.2, abs=1e-3)
    assert only_left.wav[0].mean() == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------
# running both planes
# --------------------------------------------------------------------------

def test_every_signal_column_is_emitted_once_per_plane(tmp_path):
    row = run_signal(load_planes(_tone(tmp_path / "a.wav", seconds=1.0)))
    for base in ("rms_dbfs", "silence_ratio", "effective_bandwidth_hz"):
        assert f"{base}_{NATIVE}" in row and f"{base}_{CHAIN}" in row
    # and nothing arrives unsuffixed, which would collide across planes
    assert not any(c in row for c in ("rms_dbfs", "silence_ratio"))


def test_an_extractor_that_fails_on_one_plane_does_not_fail_the_other():
    """A native plane can be 8-channel where the chain plane is mono, so the two
    genuinely can differ in whether they work. One `*_ok` for both would hide
    which one broke."""
    planes = {NATIVE: Plane(NATIVE, np.zeros((1, 0), dtype=np.float32), 44100),
              CHAIN: Plane(CHAIN, np.full((1, 16_000), 0.1, dtype=np.float32), 16_000)}
    row = run_signal(planes, names=("level",))
    assert row["level_ok_native"] is False
    assert row["level_error_native"] == "empty waveform"
    assert row["level_ok_chain"] is True


def test_a_signal_extractor_cannot_be_told_which_plane_it_is_on():
    """🔴 R1 asks for the same statistic computed twice. An extractor that knew
    which time it was would quietly make it two statistics -- so the registry
    refuses the signature, and this is the guard, not a code review."""
    with pytest.raises(ExtractorError, match="take exactly 2"):
        SIGNAL.add("peeker", ("x", "peeker_ok"), "peeker_ok",
                   lambda wav, sample_rate, plane: {"x": 1, "peeker_ok": True})
    with pytest.raises(ExtractorError, match=r"\*args/\*\*kwargs"):
        SIGNAL.add("variadic", ("x", "variadic_ok"), "variadic_ok",
                   lambda *a: {"x": 1, "variadic_ok": True})


# --------------------------------------------------------------------------
# level
# --------------------------------------------------------------------------

def test_level_reads_a_known_signal():
    sr = 16_000
    wav = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)[None]
    row = level_row(wav, sr)
    assert row["peak_dbfs"] == pytest.approx(-6.02, abs=0.05)      # 0.5 -> -6 dBFS
    assert row["rms_dbfs"] == pytest.approx(-9.03, abs=0.05)       # sine: peak - 3 dB
    assert row["crest_factor_db"] == pytest.approx(3.01, abs=0.05)
    assert row["dc_offset"] == pytest.approx(0.0, abs=1e-6)
    assert row["clipping_ratio"] == 0.0
    assert row["stat_t_pwr"] == pytest.approx(row["stat_t_rms"] ** 2, rel=1e-9)


def test_silence_reports_the_floor_and_not_negative_infinity():
    """🔴 Parquet holds -inf happily. Every consumer of the column does not: one
    digitally-silent file makes a source's mean level -inf, and the whole
    distribution disappears into it.

    Mutation: return `20*log10(0)` unguarded and the assertion on the mean
    below fails with -inf."""
    row = level_row(np.zeros((1, 1000), dtype=np.float32), 16_000)
    assert row["peak_dbfs"] == DB_FLOOR and row["rms_dbfs"] == DB_FLOOR
    assert np.isfinite([row["peak_dbfs"], row["rms_dbfs"]]).all()
    assert np.mean([row["rms_dbfs"], -20.0]) > -100


def test_dc_offset_and_clipping_are_measured_not_assumed():
    n = 10_000
    shifted = np.full((1, n), 0.25, dtype=np.float32)
    assert level_row(shifted, 16_000)["dc_offset"] == pytest.approx(0.25, abs=1e-6)

    wav = np.zeros((1, n), dtype=np.float32)
    wav[0, : n // 10] = 1.0
    assert level_row(wav, 16_000)["clipping_ratio"] == pytest.approx(0.1, abs=1e-9)


def test_level_measures_every_channel_not_a_downmix():
    """A downmix hides a silent channel, and 'one channel is silent' is a real
    property of scraped stereo music that `mid_side` would turn into signal."""
    n = 1000
    wav = np.stack([np.full(n, 0.4, dtype=np.float32), np.zeros(n, dtype=np.float32)])
    # downmixed this would be a constant 0.2; over all samples it is not
    assert level_row(wav, 16_000)["dc_offset"] == pytest.approx(0.2, abs=1e-6)
    assert level_row(wav, 16_000)["peak_dbfs"] == pytest.approx(-7.96, abs=0.05)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def test_timing_finds_lead_and_tail_silence():
    sr = 16_000
    wav = np.zeros((1, 10 * sr), dtype=np.float32)
    wav[0, 2 * sr: 7 * sr] = 0.5           # 2 s lead, 5 s loud, 3 s tail
    row = timing_row(wav, sr)
    assert row["lead_silence_s"] == pytest.approx(2.0, abs=0.05)
    assert row["tail_silence_s"] == pytest.approx(3.0, abs=0.05)
    assert row["silence_ratio"] == pytest.approx(0.5, abs=0.01)
    assert row["longest_valid_span_s"] == pytest.approx(5.0, abs=0.05)
    assert row["n_valid_spans_ge_4s"] == 1
    assert row["duration_s_decoded"] == pytest.approx(10.0)


def test_a_span_shorter_than_the_sampler_floor_does_not_count():
    """🔴 `rirs-isotropic` in one assertion: 314 of 417 files sat below the 4 s
    floor, so the group 'contributed nothing while still counting toward the
    file tally'. A file with three 3-second bursts has zero valid windows."""
    sr = 16_000
    wav = np.zeros((1, 20 * sr), dtype=np.float32)
    for start in (0, 6, 12):
        wav[0, start * sr: (start + 3) * sr] = 0.5
    row = timing_row(wav, sr)
    assert row["longest_valid_span_s"] == pytest.approx(3.0, abs=0.05)
    assert row["n_valid_spans_ge_4s"] == 0


def test_the_silence_threshold_is_absolute_not_relative():
    """🔴 A relative threshold rescales with the content, so a quiet field
    recording and a loud one report the same silence ratio. What this feeds is
    the sampler's ability to find a window with audio in it, and the sampler
    does not renormalize first.

    Mutation: make the threshold `peak_db + SILENCE_DBFS` and the quiet file
    below reports 0.0 silence instead of 1.0."""
    sr = 16_000
    quiet = np.full((1, 4 * sr), 10 ** (-70 / 20), dtype=np.float32)   # -70 dBFS
    loud = np.full((1, 4 * sr), 10 ** (-20 / 20), dtype=np.float32)    # -20 dBFS
    assert SILENCE_DBFS == -50.0
    assert timing_row(quiet, sr)["silence_ratio"] == 1.0
    assert timing_row(loud, sr)["silence_ratio"] == 0.0


def test_a_file_shorter_than_one_chunk_is_a_row_not_a_failure():
    row = timing_row(np.full((1, 100), 0.5, dtype=np.float32), 16_000)
    assert row["timing_ok"] is True
    assert row["n_valid_spans_ge_4s"] == 0 and row["silence_ratio"] == 1.0


# --------------------------------------------------------------------------
# spectral
# --------------------------------------------------------------------------

def test_bandwidth_finds_a_low_pass_edge():
    """🔴 This is the measurement the music-head strategy rests on. A file
    generated at 16 kHz and republished at 44.1 has nothing above 8 kHz, and
    that is exactly what a fake-music source looks like."""
    sr, n = 44100, 44100
    rng = np.random.default_rng(0)
    full = rng.standard_normal(n).astype(np.float32) * 0.1

    # band-limited by resampling down and back up, the way a republished file is
    from training.render import resample_poly_to
    narrow = resample_poly_to(resample_poly_to(full[None], sr, 16_000), 16_000, sr)

    wide_bw = spectral_row(full[None], sr)["effective_bandwidth_hz"]
    narrow_bw = spectral_row(narrow, sr)["effective_bandwidth_hz"]
    assert wide_bw > 20_000
    # ⚠️ ~10.2 kHz, not 8: `resample_poly`'s anti-alias filter is not a brick
    # wall and its skirt is still within 60 dB of the peak a couple of kHz
    # above the cutoff. The statistic separates the two cases by more than 2x,
    # which is what it is for -- but a reader must not read `effective_
    # bandwidth_hz` as "the generator's sample rate over two".
    assert narrow_bw < 12_000
    assert wide_bw > 2 * narrow_bw
    # 🔴 and `hf_ratio_8k` is the one that says it cleanly. ⚠️ Measured at
    # **0.005**, not 0: `resample_poly`'s skirt leaves half a percent of the
    # energy above 8 kHz even for a file that genuinely has no content there.
    # The separating threshold for a republished fake is therefore around a
    # percent, not zero, and a check written as `== 0` would find nothing.
    narrow_hf = spectral_row(narrow, sr)["hf_ratio_8k"]
    wide_hf = spectral_row(full[None], sr)["hf_ratio_8k"]
    assert narrow_hf < 0.01
    assert wide_hf > 0.5
    assert wide_hf > 50 * narrow_hf


def test_near_nyquist_ratio_does_not_see_a_cutoff_far_below_its_bands():
    """🔴 Recorded because the docstring said otherwise and the measurement
    disagreed. The statistic compares [0.94, 0.99]*Nyquist against
    [0.80, 0.94]*Nyquist, so a 44.1 kHz file band-limited to 8 kHz -- exactly
    what a 16 kHz generator republished at 44.1 looks like -- has *both* bands
    deep in the stopband and reports the skirt's shape, not the cutoff.

    This is a limit of the column, not a defect to fix: it is the right
    statistic for a cutoff near Nyquist. `hf_ratio_8k` covers the other case,
    and a reader who confuses them concludes a fingerprint survives when it
    does not."""
    sr, n = 44100, 44100
    from training.render import resample_poly_to
    full = (np.random.default_rng(0).standard_normal(n).astype(np.float32) * 0.1)
    narrow = resample_poly_to(resample_poly_to(full[None], sr, 16_000), 16_000, sr)
    assert spectral_row(narrow, sr)["near_nyquist_ratio"] > 0.1
    assert spectral_row(narrow, sr)["hf_ratio_8k"] < 0.01


def test_flatness_separates_a_tone_from_noise():
    sr = 16_000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[None]
    noise = (0.1 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32)[None]
    assert spectral_row(tone, sr)["spectral_flatness"] < 0.01
    assert spectral_row(noise, sr)["spectral_flatness"] > 0.9


def test_zero_bins_are_floored_so_flatness_is_computed_not_abandoned():
    """Exactly-zero PSD bins make `log(0)` -inf and the flatness 0.0 -- which is
    already this module's value for *silent*. A computed 0.0 and a gave-up 0.0
    would be indistinguishable, so the bins are floored 120 dB down instead.

    ⚠️ They are rare, and the honest note is in the module: a 16-bit tone behind
    a 5 s digital-silence pad -- the TTS shape, and the likeliest real case --
    produces **none**, because Welch averages over segments. The construction
    below is the one that does: a lone impulse in an otherwise zero file, whose
    Hann window suppresses it to nothing and leaves all-zero segments.

    Mutation: filter `psd > 0` and fall back to 0.0 when a bin was dropped.
    The row then reports exactly 0.0 for a file that is not silent."""
    sr = 16_000
    impulse = np.zeros((1, sr), dtype=np.float32)
    impulse[0, 0] = 1.0
    row = spectral_row(impulse, sr)
    assert row["spectral_error"] is None, "not silent -- there is a sample in it"
    assert row["spectral_flatness"] > 0.0

    # and the everyday case is unaffected: an impulse in the middle of the file
    # is windowed properly, has no zero bins, and reads as the flat spectrum it is
    mid = np.zeros((1, sr), dtype=np.float32)
    mid[0, sr // 2] = 1.0
    assert spectral_row(mid, sr)["spectral_flatness"] > 0.9


def test_band_energies_are_fractions_that_sum_to_one_at_16k():
    """At 16 kHz the eight 1 kHz bands tile the whole spectrum, so they must
    account for all of it. Absolute powers would rank sources by mastering
    level, which `level` already reports."""
    sr = 16_000
    noise = (0.1 * np.random.default_rng(1).standard_normal(sr)).astype(np.float32)[None]
    row = spectral_row(noise, sr)
    total = sum(row[f"band_energy_{i}"] for i in range(8))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_a_silent_spectrum_is_reported_as_silent_not_as_a_decode_failure():
    row = spectral_row(np.zeros((1, 16_000), dtype=np.float32), 16_000)
    assert row["spectral_ok"] is True
    assert "silent" in row["spectral_error"]
    assert row["effective_bandwidth_hz"] == 0.0
    assert row["band_energy_0"] == 0.0


# --------------------------------------------------------------------------
# the draw, and the pass that consumes it
# --------------------------------------------------------------------------

def _signal_corpus(tmp_path, n_per_source=6):
    """Two sources in one pool, with enough rows to sample from."""
    from eda.config import EdaConfig, ProbeConfig, SampleConfig, SourceSpec
    from eda.driver import consolidate, probe_source

    root = tmp_path / "interim"
    rng = np.random.default_rng(0)
    for name in ("src-a", "src-b"):
        for i in range(n_per_source):
            path = root / f"{name}/v1" / f"{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, (rng.standard_normal(8_000) * 0.1).astype(np.float32), 16_000)
    cfg = EdaConfig(
        root=root, out=tmp_path / "out",
        sources=(SourceSpec(name="src-a", pool="A", root="src-a/v1", suffixes=(".wav",)),
                 SourceSpec(name="src-b", pool="A", root="src-b/v1", suffixes=(".wav",))),
        probe=ProbeConfig(workers=2, shard_size=4),
        sample=SampleConfig(seed=0, per_stratum=3, stratify_by=("source_name",),
                            full_pools=()),
    )
    for src in cfg.sources:
        probe_source(cfg, src)
    consolidate(cfg, "A")
    return cfg


def test_the_draw_is_stratified_seeded_and_recorded(tmp_path):
    from eda.driver import load_files
    from eda.sample import draw, load_sample

    cfg = _signal_corpus(tmp_path)
    files = load_files(cfg, "A")
    sample = draw(cfg, files, "A")

    assert sample.n_population == 12 and sample.n_drawn == 6    # 3 per source
    drawn = files.set_index("file_id").loc[list(sample.file_ids)]
    assert drawn["source_name"].value_counts().to_dict() == {"src-a": 3, "src-b": 3}
    # and it is on disk, which is what makes it reproducible from originals
    assert load_sample(cfg, "A").file_ids == sample.file_ids


def test_the_same_seed_draws_the_same_files_whatever_the_row_order(tmp_path):
    """🔴 `files.parquet`'s row order is an artifact of shard scheduling --
    thread completion order, and which parts happened to exist. A draw taken in
    table order is reproducible only until someone re-probes one source.

    Mutation: drop the `sort_values("file_id")` and this fails."""
    from eda.driver import load_files
    from eda.sample import draw

    cfg = _signal_corpus(tmp_path)
    files = load_files(cfg, "A")
    first = draw(cfg, files, "A")
    shuffled = files.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = draw(cfg, shuffled, "A", force=True)
    assert first.file_ids == second.file_ids


def test_a_recorded_draw_is_not_silently_replaced(tmp_path):
    """🔴 The sample is part of the corpus definition (docs/EDA/00 §3).
    Redrawing invalidates every S-tier number published from the old draw, so it
    takes a flag rather than happening because a pass was re-run."""
    from eda.driver import load_files
    from eda.sample import SampleExists, draw

    cfg = _signal_corpus(tmp_path)
    files = load_files(cfg, "A")
    draw(cfg, files, "A")
    with pytest.raises(SampleExists, match="Re-read it rather than redrawing"):
        draw(cfg, files, "A")
    draw(cfg, files, "A", force=True)          # deliberate is allowed


def test_a_full_pool_is_measured_whole_not_sampled(tmp_path):
    """Pool D because five generator families are too few to sample from, pool E
    because its last surprise cost a corpus rebuild."""
    import dataclasses

    from eda.config import SampleConfig
    from eda.driver import load_files
    from eda.sample import draw

    cfg = _signal_corpus(tmp_path)
    cfg = dataclasses.replace(cfg, sample=SampleConfig(
        seed=0, per_stratum=3, stratify_by=("source_name",), full_pools=("A",)))
    sample = draw(cfg, load_files(cfg, "A"), "A")
    assert sample.full and sample.n_drawn == sample.n_population == 12


def test_a_draw_whose_files_have_gone_is_refused(tmp_path):
    """🔴 A recorded id missing from `files.parquet` means the corpus moved
    under the draw -- a source re-probed with new exclusions, or a blocked
    source's rows dropped. Measuring the remainder would publish an S tier that
    does not match its own sample.json."""
    from eda.driver import load_files
    from eda.sample import draw, sampled_rows

    cfg = _signal_corpus(tmp_path)
    files = load_files(cfg, "A")
    sample = draw(cfg, files, "A")
    shrunk = files[files["source_name"] != "src-b"]
    with pytest.raises(KeyError, match="no longer in files.parquet"):
        sampled_rows(shrunk, sample)


def test_the_signal_pass_writes_both_planes_and_joins_back_to_the_census(tmp_path):
    from eda.driver import load_files, load_signal, signal_partition
    from eda.sample import draw

    cfg = _signal_corpus(tmp_path)
    draw(cfg, load_files(cfg, "A"), "A")
    res = signal_partition(cfg, "A")
    assert res.files == 6 and res.failures == 0

    signal = load_signal(cfg, "A")
    assert len(signal) == 6
    assert signal["signal_ok"].all()
    for base in ("rms_dbfs", "silence_ratio", "effective_bandwidth_hz"):
        assert f"{base}_{NATIVE}" in signal and f"{base}_{CHAIN}" in signal
    # the join is what makes the S tier labellable at all
    assert set(signal["pool"]) == {"A"}
    assert "duration_s" in signal


def test_the_signal_pass_is_resumable_and_does_not_redo_finished_shards(tmp_path):
    from eda.driver import load_files, signal_partition
    from eda.sample import draw

    cfg = _signal_corpus(tmp_path)
    draw(cfg, load_files(cfg, "A"), "A")
    first = signal_partition(cfg, "A")
    second = signal_partition(cfg, "A")
    assert first.shards_run > 0 and first.shards_skipped == 0
    assert second.shards_run == 0 and second.shards_skipped == first.shards_run


def test_a_file_that_cannot_be_decoded_is_a_row_not_a_lost_census(tmp_path):
    """🔴 F-S2: corruption is a finding. A file that vanishes from the S tier
    makes the census wrong in the direction that hides problems."""
    from eda.driver import load_files, signal_partition
    from eda.sample import draw

    cfg = _signal_corpus(tmp_path)
    files = load_files(cfg, "A")
    sample = draw(cfg, files, "A")
    # ⚠️ A file *in the draw*. Half the corpus is not sampled, so corrupting
    # `files.iloc[0]` proves nothing -- it passed once written that way.
    victim = cfg.root / files.set_index("file_id").loc[sample.file_ids[0], "path"]
    victim.write_bytes(b"truncated")

    res = signal_partition(cfg, "A")
    assert res.files == 6
    frame = __import__("pandas").read_parquet(res.table)
    assert len(frame) == 6, "the broken file is still a row"
    assert int((~frame["signal_ok"]).sum()) >= 1
    assert frame.loc[~frame["signal_ok"], "signal_error"].notna().all()


def test_the_signal_pass_refuses_to_improvise_a_sample(tmp_path):
    """The draw is written before anything decodes. A pass that drew its own
    would measure a different set every time a source was re-probed."""
    from eda.driver import NotProbed, signal_partition

    cfg = _signal_corpus(tmp_path)
    with pytest.raises(NotProbed, match="draw the sample first"):
        signal_partition(cfg, "A")
