"""Phase 0 of the EDA pass: docs/EDA/07 section 1.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from eda.analyze import duplicates as dup
from eda.analyze import grouping as grp
from eda.analyze import shortcut as sc
from eda.config import (AnalysisConfig, ConfigError, EdaConfig, GateConfig, ProbeConfig,
                        SourceSpec, eda_config_from_dict)
from eda.driver import consolidate, enumerate_source, load_files, probe_source
from eda.extract import METADATA, Extractor, ExtractorError, Registry, run_metadata
from eda.ids import file_id_for, relpath_of, under_any
from eda.gates import FAIL, NA, PASS, _g_eda3, run_gates, worst

SR = 16000


def _write(path, seconds=0.25, sr=SR, seed=0, subtype="PCM_16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, rng.standard_normal(int(seconds * sr)) * 0.1, sr, subtype=subtype)
    return path


@pytest.fixture
def corpus(tmp_path):
    """Two pools, four sources, one deliberate cross-source duplicate."""
    root = tmp_path / "interim"
    for i in range(4):
        _write(root / "src-a/v1" / f"spk{i % 2}" / f"a{i}.wav", seed=i)
    for i in range(4):
        _write(root / "src-b/v1" / f"gen{i % 2}" / f"b{i}.wav", seed=100 + i)
    # src-c row 0 is byte-identical to src-a's -- the RIRS/MUSAN shape.
    _write(root / "src-c/v1" / "c0.wav", seed=0)
    _write(root / "src-c/v1" / "c1.wav", seed=7)
    cfg = EdaConfig(
        root=root, out=tmp_path / "out",
        sources=(
            SourceSpec(name="src-a", pool="A", root="src-a/v1", suffixes=(".wav",)),
            SourceSpec(name="src-b", pool="B", root="src-b/v1", suffixes=(".wav",)),
            SourceSpec(name="src-c", pool="A", root="src-c/v1", suffixes=(".wav",)),
        ),
        probe=ProbeConfig(workers=2, shard_size=3),
    )
    return cfg


def _probe_all(cfg):
    for s in cfg.sources:
        probe_source(cfg, s)
    for partition in cfg.partitions():
        consolidate(cfg, partition)
    return load_files(cfg)


# --------------------------------------------------------------------------- #
# the registry contract
# --------------------------------------------------------------------------- #

def test_an_extractor_that_emits_an_undeclared_column_raises():
    """The dup_group defect in miniature: a value computed and never declared
    cannot reach the table, so it cannot be silently dropped either."""
    ex = Extractor("liar", "M", ("a", "ok"), "ok", lambda p, c: {"a": 1, "ok": True, "b": 2})
    with pytest.raises(ExtractorError, match="undeclared=\\['b'\\]"):
        ex(None, None)


def test_an_extractor_that_skips_a_declared_column_raises():
    ex = Extractor("lazy", "M", ("a", "ok"), "ok", lambda p, c: {"ok": True})
    with pytest.raises(ExtractorError, match="missing=\\['a'\\]"):
        ex(None, None)


def test_registration_refuses_an_extractor_with_the_wrong_arity():
    """A signal extractor takes exactly (wav, sample_rate). The mutation -- a
    third parameter, which is how it would learn which plane it is on."""
    r = Registry("S", 2, ("wav", "sample_rate"))
    with pytest.raises(ExtractorError, match="takes 3 argument"):
        r.add("peeker", ("x", "ok"), "ok", lambda wav, sr, plane: {"x": 1, "ok": True})
    r.add("fine", ("x", "ok"), "ok", lambda wav, sr: {"x": 1, "ok": True})
    assert "fine" in r


def test_registration_refuses_a_variadic_extractor():
    r = Registry("S", 2, ("wav", "sample_rate"))
    with pytest.raises(ExtractorError, match="\\*args"):
        r.add("splat", ("x", "ok"), "ok", lambda *a: {"x": 1, "ok": True})


def test_registration_refuses_an_ok_column_outside_the_declared_set():
    r = Registry("M", 2, ("path", "probe"))
    with pytest.raises(ExtractorError, match="not among its columns"):
        r.add("bad", ("x",), "missing_ok", lambda p, c: {"x": 1})


def test_a_metadata_failure_becomes_a_row_not_an_exception(tmp_path):
    """R2 and F-S2: a file that vanishes from a census makes the census wrong in
    the direction that hides problems."""
    missing = tmp_path / "nope.wav"
    row = run_metadata(missing, ProbeConfig())
    assert row["probe_ok"] is False and row["identity_ok"] is False
    assert set(row) >= {"container", "orig_sr", "sha256", "file_bytes"}
    assert row["sha256"] is None


def test_extractor_column_collisions_are_an_error(tmp_path):
    f = _write(tmp_path / "x.wav")
    METADATA.add("clash", ("container", "clash_ok"), "clash_ok",
                 lambda p, c: {"container": "?", "clash_ok": True})
    try:
        with pytest.raises(ExtractorError, match="already written"):
            run_metadata(f, ProbeConfig())
    finally:
        METADATA.pop("clash")


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

def test_ffprobe_recovers_the_native_rate_and_channels(tmp_path):
    """`orig_sr` is load-bearing: the native plane is
    `load_audio(path, sample_rate=orig_sr)`, which short-circuits the resampler,
    so a wrong value silently turns the native plane into a resampled one."""
    f = _write(tmp_path / "x.wav", seconds=0.5, sr=22050)
    row = run_metadata(f, ProbeConfig())
    assert row["probe_ok"] is True
    assert row["orig_sr"] == 22050 and row["orig_channels"] == 1
    assert row["duration_s"] == pytest.approx(0.5, abs=0.02)
    assert row["file_bytes"] == f.stat().st_size


def test_file_id_is_derived_from_the_path_not_from_an_index(corpus):
    """A re-run that finds one more file must not renumber the corpus."""
    s = corpus.source("src-a")
    ids = {file_id_for(s.name, p.relative_to(corpus.root / s.root))
           for p in enumerate_source(corpus, s)}
    assert "src-a:spk0/a0.wav" in ids
    _write(corpus.root / "src-a/v1/spk0/a000.wav", seed=99)
    again = {file_id_for(s.name, p.relative_to(corpus.root / s.root))
             for p in enumerate_source(corpus, s)}
    assert ids < again


def test_probe_is_resumable_and_the_second_run_does_no_work(corpus):
    first = probe_source(corpus, corpus.source("src-a"))
    assert first.shards_run == 2 and first.shards_skipped == 0
    second = probe_source(corpus, corpus.source("src-a"))
    assert second.shards_run == 0 and second.shards_skipped == 2


def test_consolidate_refuses_a_part_whose_done_marker_is_missing(corpus):
    probe_source(corpus, corpus.source("src-a"))
    marker = next((corpus.out / "A" / "parts" / "src-a").glob("*.DONE"))
    marker.unlink()
    with pytest.raises(RuntimeError, match="no DONE marker"):
        consolidate(corpus, "A")


def test_enumeration_honours_an_exclusion(corpus, tmp_path):
    """G-EDA1's mechanism. The mutation: drop the exclusion and the row returns."""
    _write(corpus.root / "src-a/v1/real_half/r0.wav", seed=5)
    s = corpus.source("src-a")
    assert any("real_half" in str(p) for p in enumerate_source(corpus, s))
    guarded = SourceSpec(name="src-a", pool="A", root="src-a/v1", suffixes=(".wav",),
                         exclude=("real_half",), exclude_reason="YouTube-derived")
    assert not any("real_half" in str(p) for p in enumerate_source(corpus, guarded))


def test_an_exclusion_without_a_reason_is_refused():
    with pytest.raises(ConfigError, match="exclude_reason"):
        SourceSpec(name="x", pool="D", root="x", exclude=("real",))


#: The smallest config that is not itself an error, so a test of some *other*
#: rejection is not satisfied by the empty-sources one.
_MINIMAL = {"sources": [{"name": "s", "pool": "A", "root": "s"}]}


def test_unknown_config_keys_are_an_error():
    with pytest.raises(ConfigError, match="unknown key"):
        eda_config_from_dict({**_MINIMAL, "wrkers": 4})
    with pytest.raises(ConfigError, match="unknown key"):
        eda_config_from_dict({**_MINIMAL, "probe": {"wrkers": 4}})
    with pytest.raises(ConfigError, match=r"sources\[0\]: unknown key"):
        eda_config_from_dict({"sources": [{"name": "s", "pool": "A", "root": "s",
                                           "sufixes": [".wav"]}]})


def test_a_config_with_no_sources_is_refused():
    """It loads cleanly and then does nothing: `probe` selects no source,
    `consolidate` finds no parts, and the run reports success. Same shape as a
    knob that validates and silently does nothing."""
    with pytest.raises(ConfigError, match="`sources` is empty"):
        eda_config_from_dict({})
    with pytest.raises(ConfigError, match="`sources` is empty"):
        eda_config_from_dict({"root": ".", "sources": []})
    assert len(eda_config_from_dict(_MINIMAL).sources) == 1


# --------------------------------------------------------------------------- #
# the analyses
# --------------------------------------------------------------------------- #

def test_the_duplicate_sweep_finds_cross_source_byte_identity(corpus):
    """The RIRS/MUSAN shape: same bytes, different source_name, so
    `grouping_atoms`' union-find cannot link them."""
    files = _probe_all(corpus)
    report = dup.cross_source_report(files)
    assert report["cross_source_groups"] == 1
    assert tuple(report["pairs"]["sources"].iloc[0]) == ("src-a", "src-c")
    groups = dup.assign_dup_groups(files)
    linked = files.loc[groups.notna(), "source_name"]
    assert set(linked) == {"src-a", "src-c"} and len(linked) == 2


def test_a_unique_corpus_yields_no_duplicate_groups(corpus):
    files = _probe_all(corpus)
    files = files[files["file_id"] != "src-c:c0.wav"]
    assert dup.cross_source_report(files)["cross_source_groups"] == 0
    assert dup.assign_dup_groups(files).isna().all()


def test_head_labels_are_null_where_the_metric_ignores_them():
    """A pool-A row has no `music_fake`; `POOL_LABELS` says so with None, and
    the masked EER ignores those rows too -- so the audit must drop them rather
    than coerce them to 0."""
    files = pd.DataFrame({"pool": ["A", "B", "C", "D", "E"]})

    def labelled(head):
        """`(pool -> label)` for the rows the head actually scores."""
        s = sc.head_labels(files, head)
        return {files["pool"][i]: int(s[i]) for i in s.index[s.notna()]}

    # Critical: null is not 0. A pool-C row carries no voice_fake at all, and a
    # test asserting `== 0` would pass while the audit trained on 400 invented
    # negatives -- which is the shape of the masked-EER pools in metrics/.
    assert labelled("voice_fake") == {"A": 0, "B": 1}
    assert labelled("music_fake") == {"C": 0, "D": 1}
    assert labelled("voice_present") == {"A": 1, "B": 1, "C": 0, "D": 0, "E": 0}
    assert labelled("music_present") == {"A": 0, "B": 0, "C": 1, "D": 1, "E": 0}
    assert sc.head_labels(files, "voice_fake").notna().tolist() == [True, True, False, False, False]


def test_missingness_is_a_feature(corpus):
    """`bit_rate` is null for every wav and present for every mp3. An imputation
    that hid that would hide the confound rather than measure it."""
    files = pd.DataFrame({"orig_sr": [16000, None], "container": ["wav", "mp3"]})
    x, names = sc.build_design(files)
    assert "orig_sr_isna" in names
    assert x[:, names.index("orig_sr_isna")].tolist() == [0.0, 1.0]


def test_the_shortcut_audit_detects_a_planted_confound():
    """The gate has to be able to fire. Container is made to predict the label
    perfectly; the mutation is the line below it, where it does not."""
    n = 200
    planted = pd.DataFrame({
        "pool": ["A"] * n + ["B"] * n,
        "source_name": (["a1", "a2"] * n)[:n] + (["b1", "b2"] * n)[:n],
        "container": ["wav"] * n + ["mp3"] * n,
        "orig_sr": [16000] * n + [22050] * n,
    })
    audit = sc.shortcut_audit(planted).set_index("head")
    assert audit.loc["voice_fake", "auc"] > 0.99
    assert "container=" in audit.loc["voice_fake", "top_features"]

    rng = np.random.default_rng(0)
    clean = planted.copy()
    clean["container"] = rng.choice(["wav", "mp3"], size=2 * n)
    clean["orig_sr"] = rng.choice([16000, 22050], size=2 * n)
    assert sc.shortcut_audit(clean).set_index("head").loc["voice_fake", "auc"] < 0.60


def test_a_head_with_one_class_is_not_scored(corpus):
    """It reports nan, and `eda.gates` turns that into `na`. A head that could
    not be audited must never read as audited."""
    files = pd.DataFrame({"pool": ["A"] * 8, "source_name": ["a"] * 8,
                          "container": ["wav"] * 8})
    row = sc.shortcut_audit(files).set_index("head").loc["voice_fake"]
    assert np.isnan(row["auc"]) and "two classes" in row["note"]


def test_the_depth_profile_locates_the_grouping_directory(corpus):
    files = _probe_all(corpus)
    prof = grp.depth_profile(files)
    a0 = prof[(prof["source_name"] == "src-a") & (prof["depth"] == 0)].iloc[0]
    assert a0["distinct_prefixes"] == 2            # spk0, spk1
    report = grp.grouping_report(
        files, gates=GateConfig(min_groups_per_role=2)).set_index("source_name")
    assert report.loc["src-a", "candidate_groups"] == 2
    assert report.loc["src-a", "meets_floor"]
    # src-c is flat: no depth carries a group, so the floor is not met.
    assert not report.loc["src-c", "meets_floor"]


# --------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------- #

def test_fail_beats_na_beats_pass():
    """`training.validate`'s `quotable` ordering, restated. The mutation is any
    reordering of `_ORDER`."""
    assert worst([PASS, NA]) == NA
    assert worst([PASS, NA, FAIL]) == FAIL
    assert worst([PASS, PASS]) == PASS
    assert worst([]) == NA


def test_a_gate_whose_input_is_missing_reads_na_not_pass(corpus):
    """VG1 A10 reported SKIP on every run because its input was null, and it
    read as fine. This is that lesson as a test."""
    files = _probe_all(corpus)
    gates = run_gates(corpus, files).set_index("gate")
    assert gates.loc["G-EDA2", "verdict"] == NA
    assert gates.loc["G-EDA3", "verdict"] == NA
    assert gates.loc["G-EDA5", "verdict"] == NA
    assert gates.attrs["aggregate"] == NA


def test_the_duplicate_gate_fails_on_the_planted_duplicate(corpus):
    files = _probe_all(corpus)
    gates = run_gates(corpus, files,
                      dupes=dup.cross_source_report(files)).set_index("gate")
    assert gates.loc["G-EDA5", "verdict"] == FAIL
    assert "1 cross-source duplicate group" in gates.loc["G-EDA5", "detail"]
    assert gates.attrs["aggregate"] == FAIL


def test_pool_d_without_an_exclusion_decision_reads_na(tmp_path):
    """docs/EDA/04 D1 is a licence gate, and 'we have not looked' is not a pass."""
    root = tmp_path / "i"
    _write(root / "d/v1/x.wav")
    cfg = EdaConfig(root=root, out=tmp_path / "o",
                    sources=(SourceSpec(name="d", pool="D", root="d/v1",
                                        suffixes=(".wav",)),),
                    probe=ProbeConfig(workers=1, shard_size=4))
    files = _probe_all(cfg)
    gates = run_gates(cfg, files).set_index("gate")
    assert gates.loc["G-EDA1/exclude/d", "verdict"] == NA
    assert "MusicCaps" in gates.loc["G-EDA1/exclude/d", "detail"]


def test_an_absent_allowlist_is_na_and_a_violated_one_fails(tmp_path):
    root = tmp_path / "i"
    for name in ("000002.mp3", "000009.mp3"):
        _write(root / "c/v1" / name.replace(".mp3", ".wav"))
    src = SourceSpec(name="c", pool="C", root="c/v1", suffixes=(".wav",),
                     allowlist="_lic/allow.csv", allowlist_key="int_stem")
    cfg = EdaConfig(root=root, out=tmp_path / "o", sources=(src,),
                    probe=ProbeConfig(workers=1, shard_size=4))
    files = _probe_all(cfg)
    g = run_gates(cfg, files).set_index("gate")
    assert g.loc["G-EDA1/allowlist/c", "verdict"] == NA

    lic = root / "_lic" / "allow.csv"
    lic.parent.mkdir(parents=True, exist_ok=True)
    lic.write_text("id\n2\n", encoding="utf-8")
    g = run_gates(cfg, files).set_index("gate")
    assert g.loc["G-EDA1/allowlist/c", "verdict"] == FAIL
    assert "1 of 2 rows" in g.loc["G-EDA1/allowlist/c", "detail"]

    lic.write_text("id\n2\n9\n", encoding="utf-8")
    g = run_gates(cfg, files).set_index("gate")
    assert g.loc["G-EDA1/allowlist/c", "verdict"] == PASS


# --------------------------------------------------------------------------- #
# regressions -- one per defect found in the 2026-09-11 review pass
# --------------------------------------------------------------------------- #

def test_exclusion_matches_path_components_not_substrings():
    """The first version was `rel.startswith(x) or f"/{x}" in f"/{rel}"`, which
    reads correctly and silently excluded `real_half/` and `a/really/` for
    `exclude: [real]`. It existed in two places and was wrong in both."""
    assert under_any("real/x.wav", ["real"])
    assert under_any("a/real/x.wav", ["real"])
    assert not under_any("real_half/x.wav", ["real"])
    assert not under_any("a/really/x.wav", ["real"])
    assert not under_any("unreal/x.wav", ["real"])
    # A multi-component prefix, which is the shape CompSpoof's env_sources needs.
    assert under_any("env_sources/EnvSDD/b/x.wav", ["env_sources/EnvSDD"])
    assert not under_any("env_sources/Other/x.wav", ["env_sources/EnvSDD"])


def test_the_driver_and_the_gate_agree_about_an_exclusion(tmp_path):
    """They had two copies of the matcher. Now `SourceSpec` owns it, so this
    asserts they cannot diverge rather than that they currently agree."""
    root = tmp_path / "i"
    _write(root / "s/v1/real_half/a.wav")
    _write(root / "s/v1/gen/b.wav")
    src = SourceSpec(name="s", pool="D", root="s/v1", suffixes=(".wav",),
                     exclude=("real_half",), exclude_reason="YouTube-derived")
    cfg = EdaConfig(root=root, out=tmp_path / "o", sources=(src,),
                    probe=ProbeConfig(workers=1, shard_size=4))
    files = _probe_all(cfg)
    assert len(files) == 1 and files["file_id"].iloc[0] == "s:gen/b.wav"
    g = run_gates(cfg, files).set_index("gate")
    assert g.loc["G-EDA1/exclude/s", "verdict"] == PASS


def test_file_id_refuses_a_non_string_source():
    """An f-string interpolates a SourceFile spec happily, producing an id built
    from a dataclass repr that then joins to nothing."""
    with pytest.raises(TypeError, match="Pass `source.name`"):
        file_id_for(SourceSpec(name="s", pool="A", root="s"), "a.wav")
    with pytest.raises(ValueError, match="contains"):
        file_id_for("bad:name", "a.wav")
    assert relpath_of(file_id_for("s", "a/b.wav")) == "a/b.wav"


def test_a_clean_duplicate_sweep_reads_pass_not_na(corpus):
    """The detail table was the only artifact, so a corpus with no duplicates
    wrote no parquet and `eda gates` then reported `na` for a pass -- the exact
    confusion the tri-state exists to prevent."""
    files = _probe_all(corpus)
    clean = files[files["file_id"] != "src-c:c0.wav"]
    report = dup.cross_source_report(clean)
    assert report["detail"].empty
    summary = dup.summary_of(report)
    assert summary == {"groups": 0, "rows": 0, "cross_source_groups": 0,
                       "cross_pool_groups": 0}
    g = run_gates(corpus, clean, dupes=summary).set_index("gate")
    assert g.loc["G-EDA5", "verdict"] == PASS
    # And absence still means absence.
    assert run_gates(corpus, clean).set_index("gate").loc["G-EDA5", "verdict"] == NA


def test_consolidate_refuses_a_part_missing_the_fixed_columns(corpus):
    """A part written by an older revision, or by a probe run with a narrowed
    `extractors` list, would otherwise surface as a KeyError three modules away."""
    probe_source(corpus, corpus.source("src-a"))
    part = next((corpus.out / "A" / "parts" / "src-a").glob("*.parquet"))
    df = pd.read_parquet(part).drop(columns=["pool"])
    df.to_parquet(part, index=False)
    with pytest.raises(RuntimeError, match=r"missing \['pool'\]"):
        consolidate(corpus, "A")


def test_heads_follow_the_manifest_label_order():
    """They were written out as literal indices into a POOL_LABELS tuple, so
    reordering `manifest._LABEL_COLUMNS` would have silently swapped two heads
    -- the shape of the File/Voice swap that once passed 749 of 749 tests."""
    from training.manifest import POOL_LABELS, _LABEL_COLUMNS
    assert sc.HEADS == {c.removeprefix("label_"): i
                        for i, c in enumerate(_LABEL_COLUMNS)}
    for pool, labels in POOL_LABELS.items():
        for head, i in sc.HEADS.items():
            assert labels[i] == POOL_LABELS[pool][i]
    assert POOL_LABELS["D"][sc.HEADS["music_fake"]] == 1
    assert POOL_LABELS["B"][sc.HEADS["voice_fake"]] == 1


def test_valid_pools_follow_the_manifest():
    from eda.config import MIXED, VALID_POOLS
    from training.manifest import POOLS
    assert VALID_POOLS == (*POOLS, MIXED)
    with pytest.raises(ConfigError, match="pool must be one of"):
        SourceSpec(name="x", pool="Z", root="x")


def test_an_incomplete_duplicate_summary_reads_na_not_pass():
    """A summary written by an older revision is missing keys. Reading it as a
    pass would report a check that never ran; indexing it blind would raise a
    KeyError from inside a gate."""
    cfg = EdaConfig(root=Path("."), out=Path("."),
                    sources=(SourceSpec(name="s", pool="A", root="s"),))
    files = pd.DataFrame({"file_id": ["s:a.wav"], "path": ["s/a.wav"],
                          "source_name": ["s"], "pool": ["A"]})
    g = run_gates(cfg, files, dupes={"groups": 0}).set_index("gate")
    assert g.loc["G-EDA5", "verdict"] == NA
    assert "missing" in g.loc["G-EDA5", "detail"]


def test_analysis_knobs_reach_the_analyses():
    """`n_splits` and `top_k` were hardcoded inside the audit, and the seed it
    used was `SampleConfig.seed` -- the knob that picks which files are measured,
    which is part of the corpus definition rather than a CV shuffle."""
    n = 60
    files = pd.DataFrame({
        "pool": ["A"] * n + ["B"] * n,
        "source_name": ["a"] * n + ["b"] * n,
        "file_id": [f"a:{i}.wav" for i in range(n)] + [f"b:{i}.wav" for i in range(n)],
        "container": [f"c{i % 40}" for i in range(2 * n)],
    })
    wide = sc.build_design(files, top_k=40)[1]
    narrow = sc.build_design(files, top_k=3)[1]
    assert len(wide) > len(narrow)
    assert sum(n.startswith("container=") for n in narrow) == 4   # 3 + __other__

    # 🔴 And the knob must reach `shortcut_audit`, not just `build_design`. A
    # container shared by several rows is separable only when its one-hot
    # survives the top-k fold, so the audit's own AUC moves with the knob --
    # which is what makes a hardcoded 20 inside the audit detectable.
    shared = pd.DataFrame({
        "pool": ["A"] * n + ["B"] * n,
        "source_name": ["a"] * n + ["b"] * n,
        "container": [f"c{i % 10}" for i in range(n)]
                     + [f"c{10 + i % 10}" for i in range(n)],
    })
    auc = {k: sc.shortcut_audit(shared, AnalysisConfig(top_k=k))
              .set_index("head").loc["voice_fake", "auc"] for k in (1, 20)}
    assert auc[20] > 0.99 and auc[1] < 0.70, auc


def test_the_cli_records_a_clean_sweep(corpus, capsys):
    """`analyze` must leave evidence that the duplicate sweep ran even when it
    finds nothing, or a later `eda gates` reports `na` for a pass.

    ⚠️ Caveat, and it took two attempts to get right: `src-c` is a flat
    directory, so no depth carries a grouping atom and G-EDA3 fires at any
    floor -- both verbs exit 1, correctly. Asserting the exit code here would
    test the grouping floor rather than the thing this test is about, and the
    first two versions of this test failed for that reason rather than for the
    one named above. It asserts on the artifact and on G-EDA5's own row.
    """
    import json

    from eda import cli

    (corpus.root / "src-c" / "v1" / "c0.wav").unlink()    # the planted twin
    cfg = dataclasses.replace(corpus, gates=GateConfig(min_groups_per_role=2))
    _probe_all(cfg)

    cli.cmd_analyze(cfg, None)
    shared = cfg.out / "_shared"
    summary = shared / cli.DUPLICATE_SUMMARY
    assert summary.exists(), "a clean sweep left no evidence that it ran"
    assert not (shared / cli.DUPLICATE_TABLE).exists()
    assert json.loads(summary.read_text())["cross_source_groups"] == 0

    capsys.readouterr()
    cli.cmd_gates(cfg, None)
    line = next(l for l in capsys.readouterr().out.splitlines() if "G-EDA5" in l)
    assert "pass" in line, line


def test_the_cross_validation_is_clamped_to_the_minority_class_and_says_so():
    """Requesting 5 folds from a 3-member class makes sklearn warn and the last
    folds degenerate. A narrowed CV is a different measurement, so it is
    reported rather than substituted silently."""
    files = pd.DataFrame({
        "pool": ["A"] * 30 + ["B"] * 3,
        "source_name": ["a"] * 30 + ["b"] * 3,
        "container": ["wav"] * 33,
    })
    row = sc.shortcut_audit(files, AnalysisConfig(n_splits=5)).set_index("head").loc["voice_fake"]
    assert row["n_splits"] == 3
    assert "cv narrowed to 3 folds" in row["note"]

    balanced = pd.DataFrame({
        "pool": ["A"] * 30 + ["B"] * 30,
        "source_name": ["a"] * 30 + ["b"] * 30,
        "container": ["wav"] * 60,
    })
    ok = sc.shortcut_audit(balanced, AnalysisConfig(n_splits=5)).set_index("head").loc["voice_fake"]
    assert ok["n_splits"] == 5 and "narrowed" not in ok["note"]


def test_a_probe_without_hashes_leaves_the_duplicate_gate_na(tmp_path):
    """`ProbeConfig.extractors` can narrow the M tier for a fast first look, and
    such a run writes no `sha256`. A corpus with no duplicates and a corpus
    nobody hashed are different findings, so the sweep refuses rather than
    returning an empty result that would read as a pass."""
    root = tmp_path / "i"
    _write(root / "s/v1/a.wav")
    cfg = EdaConfig(root=root, out=tmp_path / "o",
                    sources=(SourceSpec(name="s", pool="A", root="s/v1",
                                        suffixes=(".wav",)),),
                    probe=ProbeConfig(workers=1, shard_size=4,
                                      extractors=("ffprobe",)))
    files = _probe_all(cfg)
    assert "sha256" not in files.columns and "orig_sr" in files.columns
    with pytest.raises(dup.NoHashes, match="excluded `identity`"):
        dup.cross_source_report(files)
    # and the gate says `na`, not `pass`
    assert run_gates(cfg, files).set_index("gate").loc["G-EDA5", "verdict"] == NA


def test_a_partition_with_no_table_is_reported_not_skipped(corpus):
    """An audit over half the corpus produces real-looking numbers for the heads
    it can still score, and nothing on the page says so."""
    for s in corpus.sources:
        probe_source(corpus, s)
    consolidate(corpus, "A")       # partition B deliberately left unconsolidated
    files = load_files(corpus)
    assert files.attrs["missing_partitions"] == ["B"]
    assert set(files["pool"]) == {"A"}
    # voice_fake now has one class, so the shortcut gate cannot pass either.
    audit = sc.shortcut_audit(files, corpus.analysis)
    assert run_gates(corpus, files, audit=audit).set_index("gate").loc[
        "G-EDA2", "verdict"] == NA


def test_importing_the_driver_populates_the_signal_and_vector_registries():
    """🔴 The same guard as below, for the two tiers that decode -- and it is
    here because the defect happened. `eda.extract.vectors` was written, tested
    and wired into `_signal_row`, and its registration import was never added
    to this module: `run_vectors` returned `{}`, every part wrote a
    `vectors.npz` holding nothing but `file_id`, and the pass reported success.

    ⚠️ Run in a clean interpreter, and deliberately **not** importing
    `eda.extract.vectors` here. The end-to-end test that should have caught it
    imported the module itself, which registered the extractor as a side effect
    and made the missing line invisible.
    """
    import subprocess
    import sys as _sys

    probe = (
        "import eda.driver as d; from eda.extract import SIGNAL, VECTOR;"
        "assert sorted(SIGNAL) == ['level', 'spectral', 'timing'], sorted(SIGNAL);"
        "assert sorted(VECTOR) == ['vector'], sorted(VECTOR);"
        "assert VECTOR.widths == {'vector': 128}, VECTOR.widths;"
        "print('ok')"
    )
    r = subprocess.run([_sys.executable, "-c", probe], capture_output=True,
                       text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr


def test_importing_the_driver_populates_the_metadata_registry():
    """`eda.driver` imports the extractor modules for their registration side
    effect, and a tidy-up that removes those lines leaves the registry empty --
    `probe` would then write a table of ids and nothing else, which looks like a
    successful run. A comment is not a guard; this is.

    Run in a clean interpreter because import caching would otherwise let this
    pass on whatever some earlier test happened to import.
    """
    import subprocess
    import sys as _sys

    probe = (
        "import eda.driver as d; from eda.extract import METADATA;"
        "names = sorted(METADATA);"
        "print(names);"
        "assert names == ['ffprobe', 'identity', 'mp3_header'], names"
    )
    r = subprocess.run([_sys.executable, "-c", probe], capture_output=True,
                       text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr
    assert "ffprobe" in r.stdout and "identity" in r.stdout


# --------------------------------------------------------------------------- #
# whole_file sources -- the SONICS correction
# --------------------------------------------------------------------------- #

def test_a_component_source_and_a_whole_file_source_carry_opposite_keys():
    """`validate_manifest`'s invariant, enforced at config load so the failure
    lands before the EDA tables are written rather than at manifest-build time,
    after a fold table may already rest on them."""
    comp = SourceSpec(name="c", pool="A", root="c")
    assert comp.row_kind == "component" and comp.cell is None and comp.partition == "A"

    whole = SourceSpec(name="w", row_kind="whole_file", cell=8, root="w")
    assert whole.pool is None and whole.partition == "cell8"

    with pytest.raises(ConfigError, match="must have cell = null"):
        SourceSpec(name="x", pool="A", root="x", cell=8)
    with pytest.raises(ConfigError, match="must have pool = null"):
        SourceSpec(name="x", row_kind="whole_file", cell=8, pool="D", root="x")
    with pytest.raises(ConfigError, match="cell must be one of"):
        SourceSpec(name="x", row_kind="whole_file", root="x")
    with pytest.raises(ConfigError, match="row_kind must be one of"):
        SourceSpec(name="x", row_kind="componant", pool="A", root="x")


def test_cells_6_and_7_cannot_be_whole_file_sources():
    """They hold one real and one fake component, which is the whole reason they
    cannot be scraped -- and the reason the competition has two fake heads."""
    for cell in (6, 7):
        with pytest.raises(ConfigError, match="cannot be a whole_file source"):
            SourceSpec(name="x", row_kind="whole_file", cell=cell, root="x")
    for cell in (1, 5, 8, 9):
        SourceSpec(name="x", row_kind="whole_file", cell=cell, root="x")


def test_head_labels_read_the_cell_table_for_whole_file_rows():
    """🔴 The SONICS defect as a test. It was acquired as pool D; pool D asserts
    `voice_present = 0`, and its own fake_songs.csv reports `no_vocal = False`
    for all 49,074 rows. Cell 8 is what an AI song actually is."""
    files = pd.DataFrame({
        "row_kind": ["component"] * 2 + ["whole_file"] * 3,
        "pool": ["A", "D", None, None, None],
        "cell": [None, None, 8, 5, 9],
    })
    got = {h: sc.head_labels(files, h).tolist() for h in sc.HEADS}

    # A cell-8 row (index 2) is positive on every head; a pool-D row (index 1)
    # is the thing SONICS would wrongly have been.
    assert got["voice_present"][2] == 1 and got["music_present"][2] == 1
    assert got["voice_fake"][2] == 1 and got["music_fake"][2] == 1
    assert got["voice_present"][1] == 0, "pool D asserts no voice -- the defect"
    assert pd.isna(got["voice_fake"][1])

    assert got["voice_fake"][3] == 0 and got["music_fake"][3] == 0      # cell 5
    assert got["voice_present"][4] == 0 and pd.isna(got["voice_fake"][4])   # cell 9


def test_a_whole_file_source_probes_into_its_own_partition(tmp_path):
    """It has no pool to partition by, so it writes to `cell<N>/` -- and both
    label columns are written either way, so the table's schema does not depend
    on which sources happen to be in it."""
    root = tmp_path / "i"
    _write(root / "songs/v1/a.mp3".replace(".mp3", ".wav"))
    cfg = EdaConfig(
        root=root, out=tmp_path / "o",
        sources=(SourceSpec(name="songs", row_kind="whole_file", cell=8,
                            root="songs/v1", suffixes=(".wav",)),),
        probe=ProbeConfig(workers=1, shard_size=4))
    assert cfg.partitions() == ["cell8"]
    files = _probe_all(cfg)
    assert (cfg.out / "cell8" / "files.parquet").exists()
    row = files.iloc[0]
    assert row["row_kind"] == "whole_file" and row["cell"] == 8
    assert pd.isna(row["pool"])
    assert sc.head_labels(files, "music_fake").tolist() == [1]


def test_the_shipped_config_registers_sonics_as_cell_8_not_pool_d():
    """A pin on the correction itself: SONICS' acquisition record says
    `pool: D`, and the EDA config must not repeat it."""
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    sonics = cfg.source("sonics")
    assert sonics.row_kind == "whole_file" and sonics.cell == 8
    assert sonics.pool is None, "pool D asserts voice_present=0 on 49k songs with vocals"

    # CtrSVDD is fake *sung voice*; the rules classify vocals as voice.
    ctr = cfg.source("ctrsvdd")
    assert ctr.row_kind == "component" and ctr.pool == "B"

    # And pool D is still only FakeMusicCaps -- SONICS does not fill it.
    assert [s.name for s in cfg.sources_in("D")] == ["fakemusiccaps"]


def test_the_allowlist_keys_match_the_shipped_csv_shapes(tmp_path):
    """🔴 Both shipped allowlists name the column `id`, not `track_id`, and
    Jamendo's ids are **paths** (`14/214.mp3`) while FMA's are bare integers
    (`2`). Matching Jamendo on the stem compares `214` against `14/214.mp3`, so
    every row fails and G-EDA1 reports FAIL for a reason that is not a licence.
    """
    lic = tmp_path / "_lic"
    lic.mkdir()
    (lic / "fma.csv").write_text("id,licence,verdict\n2,x,allow\n", encoding="utf-8")
    (lic / "jam.csv").write_text("id,licence,verdict\n14/214.mp3,x,allow\n",
                                 encoding="utf-8")

    def verdict(name, key, rels):
        files = pd.DataFrame({
            "file_id": [f"{name}:{r}" for r in rels],
            "path": [f"{name}/v1/{r}" for r in rels],
            "source_name": [name] * len(rels), "pool": ["C"] * len(rels),
            "cell": [None] * len(rels), "row_kind": ["component"] * len(rels)})
        src = SourceSpec(name=name, pool="C", root=f"{name}/v1",
                         allowlist=f"_lic/{name}.csv", allowlist_key=key)
        cfg = EdaConfig(root=tmp_path, out=tmp_path / "o", sources=(src,))
        return run_gates(cfg, files).set_index("gate").loc[
            f"G-EDA1/allowlist/{name}"]

    # zero-padded filename vs an integer id
    assert verdict("fma", "int_stem", ["000002.mp3"])["verdict"] == PASS
    assert verdict("fma", "int_stem", ["000009.mp3"])["verdict"] == FAIL
    # a path-shaped id
    assert verdict("jam", "relpath", ["14/214.mp3"])["verdict"] == PASS
    assert verdict("jam", "relpath", ["15/215.mp3"])["verdict"] == FAIL
    # the defect: stem-matching a path-shaped id rejects a permitted track
    assert verdict("jam", "stem", ["14/214.mp3"])["verdict"] == FAIL


def test_a_misnamed_allowlist_column_is_na_not_fail(tmp_path):
    """`track_id` was the configured column and `id` is the real one. A
    mis-named column rejects every row, and reporting that as FAIL would name a
    licence violation that does not exist."""
    lic = tmp_path / "_lic"
    lic.mkdir()
    (lic / "c.csv").write_text("id,verdict\n2,allow\n", encoding="utf-8")
    files = pd.DataFrame({"file_id": ["c:000002.mp3"], "path": ["c/v1/000002.mp3"],
                          "source_name": ["c"], "pool": ["C"], "cell": [None],
                          "row_kind": ["component"]})
    src = SourceSpec(name="c", pool="C", root="c/v1", allowlist="_lic/c.csv",
                     allowlist_column="track_id", allowlist_key="int_stem")
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "o", sources=(src,))
    row = run_gates(cfg, files).set_index("gate").loc["G-EDA1/allowlist/c"]
    assert row["verdict"] == NA and "has no column" in row["detail"]


def test_the_shipped_config_names_the_real_allowlist_columns():
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    assert cfg.source("fma").allowlist_column == "id"
    assert cfg.source("fma").allowlist_key == "int_stem"
    assert cfg.source("mtg-jamendo").allowlist_column == "id"
    assert cfg.source("mtg-jamendo").allowlist_key == "relpath"


def test_a_blocked_source_raises_rather_than_enumerating(tmp_path):
    """🔴 CtrSVDD is the case. 14.6% of it is `bonafide` -- real sung voice --
    and the label lives only in `train.txt`, keyed by utterance id, while
    `train_set.zip` is flat. `exclude` is path-based and cannot express it, so
    registering it as plain pool B would label 18,716 real singers
    `voice_fake = 1`.

    `blocked` exists so such a source fails loudly instead of being quietly
    deleted from the config and rediscovered later. The mutation is an empty
    `blocked` string, which returns the files."""
    root = tmp_path / "i"
    _write(root / "s/v1/a.wav")
    src = SourceSpec(name="s", pool="B", root="s/v1", suffixes=(".wav",),
                     blocked="14.6% of it is bonafide")
    cfg = EdaConfig(root=root, out=tmp_path / "o", sources=(src,),
                    probe=ProbeConfig(workers=1, shard_size=4))
    with pytest.raises(RuntimeError, match="blocked: 14.6% of it is bonafide"):
        enumerate_source(cfg, src)
    with pytest.raises(RuntimeError, match="is blocked"):
        probe_source(cfg, src)

    # and an unblocked twin of the same source enumerates normally
    ok = SourceSpec(name="s", pool="B", root="s/v1", suffixes=(".wav",))
    assert len(enumerate_source(cfg, ok)) == 1


def test_the_shipped_config_blocks_ctrsvdd_with_a_reason():
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    assert "bonafide" in cfg.source("ctrsvdd").blocked
    # Everything wave 1 needs is runnable.
    for name in ("fakemusiccaps", "fma", "mlaad", "ljspeech", "zeroth-korean",
                 "musan-music", "musan-noise", "sonics"):
        assert not cfg.source(name).blocked, name


def test_the_shipped_config_keeps_generated_environmental_audio_out_of_pool_e():
    """🔴 CompSpoof's `env_sources/<corpus>/spoofed/` is *generated*
    environmental sound -- 12,273 of it AudioLDM. It is not pool E (which is
    real) and not pool D (which is fake **music**), and it has no cell of its
    own. The source is rooted at the whole ESDD2 tree because that is where the
    bonafide halves live, so the exclusion is the only thing standing between
    12,273 AudioLDM clips and the label `real noise`."""
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    src = cfg.source("compspoof-env-bonafide")
    assert src.pool == "E" and src.exclude_reason
    for tree in ("spoofed", "speech_sources", "mixed_audio", "original_audio"):
        assert tree in src.exclude, tree

    # and it excludes on the real path shape, at the depth it really occurs
    assert src.excludes_path(
        "eval_source/env_sources/EnvSDD/spoofed/TTA/audioldm1/x.wav")
    assert not src.excludes_path(
        "eval_source/env_sources/EnvSDD/bonafide/x.wav")
    # substring, not component, would have matched this and dropped it
    assert not src.excludes_path(
        "eval_source/env_sources/EnvSDD/bonafide/spoofed_like_name.wav")


def test_a_source_can_claim_half_a_directory_by_filename(tmp_path):
    """🔴 `exclude` matches path components, and OpenSLR-28's
    `real_rirs_isotropic_noises/` holds 325 impulse responses beside 92 noise
    recordings with **no subtree** between them -- only the filenames differ.
    Treating the directory as one kind is what E0 got wrong."""
    root = tmp_path / "i"
    for name in ("RVB2014_type1_noise_largeroom1_1.wav",
                 "RWCP_type4_rir_circle_ane_imp001.wav",
                 "air_type1_air_binaural_stairway_1_1_1.wav"):
        _write(root / "r/v1" / name)
    cfg = EdaConfig(root=root, out=tmp_path / "o", sources=())

    noise = SourceSpec(name="n", pool="E", root="r/v1", suffixes=(".wav",),
                       name_glob=("*_noise_*",), name_glob_reason="the noise half")
    rirs = SourceSpec(name="r", pool="E", root="r/v1", suffixes=(".wav",),
                      name_glob=("*_rir_*", "air_*"), name_glob_reason="the kernels")
    assert len(enumerate_source(cfg, noise)) == 1
    assert len(enumerate_source(cfg, rirs)) == 2

    # and a source with no glob is still the whole directory
    whole = SourceSpec(name="w", pool="E", root="r/v1", suffixes=(".wav",))
    assert len(enumerate_source(cfg, whole)) == 3


def test_a_filename_glob_must_say_why():
    """The same rule `exclude_reason` carries: selecting part of a directory
    says this source is not the whole directory, and the reason travels with
    it."""
    with pytest.raises(ConfigError, match="name_glob_reason"):
        SourceSpec(name="x", pool="E", root="r", name_glob=("*_noise_*",))


def test_the_shipped_config_splits_the_rirs_directory_by_filename():
    """🔴 E0 blocked `real_rirs_isotropic_noises` whole, on a 1.365 s median and
    314 sub-4s files. Both statistics belong to the impulse-response half: the
    92 noise recordings have a 30.0 s median and **none** below the floor. The
    split is what lets each half state its own case."""
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    kernels = cfg.source("rirs-isotropic-rir")
    noise = cfg.source("rirs-isotropic-noise")
    assert kernels.root == noise.root, "same directory, different halves"
    assert kernels.blocked and not noise.blocked
    assert kernels.selects_name("RWCP_type4_rir_circle_ane_imp001.wav")
    assert not kernels.selects_name("RVB2014_type1_noise_largeroom1_1.wav")
    assert noise.selects_name("RVB2014_type1_noise_largeroom1_1.wav")
    assert not noise.selects_name("RWCP_type4_rir_circle_ane_imp001.wav")
    # no file can belong to both, or the census would count it twice
    for name in ("RVB2014_type1_noise_simroom1_3.wav",
                 "air_type1_air_binaural_stairway_1_1_1.wav"):
        assert kernels.selects_name(name) != noise.selects_name(name), name


def test_the_shipped_config_drops_both_rirs_sources_and_says_what_was_measured():
    """🔴 Both were registered in pool E and both are out, for reasons the run
    measured rather than assumed. `rirs-pointsource` is 843 of 843 byte-identical
    MUSAN copies and contributes 0.00 h MUSAN does not already have.
    `rirs-isotropic` is impulse responses -- 1.365 s median, 314 of 417 below the
    4 s sampler floor, up to 30 channels -- which belong to the reverb transform,
    not to a pool. Blocked rather than deleted so the finding keeps its evidence."""
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    for name in ("rirs-pointsource", "rirs-isotropic-rir"):
        assert cfg.source(name).blocked, name

    # The reason has to carry the measurement, not just a verdict -- a bare
    # "dropped" reads as an opinion the next session is free to reverse.
    assert "843 of 843" in cfg.source("rirs-pointsource").blocked
    assert "1.25 s median" in cfg.source("rirs-isotropic-rir").blocked

    # And the consequence: dropping them left `musan-noise` alone in pool E,
    # which `build_folds` cannot rotate. `compspoof-env-bonafide` is the answer
    # to that and was fetched for it -- so the invariant is that pool E never
    # goes back to a single runnable source, not that it has a particular one.
    runnable = {s.name for s in cfg.sources_in("E") if not s.blocked}
    assert runnable == {"musan-noise", "compspoof-env-bonafide",
                        "rirs-isotropic-noise"}


def test_the_grouping_gate_says_when_it_truncated_the_name_list(tmp_path):
    """🔴 `.head(5)` cut the list with nothing saying so: the detail read
    "6 of 9 source(s) below 6 ... : <five names>". A count that disagrees with
    the list it is followed by reads as a typo, and the name that got cut is
    the one the reader has not thought about yet -- it was SONICS."""
    groups = pd.DataFrame({"source_name": [f"s{i}" for i in range(7)],
                           "meets_floor": [False] * 6 + [True]})
    cfg = EdaConfig(root=tmp_path, out=tmp_path, sources=(), gates=GateConfig())
    res = _g_eda3(cfg, groups)
    assert res.verdict == FAIL
    assert "6 of 7" in res.detail and "(+1 more)" in res.detail

    # five or fewer and there is nothing to say
    groups["meets_floor"] = [False] * 5 + [True] * 2
    assert "more)" not in _g_eda3(cfg, groups).detail


def test_load_files_accepts_one_partition_named_as_a_string(tmp_path):
    """🔴 A bare `str` is a `Sequence[str]` of its own characters, so
    `load_files(cfg, "cell8")` asked for partitions c, e, l, l and 8. It
    survived because every pool name is one character -- `"E"` works by
    accident -- and only the first `cellN` partition exposed it."""
    root, out = tmp_path / "i", tmp_path / "o"
    src = SourceSpec(name="songs", row_kind="whole_file", cell=8, root="songs/v1",
                     suffixes=(".wav",))
    _write(root / "songs/v1/a.wav")
    cfg = EdaConfig(root=root, out=out, sources=(src,),
                    probe=ProbeConfig(workers=1, shard_size=4))
    probe_source(cfg, src)
    consolidate(cfg, "cell8")

    assert len(load_files(cfg, "cell8")) == 1
    assert len(load_files(cfg, ["cell8"])) == 1


def test_consolidate_skips_a_source_blocked_after_it_was_probed(tmp_path):
    """🔴 The parts of a dropped source stay on disk -- they are the evidence
    for dropping it. `consolidate` walks `cfg.sources_in(partition)`, which does
    not filter on `blocked`, so without this the 843 MUSAN copies would be merged
    straight back into the pool E census by the next run."""
    root, out = tmp_path / "i", tmp_path / "o"
    keep = SourceSpec(name="keep", pool="E", root="keep/v1", suffixes=(".wav",))
    drop = SourceSpec(name="drop", pool="E", root="drop/v1", suffixes=(".wav",))
    for src in (keep, drop):
        _write(root / src.root / "a.wav")
    cfg = EdaConfig(root=root, out=out, sources=(keep, drop),
                    probe=ProbeConfig(workers=1, shard_size=4))
    probe_source(cfg, keep)
    probe_source(cfg, drop)
    assert len(pd.read_parquet(consolidate(cfg, "E"))) == 2

    blocked = dataclasses.replace(drop, blocked="843 of 843 are byte-identical copies")
    cfg = dataclasses.replace(cfg, sources=(keep, blocked))
    df = pd.read_parquet(consolidate(cfg, "E"))
    assert list(df["source_name"]) == ["keep"]
    # and the parts it wrote are still there, unread
    assert list((out / "E/parts/drop").glob("*.parquet"))


def test_a_source_that_yields_no_files_raises_rather_than_returning_empty(tmp_path):
    """🔴 MLAAD proved this one. It is stored in S3 **unarchived** -- its
    `fake/<language>/<generator>/` tree intact, deliberately, because flattening
    it destroys the generator-disjoint split axis. `extract_archives` found no
    archive, created an empty `interim/mlaad/v9/`, and returned 0. The directory
    *existed*, so no FileNotFoundError fired and the corpus's best
    generator-diversity asset enumerated to zero with no error anywhere."""
    root = tmp_path / "i"
    (root / "empty/v1").mkdir(parents=True)
    cfg = EdaConfig(root=root, raw=tmp_path / "r", out=tmp_path / "o",
                    sources=(SourceSpec(name="empty", pool="A", root="empty/v1",
                                        suffixes=(".wav",)),))
    with pytest.raises(RuntimeError, match="yielded no files matching"):
        enumerate_source(cfg, cfg.source("empty"))

    # wrong suffix is the same defect wearing a different hat
    _write(root / "empty/v1/a.flac")
    with pytest.raises(RuntimeError, match=r"\['\.wav'\]"):
        enumerate_source(cfg, cfg.source("empty"))


def test_a_raw_stage_source_reads_from_the_sync_tree(tmp_path):
    """Two source shapes: unpacked from an archive (`interim`) and synced with
    its tree intact (`raw`, root carrying the `payload/` segment)."""
    interim, raw = tmp_path / "i", tmp_path / "r"
    _write(raw / "m/v9/payload/fake/mt/gen1/a.wav")
    _write(raw / "m/v9/payload/fake/mt/gen2/b.wav")
    interim.mkdir(parents=True, exist_ok=True)

    src = SourceSpec(name="m", pool="B", stage="raw", root="m/v9/payload",
                     suffixes=(".wav",))
    cfg = EdaConfig(root=interim, raw=raw, out=tmp_path / "o", sources=(src,),
                    probe=ProbeConfig(workers=1, shard_size=4))
    assert cfg.source_root(src) == raw / "m/v9/payload"
    files = _probe_all(cfg)
    assert len(files) == 2
    # file_id is relative to the source root, so the payload segment is not in it
    assert set(files["file_id"]) == {"m:fake/mt/gen1/a.wav", "m:fake/mt/gen2/b.wav"}
    # and `path` is relative to the stage base it actually came from
    assert files["path"].iloc[0].startswith("m/v9/payload/")

    with pytest.raises(ConfigError, match="stage must be one of"):
        SourceSpec(name="x", pool="A", root="x", stage="cooked")


def test_the_shipped_config_reads_mlaad_from_raw():
    from eda.config import load_eda_config

    cfg = load_eda_config("configs/eda.yaml")
    mlaad = cfg.source("mlaad")
    assert mlaad.stage == "raw" and mlaad.root.endswith("payload")
    # everything else is unpacked
    assert {s.stage for s in cfg.sources if s.name != "mlaad"} == {"interim"}


def test_probe_tolerates_absent_and_blocked_sources_but_not_empty_ones(tmp_path, capsys):
    """🔴 Found by the real run: `eda probe` died on the first source that was
    not fetched yet, which makes wave-by-wave probing impossible -- and staged
    fetching is the whole shape of docs/EDA/08.

    Three states, one of them fatal. **Absent** is the normal condition during a
    staged fetch; **blocked** is a recorded decision; **empty** stays fatal,
    because a source that exists and matches nothing is the silent failure
    `enumerate_source` refuses."""
    from eda import cli

    root = tmp_path / "i"
    _write(root / "have/v1/a.wav")
    (root / "hollow/v1").mkdir(parents=True)
    cfg = EdaConfig(
        root=root, raw=tmp_path / "r", out=tmp_path / "o",
        sources=(
            SourceSpec(name="have", pool="A", root="have/v1", suffixes=(".wav",)),
            SourceSpec(name="gone", pool="B", root="gone/v1", suffixes=(".wav",)),
            SourceSpec(name="stop", pool="B", root="have/v1", suffixes=(".wav",),
                       blocked="14.6% of it is bonafide"),
        ),
        probe=ProbeConfig(workers=1, shard_size=4))

    args = argparse.Namespace(partition=None, source=[], dry_run=False)
    assert cli.cmd_probe(cfg, args) == 0
    out = capsys.readouterr().out
    assert "1 probed, 1 absent, 1 blocked" in out
    assert "gone" in out and "bonafide" in out

    # an existing-but-empty source is a misconfiguration and stays fatal
    empty = dataclasses.replace(
        cfg, sources=(SourceSpec(name="hollow", pool="A", root="hollow/v1",
                                 suffixes=(".wav",)),))
    with pytest.raises(RuntimeError, match="yielded no files matching"):
        cli.cmd_probe(empty, args)

    # and nothing probed at all is a failure, not a quiet success
    none = dataclasses.replace(
        cfg, sources=(SourceSpec(name="gone", pool="B", root="gone/v1",
                                 suffixes=(".wav",)),))
    assert cli.cmd_probe(none, args) == 1


def test_an_allowlist_over_zero_rows_is_na_not_a_vacuous_pass(tmp_path):
    """🔴 Found by the first real run. `mtg-jamendo` is registered with an
    allowlist and is not fetched yet, so the gate saw zero rows, found zero
    violations, and reported **`all 0 rows permitted` -> PASS**. A gate that
    passes on an empty set has not checked anything -- the exact `na`-vs-`pass`
    confusion the tri-state exists to prevent, reappearing one level down."""
    lic = tmp_path / "_lic"
    lic.mkdir()
    (lic / "c.csv").write_text("id,verdict\n2,allow\n", encoding="utf-8")
    empty = pd.DataFrame({"file_id": [], "path": [], "source_name": [],
                          "pool": [], "cell": [], "row_kind": []})
    src = SourceSpec(name="c", pool="C", root="c/v1", allowlist="_lic/c.csv",
                     allowlist_key="int_stem", exclude=("junk",),
                     exclude_reason="macOS sidecar")
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "o", sources=(src,))
    g = run_gates(cfg, empty).set_index("gate")
    assert g.loc["G-EDA1/allowlist/c", "verdict"] == NA
    assert "vacuous pass" in g.loc["G-EDA1/allowlist/c", "detail"]
    # the exclusion half has the same hole and the same fix
    assert g.loc["G-EDA1/exclude/c", "verdict"] == NA

    # with rows present it is a real verdict again
    files = pd.DataFrame({"file_id": ["c:000002.mp3"], "path": ["c/v1/000002.mp3"],
                          "source_name": ["c"], "pool": ["C"], "cell": [None],
                          "row_kind": ["component"]})
    g = run_gates(cfg, files).set_index("gate")
    assert g.loc["G-EDA1/allowlist/c", "verdict"] == PASS


def test_sources_reports_a_raw_stage_source_as_present(tmp_path, capsys):
    """⚠️ `eda sources` is the verb a reader trusts to say what is on disk.
    Hardcoding `cfg.root / s.root` reported `mlaad` as absent while 16,006 of
    its files were already in the census -- the same stage confusion that cost
    pool B a third of its S tier.

    Mutation: `cfg.source_root(s)` put back to `cfg.root / s.root`.
    """
    raw = tmp_path / "raw"
    (raw / "src-r/v1/payload").mkdir(parents=True)
    cfg = EdaConfig(
        root=tmp_path / "interim", raw=raw, out=tmp_path / "out",
        sources=(SourceSpec(name="src-r", pool="A", root="src-r/v1/payload",
                            stage="raw"),))
    from eda.cli import cmd_sources

    cmd_sources(cfg, argparse.Namespace(partition=None))
    out = capsys.readouterr().out
    assert "True" in out, out
