"""The config is the architecture's contract, so its guards are worth testing.

A typo'd knob that silently does nothing is how an ablation ends up measuring
something other than what it claims. Every rejection below corresponds to a
decision recorded in docs/architecture/.
"""

import dataclasses
import pathlib

import pytest
import yaml

from metrics.dacon import PREDICTION_COLUMNS
from models.config import (
    LOSS_WEIGHT_KEYS,
    ConfigError,
    LossConfig,
    ModelConfig,
    dump_config,
    load_model_config,
    load_train_config,
    validate_model_config,
)
from eda.config import load_eda_config
from processing.config import load_processing_config
from training.config import load_run_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SHIPPED = sorted(CONFIGS.glob("*.yaml"))


def _a():
    return load_model_config(CONFIGS / "a_shared_trunk.yaml")


def _b():
    return load_model_config(CONFIGS / "b_three_branch.yaml")


def _raw(name):
    return yaml.safe_load((CONFIGS / name).read_text())


def _built(raw):
    """Build a ModelConfig from a raw dict via the public loader."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(raw, fh)
        return load_model_config(fh.name)


# --------------------------------------------------------------------------- #
# the shipped configs

#: The loaders a shipped config may belong to. A config must be accepted by
#: exactly one; the filename is not consulted.
LOADERS = {
    "model": load_model_config,
    "train": load_train_config,
    "run": load_run_config,
    # configs/eda.yaml. It ships here rather than under eda/ so that this sweep
    # sees it -- a config nobody classified is the thing the sweep exists to
    # catch, and hiding it in another directory would have satisfied the test by
    # removing it from the population.
    "eda": load_eda_config,
    # configs/processing_v1.yaml -- the data-processing pipeline's own
    # sections (`draw`, `render`), independent of the run configs.
    "processing": load_processing_config,
}


def _classify(path):
    """Which loaders accept `path`. Content decides, not the filename.

    Critical: this dispatched on the filename prefix with a `load_model_config`
    fallback in the `else` branch, so a config whose name matched no prefix was
    silently validated as a *model* config -- the sweep reported it as loaded
    while nothing had read its schema. That is the failure mode this test
    exists to catch, reintroduced inside the test itself. It only surfaced
    because `load_model_config` happens to reject unknown top-level keys; a
    laxer loader would have kept it green. Every loader here rejects unknown
    top-level keys, so acceptance is a real classification rather than a guess.
    """
    accepted = []
    for kind, load in LOADERS.items():
        try:
            load(path)
        except Exception:
            continue
        accepted.append(kind)
    return accepted


def test_every_shipped_config_loads():
    """Critical: every shipped config is accepted by exactly one loader.

    Not "each loader loads its own files" -- that phrasing passes while a
    config nobody classified sits unloaded. Exactly-one is what makes adding a
    fourth config kind turn this red instead of routing it to whichever loader
    happens not to complain.

    ⚠️ Both halves are reachable and both are covered below. An earlier version
    of this docstring said the ambiguity half "cannot currently fail" and stood
    down a mutation that returns the first acceptance; that was wrong. A config
    of nothing but defaults is accepted by `load_train_config` *and* by
    `load_run_config` today -- neither needs a single key to build a complete
    config -- so two loaders reading one file is a live failure mode, not a
    contract kept for a hypothetical future loader.
    """
    assert SHIPPED, "no configs found"
    for path in SHIPPED:
        accepted = _classify(path)
        assert accepted, (
            f"{path.name} is accepted by no loader: it ships in configs/ and "
            f"nothing can read it. Add its loader to LOADERS.")
        assert len(accepted) == 1, (
            f"{path.name} is accepted by {accepted} -- ambiguous. Two loaders "
            f"reading one file means neither owns its schema.")


def test_an_unclassifiable_config_is_not_silently_accepted(tmp_path):
    """The non-vacuity guard: prove the sweep above can fail.

    Caveat: without this, `test_every_shipped_config_loads` passes for exactly
    as long as every shipped file happens to be classifiable -- which is until
    someone adds a new kind, the moment it would have earned its keep.
    """
    orphan = tmp_path / "orphan.yaml"
    orphan.write_text("wholly_unknown_section:\n  a: 1\n")
    assert _classify(orphan) == [], (
        "a config belonging to no loader was accepted by one; the sweep would "
        "have reported it as loaded")


def test_a_config_two_loaders_accept_is_ambiguous_rather_than_the_first_one(tmp_path):
    """🔴 The other half of the guard, and it fails today -- not hypothetically.

    A config of nothing but defaults is accepted by `load_train_config` and by
    `load_run_config` both: neither schema requires a single key, so each builds
    a complete config out of an empty mapping and nothing in the file prefers
    either. `load_model_config` rejects it ("at least one branch is required"),
    which is the only reason this is a two-way tie rather than a three-way one.

    Critical: `_classify` must report the tie rather than resolve it. Returning
    the first acceptance is the same defect as the filename `else` branch it
    replaced -- a router picking a loader on something incidental, here dict
    ordering in LOADERS, and reporting the file as loaded. The sweep's
    `len(accepted) == 1` is what turns that into a red, so the tie has to reach
    it intact.
    """
    defaults = tmp_path / "nothing_but_defaults.yaml"
    defaults.write_text("{}\n")
    accepted = _classify(defaults)
    assert accepted == ["train", "run", "processing"], (
        "an all-defaults config is genuinely ambiguous between the train, run "
        "and processing loaders; _classify resolved it instead of reporting "
        "the tie")
    # `test_every_shipped_config_loads` asserts `len(accepted) == 1` on exactly
    # this value, so a tie that survives _classify is a red sweep. Asserting the
    # tie here rather than re-running the sweep keeps the message in one place.
    assert len(accepted) > 1


def test_a_and_b_produce_exactly_the_submission_columns():
    for cfg in (_a(), _b()):
        assert sorted(br.column for br in cfg.branches.values()) == sorted(PREDICTION_COLUMNS)


def test_b_contains_a_the_increment_is_one_frontend():
    """docs/architecture/03: 'A is stage one of B'. Assert it structurally."""
    a, b = _a(), _b()
    assert set(a.branches) == set(b.branches)
    assert len(a.frontends) == 1 and len(b.frontends) == 2
    assert set(a.frontends) < set(b.frontends), "B should keep A's frontend name"


def test_masks_mirror_the_metric():
    """Voice EER is computed only over voice-present files; the loss must match."""
    for cfg in (_a(), _b()):
        assert cfg.branches["voice"].masked_by == "voice_present"
        assert cfg.branches["music"].masked_by == "music_present"
        assert cfg.branches["file"].masked_by is None
        assert cfg.branches["v_pres"].masked_by is None
        assert cfg.branches["m_pres"].masked_by is None


def test_wav2vec_family_has_no_frequency_axis():
    """freq_pool is meaningless for (B,T,D) frontends and must be explicit."""
    b = _b()
    assert b.frontends["speech"].freq_pool.kind == "none"
    assert b.frontends["audio"].freq_pool.kind == "gem"


def test_round_trip_through_yaml_is_identity():
    for cfg in (_a(), _b()):
        again = _built(dump_config(cfg))
        assert again == cfg


# --------------------------------------------------------------------------- #
# guards

def test_unknown_key_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["embedding_size"] = 512          # plausible-looking, not a real knob
    with pytest.raises(ConfigError, match="unknown key"):
        _built(raw)


def test_unknown_nested_key_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["frontends"]["audio"]["hidden_dim"] = 42
    with pytest.raises(ConfigError, match="unknown key"):
        _built(raw)


def test_branch_source_must_be_a_declared_frontend():
    raw = _raw("a_shared_trunk.yaml")
    raw["branches"]["voice"]["sources"] = "speech"     # not declared in A
    with pytest.raises(ConfigError, match="not a declared frontend"):
        _built(raw)


def test_multi_source_branch_requires_align_to():
    raw = _raw("b_three_branch.yaml")
    del raw["branches"]["file"]["align_to"]
    with pytest.raises(ConfigError, match="align_to"):
        _built(raw)


def test_align_to_must_be_one_of_the_branches_own_sources():
    raw = _raw("b_three_branch.yaml")
    raw["branches"]["file"]["align_to"] = "nonexistent"
    with pytest.raises(ConfigError, match="align_to"):
        _built(raw)


def test_missing_submission_column_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    del raw["branches"]["m_pres"]
    with pytest.raises(ConfigError, match="exactly the five submission columns"):
        _built(raw)


def test_duplicate_column_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["branches"]["v_pres"]["column"] = "MUSIC_PRESENT_PROB"
    with pytest.raises(ConfigError, match="more than once"):
        _built(raw)


def test_float32_output_is_rejected():
    """Saturation took EER 0.0950 -> 0.3017; float32 sigmoid manufactures ties."""
    raw = _raw("a_shared_trunk.yaml")
    raw["output"]["dtype"] = "float32"
    with pytest.raises(ConfigError, match="float64"):
        _built(raw)


def test_hop_larger_than_window_is_rejected():
    """Skipping audio defeats the OR-over-segments label semantics."""
    raw = _raw("a_shared_trunk.yaml")
    raw["segmentation"] = {"mode": "tiling", "window_seconds": 5.0, "hop_seconds": 8.0}
    with pytest.raises(ConfigError, match="skip audio"):
        _built(raw)


def test_unknown_frontend_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["frontends"]["audio"]["name"] = "wav2vec9000"
    with pytest.raises(ConfigError, match="unknown frontend"):
        _built(raw)


def test_bad_mask_key_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["branches"]["voice"]["masked_by"] = "voice_fake"     # a label, not a mask
    with pytest.raises(ConfigError, match="must be one of"):
        _built(raw)


@pytest.mark.parametrize("branch,mask", [
    ("voice", "music_present"),      # the wrong pool, both keys valid
    ("voice", None),                 # unmasked: trains on music-only files
    ("music", "voice_present"),
    ("music", None),
    ("file", "voice_present"),       # File EER is over every file
    ("v_pres", "voice_present"),     # circular: masked by its own target
    ("m_pres", "music_present"),
])
def test_a_mask_that_is_not_the_metrics_pool_is_rejected(branch, mask):
    """🔴 The column decides the mask, because the metric decides the pool.

    Validation used to check only that `masked_by` was in MASK_KEYS or null, so
    `voice: masked_by: music_present` alongside `music: masked_by: null` loaded
    clean -- a voice head trained on the files the metric never scores it on,
    which is the exact defect the masks exist to prevent. Both shipped configs
    are correct and pinned by `test_masks_mirror_the_metric`; this protects the
    next one.
    """
    raw = _raw("a_shared_trunk.yaml")
    raw["branches"][branch]["masked_by"] = mask
    with pytest.raises(ConfigError, match="is scored over"):
        _built(raw)


def test_unfrozen_frontend_with_adapters_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["frontends"]["audio"]["freeze"] = False
    with pytest.raises(ConfigError, match="adapter"):
        _built(raw)


@pytest.mark.parametrize("field,value", [
    ("kind", "median"),
    ("k", 0),
    ("quantile", 1.5),
])
def test_bad_aggregation_is_rejected(field, value):
    raw = _raw("a_shared_trunk.yaml")
    raw["aggregation"][field] = value
    with pytest.raises(ConfigError):
        _built(raw)


def test_train_config_rejects_unknown_head():
    import tempfile
    raw = yaml.safe_load((CONFIGS / "train_joint.yaml").read_text())
    raw["loss"]["weights"]["drums"] = 1.0
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(raw, fh)
        with pytest.raises(ConfigError, match="unknown head"):
            load_train_config(fh.name)


def test_train_config_requires_every_head_weight():
    """🔴 A partial `weights` dict used to load clean and default the rest.

    `weights: {file: 0.45}` passed the unknown-key check -- every key it carried
    *was* known -- and `multitask_loss` filled the four absent heads in at its
    own 1.0 fallback. That trains both presence heads at 20x the 0.05 the metric
    gives them, with nothing wrong-looking in the YAML to see (docs/training/02
    §4). Now the five keys are required, and the loss indexes rather than
    `.get`s them.
    """
    import tempfile
    raw = yaml.safe_load((CONFIGS / "train_joint.yaml").read_text())
    raw["loss"]["weights"] = {"file": 0.45}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(raw, fh)
        with pytest.raises(ConfigError, match="missing head"):
            load_train_config(fh.name)


def test_loss_config_rejects_a_partial_weights_dict_however_it_is_built():
    """The rule is on LossConfig, not on the YAML loader.

    A `dataclasses.replace` or a checkpoint round-trip reaches the same object
    without passing through `load_train_config`, so the check lives where every
    path meets.
    """
    with pytest.raises(ConfigError, match="missing head"):
        LossConfig(weights={"file": 0.45})
    with pytest.raises(ConfigError, match="unknown head"):
        LossConfig(weights=dict(LossConfig().weights, drums=1.0))
    # The default is complete, and replacing one weight keeps it so.
    assert set(LossConfig().weights) == set(LOSS_WEIGHT_KEYS)
    assert dataclasses.replace(
        LossConfig(), weights=dict(LossConfig().weights, file=0.9)).weights["file"] == 0.9


def test_configs_are_frozen():
    cfg = _a()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


# --------------------------------------------------------------------------- #
# knobs added after auditing the config against docs/architecture/

def test_low_band_member_is_expressible():
    """09 B7 / D9: low-frequency subbands cut EER up to 25% relative under codecs."""
    raw = _raw("a_shared_trunk.yaml")
    raw["audio"]["band_hz"] = [0, 4000]
    cfg = _built(raw)
    assert cfg.audio.band_hz == (0, 4000)


def test_band_above_nyquist_is_rejected():
    raw = _raw("a_shared_trunk.yaml")
    raw["audio"]["band_hz"] = [0, 12000]          # 16 kHz => Nyquist 8 kHz
    with pytest.raises(ConfigError, match="Nyquist"):
        _built(raw)


def test_stereo_policy_is_expressible_and_checked():
    raw = _raw("a_shared_trunk.yaml")
    raw["audio"]["channels"] = "mid_side"
    assert _built(raw).audio.channels == "mid_side"
    raw["audio"]["channels"] = "surround"
    with pytest.raises(ConfigError, match="channels"):
        _built(raw)


def test_sample_rate_is_pinned_to_16k():
    raw = _raw("a_shared_trunk.yaml")
    raw["audio"]["sample_rate"] = 22050
    with pytest.raises(ConfigError, match="16000"):
        _built(raw)


def test_runtime_precision_is_expressible():
    raw = _raw("b_three_branch.yaml")
    assert _built(raw).runtime.compile is False, "compile defaults off (06 §4)"
    raw["runtime"]["precision"] = "int4"
    with pytest.raises(ConfigError, match="precision"):
        _built(raw)


def test_teachers_carry_a_full_frontend_spec():
    """A teacher needs a weights path and depth, not just a name."""
    cfg = load_train_config(CONFIGS / "train_joint.yaml")
    assert cfg.teachers, "expected distillation teachers"
    for name, fe in cfg.teachers.items():
        assert fe.freeze, f"{name} must be frozen"
        assert fe.output_dim > 0


def test_unfrozen_teacher_is_rejected():
    import tempfile
    raw = yaml.safe_load((CONFIGS / "train_joint.yaml").read_text())
    next(iter(raw["teachers"].values()))["freeze"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(raw, fh)
        with pytest.raises(ConfigError, match="must be frozen"):
            load_train_config(fh.name)


def test_c_lite_is_a_one_line_config_change():
    """C-lite = B + an auxiliary separation head discarded at inference."""
    raw = _raw("b_three_branch.yaml")
    raw["aux"]["separation_head"] = True
    assert _built(raw).aux.separation_head is True
