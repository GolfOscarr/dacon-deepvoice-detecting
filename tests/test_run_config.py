"""`training.config`: the four run-determining dataclasses, from YAML.

Two properties, and they are different. **Reachable** -- every field can be set
from a file and lands -- is here. **Load-bearing** -- every field, once set,
moves something observable -- is a knob table per dataclass, in
`test_sampler.py`, `test_render.py`, `test_folds.py` and `test_loop.py`. A
config surface with only the first is how an ablation reports a difference it
never tested.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from training.config import (RESAMPLERS, SECTIONS, ConfigError, RunConfig,
                             dump_run_config, load_run_config,
                             run_config_from_dict)
from training.render import resample_poly_to
from training.sampler import CellMix, SamplerConfig

REPO = Path(__file__).resolve().parents[1]

#: One non-default value per field of every section. Critical: asserted
#: exhaustive below, so a field added to any of the four dataclasses arrives
#: with a value here or the suite goes red -- which is the only thing that stops
#: a knob from being added and then being unreachable from a config file.
_ALT: dict[str, dict] = {
    "sampler": {
        # Valid under C1 and C3 (gap 0.0019): 0.03 moved from cell 6 to cell 8.
        # A mix that violated either would be rejected by `SamplerConfig`, which
        # `test_a_violating_mix_is_rejected_through_yaml_too` covers.
        "cell_mix": {"p": {1: 0.060, 2: 0.130, 3: 0.060, 4: 0.135, 5: 0.155,
                           6: 0.095, 7: 0.125, 8: 0.125, 9: 0.115}},
        "f8": 1.0,
        "single_composed_rate": 0.5,
        "noise_composed_rate": 0.25,
        "balance_marginal_composedness": False,
        "domain_cap": 42,
        "duration_range": [5.0, 30.0],
        "gain_db_range": [-10.0, 10.0],
        "gain_db_mean": -1.0,
        "gain_db_sigma": 2.0,
        "sequential_prob": 0.75,
        "crossfade_ms_range": [5.0, 50.0],
        "silence_lead_s": 0.4,
        "silence_tail_s": 0.3,
        "scheme_version": "synthetic-v2",
        "allow_unsound_mix": True,
    },
    "render": {
        "root": "/corpus",
        "audio": {"sample_rate": 16_000, "min_seconds": 3.0, "max_seconds": 20.0,
                  "channels": "mid_side", "band_hz": [300.0, 3400.0]},
        "resampler": "training.render.resample_poly_to",
        "crossfade_shape": "linear",
        "check_duration": False,
    },
    "folds": {
        "n_folds": 3,
        "probe_share": 0.25,
        "scheme_version": "synthetic-v2",
        "caveat_families_per_val_fold": 4,
        "require_component_coverage": False,
        "allow_no_probe": True,
        "assigned_at": "2026-01-01T00:00:00",
        "probe_budget_drawable_only": False,
        "probe_min_real_hours": 1.0,
        "probe_min_real_atoms": 2,
        "balance_on": "rows",
        "caveat_min_role_hours": 0.0,
    },
    "loop": {
        "out_dir": "runs/alt",
        "n_buckets": 2,
        "ema_decay": 0.5,
        "grad_clip": 1.0,
        "device": "meta",
        "checkpoint_every": 3,
        "max_steps": 7,
        "render_workers": 2,
        "lr_schedule": "cosine",
        "warmup_steps": 5,
        "min_lr_ratio": 0.1,
        "frontend_lr_scale": 0.5,
        "log_every": 10,
        "grad_checkpointing": True,
        "render_on_ht_siblings": True,
        "grad_accum": 2,
        "frontend_hold_steps": 100,
    },
}


def test_the_alternative_value_table_names_every_field_of_every_section():
    """A field added without a value here is a field no config file can set."""
    assert set(_ALT) == set(SECTIONS)
    for name, cls in SECTIONS.items():
        assert set(_ALT[name]) == {f.name for f in dataclasses.fields(cls)}, name


# --------------------------------------------------------------------------- #
# round trip


def test_the_defaults_round_trip():
    cfg = RunConfig()
    assert run_config_from_dict(dump_run_config(cfg)) == cfg


def test_every_field_round_trips_at_a_non_default_value():
    """The defaults round-tripping proves almost nothing on its own: a loader
    that ignored a key entirely would still return the default it started at."""
    cfg = run_config_from_dict(_ALT)
    assert run_config_from_dict(dump_run_config(cfg)) == cfg
    for name in SECTIONS:
        assert getattr(cfg, name) != getattr(RunConfig(), name), name


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_every_field_arrives_from_yaml_with_the_value_the_file_gave(section,
                                                                    tmp_path):
    """Through an actual file, not through a dict: `yaml.safe_load` is where a
    quoted integer key or a `1e-3` string would go wrong."""
    path = tmp_path / f"{section}.yaml"
    path.write_text(yaml.safe_dump({section: _ALT[section]}, sort_keys=False))
    got = getattr(load_run_config(path), section)

    for field, want in _ALT[section].items():
        value = getattr(got, field)
        if field == "cell_mix":
            assert value == CellMix({int(k): v for k, v in want["p"].items()})
        elif field == "resampler":
            assert value is RESAMPLERS[want]
        elif field in ("root", "out_dir"):
            # Critical: a Path, not the string YAML gave. `Path("x") != "x"`,
            # so a str here silently fails every comparison downstream.
            assert isinstance(value, Path) and value == Path(want)
        elif isinstance(want, list):
            assert value == tuple(want)
        elif isinstance(want, dict):                       # nested dataclass
            assert dataclasses.is_dataclass(value)
            for k, v in want.items():
                assert getattr(value, k) == (tuple(v) if isinstance(v, list) else v)
        else:
            assert value == want, field


def test_the_dump_is_safe_dumpable():
    """Critical: `dataclasses.asdict` alone is not. It leaves `Path` objects and
    a bare function in the tree, and `safe_dump` refuses both -- so a ledger
    that serialised the config the obvious way would raise at write time."""
    text = yaml.safe_dump(dump_run_config(run_config_from_dict(_ALT)))
    assert run_config_from_dict(yaml.safe_load(text)) == run_config_from_dict(_ALT)

    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(dataclasses.asdict(RunConfig()))


# --------------------------------------------------------------------------- #
# what must be rejected


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_an_unknown_key_in_any_section_is_an_error(section):
    """`models.config`'s rule, for its reason: a typo'd knob that silently does
    nothing is how an ablation ends up measuring the wrong thing."""
    with pytest.raises(ConfigError, match="unknown key"):
        run_config_from_dict({section: {"not_a_knob": 1}})


def test_an_unknown_section_is_an_error():
    with pytest.raises(ConfigError, match="unknown section"):
        run_config_from_dict({"optimiser": {}})


def test_an_unresolvable_resampler_raises_rather_than_defaulting():
    """Critical: the failure mode this replaces is silent. A callable cannot come
    from YAML, so the two obvious designs both lose: ignoring the key gives the
    default kernel under a config that says otherwise, and dropping the field
    from the dump leaves the ledger unable to say which kernel a run used --
    while A-S1/G1 record swapping it as a live plan."""
    with pytest.raises(ConfigError, match="not a resampler"):
        run_config_from_dict({"render": {"resampler": "soxr_vhq"}})


def test_the_named_resampler_resolves_to_the_function():
    cfg = run_config_from_dict(
        {"render": {"resampler": "training.render.resample_poly_to"}})
    assert cfg.render.resampler is resample_poly_to
    assert dump_run_config(cfg)["render"]["resampler"] == \
        "training.render.resample_poly_to"


@pytest.mark.parametrize("keys", [
    {"1": .06, "2": .13, "3": .06, "4": .135, "5": .155,
     "6": .125, "7": .125, "8": .095, "9": .115},          # quoted YAML keys
    {1: .06, 2: .13, 3: .06, 4: .135, 5: .155,
     6: .125, 7: .125, 8: .095, 9: .115},                  # plain YAML keys
])
def test_cell_keys_may_be_written_either_way(keys):
    """`CellMix` compares `set(self.p)` against `set(CELL_TABLE)`, so string keys
    would fail with "needs all of 1-9" -- which reads as a missing cell rather
    than as a quoting mistake."""
    cfg = run_config_from_dict({"sampler": {"cell_mix": {"p": keys}}})
    assert cfg.sampler.cell_mix == CellMix()


def test_an_unknown_key_inside_cell_mix_is_an_error():
    with pytest.raises(ConfigError, match="unknown key"):
        run_config_from_dict({"sampler": {"cell_mix": {"probs": {}}}})


def test_a_violating_mix_is_rejected_through_yaml_too():
    """The C1/C3 gate is on the dataclass, so it cannot be routed around by
    coming in through a file -- which is the whole reason it was worth putting
    there before this module existed."""
    trapped = {1: .10, 2: .10, 3: .10, 4: .10, 5: .12,
               6: .15, 7: .15, 8: .10, 9: .08}
    with pytest.raises(ValueError, match="violates C3"):
        run_config_from_dict({"sampler": {"cell_mix": {"p": trapped}}})


# --------------------------------------------------------------------------- #
# the shipped example configs


def test_the_default_run_config_is_the_shipped_defaults():
    """If the file and the code disagree, one of them is lying about the run."""
    assert load_run_config(REPO / "configs/run_default.yaml") == RunConfig()


def test_the_t1_fallback_config_differs_only_in_f8():
    """T1's whole design is that strict composedness is a one-value change.

    A second file that had drifted in any other field would make the T1
    comparison a two-variable experiment, silently.
    """
    primary = load_run_config(REPO / "configs/run_default.yaml")
    fallback = load_run_config(REPO / "configs/run_t1_strict.yaml")
    assert primary.sampler.f8 == 0.0 and fallback.sampler.f8 == 1.0
    assert dataclasses.replace(primary.sampler, f8=1.0) == fallback.sampler
    for name in ("render", "folds", "loop"):
        assert getattr(primary, name) == getattr(fallback, name), name


@pytest.mark.parametrize("name", ["run_default", "run_t1_strict"])
def test_the_example_configs_name_every_field(name):
    """An example that omits a field teaches the omission.

    Every field written out, no inheritance: docs/validation/03's ledger only
    distinguishes two runs if a row names every determinant.
    """
    raw = yaml.safe_load((REPO / f"configs/{name}.yaml").read_text())
    assert set(raw) == set(SECTIONS)
    for section, cls in SECTIONS.items():
        assert set(raw[section]) == {f.name for f in dataclasses.fields(cls)}, section


def test_a_partial_config_is_legal_but_defaults_the_rest():
    """Stated so it is a decision rather than an accident. It is why the example
    configs are complete, and why there is no inheritance between them."""
    cfg = run_config_from_dict({"sampler": {"f8": 1.0}})
    assert cfg.sampler.f8 == 1.0
    assert cfg.render == RunConfig().render
    assert cfg.folds == RunConfig().folds
    assert cfg.loop == RunConfig().loop
    assert cfg.sampler == dataclasses.replace(SamplerConfig(), f8=1.0)
