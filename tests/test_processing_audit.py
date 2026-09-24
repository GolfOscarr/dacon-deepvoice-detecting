"""The processing audit: the harness's draw features per head, tiles counted
as one draw, and the three shortcuts docs/processing/02 measured -- H1 (pool
D's onset), H2 (the zero-filled canvas) and the raw lead -- reproduced as
FAILING on the training sampler and PASSING on the processing sampler.

Critical: an audit that cannot fail on the old draw is decoration. Each
shortcut is shown red on the sampler that manufactures it before it is
trusted green on the one that does not (docs/processing/03 §6 step 6).
"""

import pytest

from dataclasses import replace

from processing.audit import (HOP_S, audit_specs, collapse_tiles, draw_features,
                              run_audit)
from processing.config import DrawConfig
from processing.sampler import Sampler
from training.sampler import Sampler as TrainingSampler
from training.sampler import SamplerConfig
from training.synthetic import synthetic_manifest


@pytest.fixture(scope="module")
def manifest():
    """The real corpus's shape (02 §2): pool D is 10 s files against a 4-60 s
    timeline, pool A long, pool B short -- and whole-file rows only for the
    mixed cells 5 and 8 (the reassigned C/D rows and SONICS). Caveat: the
    synthetic default scrapes cells 1-4 whole too, and under ``f8 = 1`` with
    ``single_composed_rate = 0`` "not composed" then MEANS "single
    component": the presence heads read composedness at 0.86 for a reason the
    real corpus does not have."""
    df = synthetic_manifest(n_per_pool=200, n_whole_file=200, seed=0)
    df = df[(df.row_kind == "component") | df.cell.isin([5, 8])].reset_index(drop=True)
    df.loc[df.pool == "D", "duration_s"] = 10.0
    df.loc[df.pool == "C", "duration_s"] = 30.0
    b = df.index[df.pool == "B"]
    df.loc[b[: len(b) // 2], "duration_s"] = 4.5
    a = df.index[df.pool == "A"]
    df.loc[a[: len(a) // 4], "duration_s"] = 4.5
    # bucket tiling: voice buckets of ~100 files on BOTH sides, as on the
    # built corpus (75 to 12,000 per speaker bucket; 8 % of either pool's
    # files in buckets under 40). The synthetic corpus's 2-file real speakers
    # and 25-file fake families would exhaust under a 40-tile slot on one
    # side only, which is a fixture artefact, not a draw property.
    v = df.index[df.pool.isin(["A", "B"])]
    df.loc[v, "speaker_ref_id"] = [f"{p}_spk{i % 2}" for i, p in enumerate(df.loc[v, "pool"])]
    return df


#: The presence heads sit at the D-2 residual (~0.59 against the 0.60 gate),
#: so the probe needs the acceptance-size draw to resolve it; at 6,000 the
#: logistic fit over ten columns lands on either side of the gate.
N_AUDIT = 12_000


@pytest.fixture(scope="module")
def new_report(manifest):
    return run_audit(Sampler(manifest, DrawConfig()), manifest, n=N_AUDIT)


@pytest.fixture(scope="module")
def old_report(manifest):
    return run_audit(TrainingSampler(manifest, SamplerConfig(f8=1.0)), manifest, n=N_AUDIT)


# --------------------------------------------------------------------------- #
# tiles are one draw


def test_collapse_tiles_merges_a_component_into_one_draw(manifest):
    s = Sampler(manifest, DrawConfig())
    seen = 0
    for spec in s.epoch_specs(300):
        c = collapse_tiles(spec)
        roles = {(d.role, d.slot) for d in spec.components}
        assert len(c.components) == len(roles)
        for d in c.components:
            tiles = [t for t in spec.components if (t.role, t.slot) == (d.role, d.slot)]
            assert d.target_start_s == pytest.approx(min(t.target_start_s for t in tiles))
            assert d.target_start_s + d.duration_s == pytest.approx(
                max(t.target_start_s + t.duration_s for t in tiles))
            seen += len(tiles) > 1
        assert c.labels == spec.labels and c.transforms == spec.transforms
    assert seen > 100


def test_i3_counts_one_draw_per_spec_and_file(manifest, new_report):
    """`training.audit` read I3 at 0.125 on a tiled stream whose collapsed value
    is ~0.5: the audit must count the file, not its tiles."""
    passed, detail = new_report.ran["I3_real_components_on_both_sides"]
    assert passed, detail


# --------------------------------------------------------------------------- #
# the draw features


def test_draw_features_read_exposure_and_coverage(manifest):
    s = Sampler(manifest, DrawConfig())
    specs = list(s.epoch_specs(200))
    f = draw_features(specs, manifest)
    assert len(f) == 200
    assert (f["first_onset_exposed"] == 0).all()
    assert (f["uncovered_frac"] < 0.6).all()
    assert (f["drawn_lead_s"] <= 3.0 + 1e-9).all() and (f["drawn_lead_s"] > 0).any()
    for spec, (_, row) in zip(specs, f.iterrows()):
        for role in ("voice", "music"):
            present = getattr(spec, f"{role}_present")
            assert (row[f"{role}_take_s"] > 0) == bool(present)
            if present:
                tiles = [c for c in spec.components if c.role == role]
                assert row[f"{role}_joins"] == len(tiles) - 1


def test_the_training_draw_exposes_the_onset_and_leaves_the_canvas_silent(manifest):
    """H1 and H2 as the features read them on the old sampler."""
    specs = list(TrainingSampler(manifest, SamplerConfig(f8=1.0)).epoch_specs(500))
    f = draw_features(specs, manifest)
    # the training draw starts at 0 whenever the file is shorter than the
    # timeline: 10 s pool D against U(4, 60) is exposed ~85 % of the time
    assert f["first_onset_exposed"].mean() > 0.3
    music = f[f["music_take_s"] > 0]
    assert (music["music_onset_exposed"] == 1).mean() > 0.4
    assert (music["music_coverage"] < 0.8).mean() > 0.3


def test_the_file_lead_reaches_the_effective_lead_only_when_exposed(manifest):
    df = manifest.copy()
    df["lead_silence_s"] = 0.0
    df.loc[df.pool == "A", "lead_silence_s"] = 1.5
    old = draw_features(list(TrainingSampler(df, SamplerConfig(f8=1.0)).epoch_specs(300)), df)
    new = draw_features(list(Sampler(df, DrawConfig(silence_lead_s=0.0)).epoch_specs(300)), df)
    assert (old["voice_lead_silence_s"] >= 1.5).any()
    assert (new["voice_lead_silence_s"] == 0.0).all(), "never exposed, no drawn lead"


# --------------------------------------------------------------------------- #
# the shortcuts: red on the old sampler, green on the new


def _fails(report, key):
    passed, detail = report.ran[key]
    return not passed, detail


def test_h1_the_music_head_draw_shortcut_fails_on_the_old_sampler(old_report):
    failed, detail = _fails(old_report, "I1c_draw_shortcut_auc_music_fake")
    assert failed, detail
    assert "music-only=0.9" in detail or "pooled=0.9" in detail, detail


def test_h1_edge_exposure_fails_on_the_old_sampler(old_report):
    failed, detail = _fails(old_report, "I1d_edge_exposure")
    assert failed and "D=0." in detail and "D=0.00" not in detail, detail


def test_the_new_sampler_passes_every_draw_shortcut_gate(new_report):
    keys = [k for k in new_report.ran if k.startswith("I1c_") or k == "I1d_edge_exposure"]
    assert len([k for k in keys if k != "I1c_warnings"]) == 6
    for k in keys:
        passed, detail = new_report.ran[k]
        assert passed, f"{k}: {detail}"


def test_the_new_sampler_passes_the_whole_audit(new_report):
    assert new_report.ok, str(new_report)


def test_h2_the_zero_filled_canvas_is_a_music_cue_on_the_old_sampler(manifest):
    """Tiling off: the training rule leaves pool D's 10 s inside a longer
    timeline as silence, and the music component's coverage alone separates
    music_fake (pool C is 30 s, pool D 10 s)."""
    from sklearn.metrics import roc_auc_score
    specs = [s for s in TrainingSampler(manifest, SamplerConfig(f8=1.0)).epoch_specs(3000)
             if s.music_present]
    f = draw_features(specs, manifest)
    y = [s.music_fake for s in specs]
    auc = roc_auc_score(y, f["music_coverage"])
    assert max(auc, 1 - auc) > 0.8


def test_the_raw_lead_cue_fails_on_the_old_sampler_and_not_on_the_new(manifest):
    """A6b: the file's own leading silence. With pool A's files carrying a
    long lead and B's none, the training draw (onset exposed) hands the
    effective lead to the voice head; the inside-crop does not."""
    df = manifest.copy()
    df["lead_silence_s"] = 0.0
    df.loc[df.pool == "A", "lead_silence_s"] = 1.5
    old = run_audit(TrainingSampler(df, SamplerConfig(f8=1.0)), df, n=4000)
    failed, detail = _fails(old, "I1c_draw_shortcut_auc_voice_fake")
    assert failed, detail
    new = run_audit(Sampler(df, DrawConfig()), df, n=4000)
    passed, detail = new.ran["I1c_draw_shortcut_auc_voice_fake"]
    assert passed, detail


def test_a_presence_head_does_not_read_its_own_label(new_report, old_report):
    """A role's own columns ARE presence; the presence probes must not see
    them, or both samplers score 1.000 for a reason that is not a shortcut."""
    for report in (new_report, old_report):
        for head in ("voice_present", "music_present"):
            _, detail = report.ran[f"I1c_draw_shortcut_auc_{head}"]
            assert "=1.0000" not in detail, detail


def test_the_manifest_is_required(manifest):
    specs = list(Sampler(manifest, DrawConfig()).epoch_specs(10))
    with pytest.raises(TypeError):
        audit_specs(specs)


def test_hop_matches_the_harness():
    assert HOP_S == 0.05


def test_layers_are_collapsed_apart_from_components_and_read_by_the_features(manifest):
    s = Sampler(manifest, DrawConfig(p_noise_layer=1.0))
    specs = list(s.epoch_specs(200))
    f = draw_features(specs, manifest)
    assert (f["noise_layer"] == 1).all() and f["noise_layer_snr_db"].between(10, 30).all()
    for spec in specs:
        c = collapse_tiles(spec)
        layers = [d for d in c.components if d.snr_db is not None]
        assert len(layers) == 1 and layers[0].role == "noise"
        assert len([d for d in c.components if d.snr_db is None]) == len(
            {(d.role, d.slot) for d in spec.components if d.snr_db is None})


# --------------------------------------------------------------------------- #
# P6 (docs/processing/06): the gates the review found blind -- each mutation
# below passed the audit as it was (05 B9-B11) and must fail it now


class _Mutated:
    """A sampler whose specs are rewritten by ``fn`` after the draw."""

    def __init__(self, sampler, fn):
        self._s, self._fn, self.slice_, self.fold = sampler, fn, sampler.slice_, sampler.fold

    def epoch_specs(self, n, epoch=0, seed=0):
        for spec in self._s.epoch_specs(n, epoch=epoch, seed=seed):
            yield self._fn(spec)


def _shift(spec, seconds):
    """Every component ``seconds`` later on a timeline ``seconds`` longer."""
    comps = tuple(replace(c, target_start_s=c.target_start_s + seconds) for c in spec.components)
    return replace(spec, duration_s=spec.duration_s + seconds, components=comps)


N_MUT = 6_000


def test_a_cue_on_cell_nine_alone_fails_the_presence_heads(manifest):
    """05 B9: every probe dropped cell 9 while the presence heads train on it."""
    biased = _Mutated(Sampler(manifest), lambda s: _shift(s, 2.0) if s.cell == 9 else s)
    rep = run_audit(biased, manifest, n=N_MUT)
    failed = [k for k, (ok, _) in rep.ran.items() if not ok]
    assert any(k.startswith("I1c_draw_shortcut_auc_") and k.endswith("_present")
               for k in failed), failed


def test_a_transform_parameter_set_by_presence_fails_i1bp(manifest):
    """05 B10: rir's wet level set by music_present passed I1 and I1b."""
    def wet_by_music(s):
        wet = 0.9 if s.music_present else 0.3
        return replace(s, transforms=s.transforms + (("rir", {"wet": wet, "pick": 0.5}),))
    rep = run_audit(_Mutated(Sampler(manifest), wet_by_music), manifest, n=N_MUT)
    ok, detail = rep.ran["I1bp_transform_params_by_music_present"]
    assert not ok, detail
    assert rep.ran["I1bp_transform_params_by_voice_present"][0]


def test_a_normalize_draw_set_by_presence_fails_i1bp(manifest):
    def mono_iff_voice(s):
        return replace(s, normalize={**s.normalize,
                                     "channels": "mono" if s.voice_present else "stereo"})
    rep = run_audit(_Mutated(Sampler(manifest), mono_iff_voice), manifest, n=N_MUT)
    assert not rep.ran["I1bp_transform_params_by_voice_present"][0]


def test_a_transform_name_taken_by_presence_fails_i1p(manifest):
    def rir_when_music(s):
        if not s.music_present:
            return s
        return replace(s, transforms=s.transforms + (("rir", {"wet": 0.5, "pick": 0.5}),))
    rep = run_audit(_Mutated(Sampler(manifest), rir_when_music), manifest, n=N_MUT)
    ok, detail = rep.ran["I1p_transform_names_by_music_present"]
    assert not ok and "'rir'" in detail, detail


def test_a_sub_gate_cue_many_standard_errors_from_chance_is_a_warning(manifest):
    """05 B11: +0.3 s of lead on fakes only scored 0.597 -- under the gate,
    17 SE from chance. It stays under the gate and is now said out loud."""
    biased = _Mutated(Sampler(manifest), lambda s: _shift(s, 0.3) if s.file_fake else s)
    rep = run_audit(biased, manifest, n=N_MUT)
    _, detail = rep.ran["I1c_warnings"]
    assert "file_fake" in detail and "SE" in detail, detail
    clean = run_audit(Sampler(manifest), manifest, n=N_MUT)
    assert "file_fake" not in clean.ran["I1c_warnings"][1]


def test_a_bimodal_cue_the_linear_probe_misses_fails_on_the_folded_feature(manifest):
    """05 B11: fake leads at both ends of the range, real in the middle,
    scored 0.509 on the linear probe. The folded single feature reads it."""
    import numpy as np

    def bimodal(s):
        # the same mean shift on both sides: fakes get 0 or 2.5 s, reals 1.25 s
        rng = np.random.default_rng(s.sample_id)
        if s.file_fake:
            return _shift(s, 2.5) if rng.random() < 0.5 else s
        return _shift(s, 1.25)
    rep = run_audit(_Mutated(Sampler(manifest), bimodal), manifest, n=N_MUT)
    ok, detail = rep.ran["I1c_draw_shortcut_auc_file_fake"]
    assert not ok and "|x-med|" in detail, detail


def test_the_val_view_audit_accepts_the_folds_family_count(manifest):
    """05 C6: I21 needs 3 fake-music families and a VAL fold has 1 by design
    (D-22); the val view reads the floor from the view."""
    df = manifest.copy()
    df.loc[df.pool == "D", "artifact_family"] = "the-one-family"
    df.loc[df.pool == "D", "domain_key"] = "the-one-family"
    df["slice"] = "val"
    s = Sampler(df, DrawConfig().for_eval(), slice_="val", fold=0)
    rep = run_audit(s, df, n=4_000, view="val", eval_floors=False)
    ok, detail = rep.ran["I21_generator_diversity"]
    assert ok and "floor 1" in detail, detail
    train_like = run_audit(s, df, n=4_000)
    assert not train_like.ran["I21_generator_diversity"][0]
    assert rep.ran["I1_transform_name_independence"][0], "no augments in eval mode"
