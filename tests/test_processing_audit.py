"""The processing audit: the harness's draw features per head, tiles counted
as one draw, and the three shortcuts docs/processing/02 measured -- H1 (pool
D's onset), H2 (the zero-filled canvas) and the raw lead -- reproduced as
FAILING on the training sampler and PASSING on the processing sampler.

Critical: an audit that cannot fail on the old draw is decoration. Each
shortcut is shown red on the sampler that manufactures it before it is
trusted green on the one that does not (docs/processing/03 §6 step 6).
"""

import pytest

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
    return df


@pytest.fixture(scope="module")
def new_report(manifest):
    return run_audit(Sampler(manifest, DrawConfig()), manifest, n=6000)


@pytest.fixture(scope="module")
def old_report(manifest):
    return run_audit(TrainingSampler(manifest, SamplerConfig(f8=1.0)), manifest, n=6000)


# --------------------------------------------------------------------------- #
# tiles are one draw


def test_collapse_tiles_merges_a_component_into_one_draw(manifest):
    s = Sampler(manifest, DrawConfig())
    seen = 0
    for spec in s.epoch_specs(300):
        c = collapse_tiles(spec)
        roles = {(d.role, d.file_id) for d in spec.components}
        assert len(c.components) == len(roles)
        for d in c.components:
            tiles = [t for t in spec.components if (t.role, t.file_id) == (d.role, d.file_id)]
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
    assert len(keys) == 6
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
