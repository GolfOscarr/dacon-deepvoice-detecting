#!/usr/bin/env python3
"""Analysis 0 -- the stream harness (docs/processing/01 §2).

Measures the shortcuts on the *rendered stream the model sees*, not on raw
files: draw specs with the real `training.sampler.Sampler`, join every drawn
component to its S-tier / envelope measurements, derive the sample's effective
features (component lengths, timeline coverage, whether the file's first frame
is in the sample, effective leading silence, level) and audit them per head
under an ungrouped and a `source_name`-grouped holdout, worst stratum taken.

Policies (docs/processing/01 §3.1 / §3.4 / §3.2) are simulated *on the drawn
stream* -- a re-draw of the take / offset / timeline with a policy RNG. The
labels never move, so the audit is the same measurement a sampler change would
produce, at a fraction of the cost. The sampler change is the implementation.

    V=/data/project/private/dacon-venvs/dacon311/bin/python
    $V scripts/strategy/stream_harness.py            # writes eda/out/_strategy/

No audio is decoded.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eda.analyze.screens import generator_key                      # noqa: E402
from eda.analyze.shortcut import univariate_auc                    # noqa: E402
from eda.config import load_eda_config                             # noqa: E402
from eda.driver import load_signal                                 # noqa: E402
from eda.extract.envelope import HOP_S                             # noqa: E402
from training.config import load_run_config                        # noqa: E402
from training.manifest import (POOL_IS_FAKE, POOL_LABELS,          # noqa: E402
                               REQUIRED_COLUMNS, validate_manifest)
from training.sampler import Sampler                               # noqa: E402
from training.spec import CELL_TABLE, is_fake_cell, stratum_of     # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = ROOT / "eda" / "out" / "_strategy"
N_SPECS = 20_000
SEED = 0
GATE = 0.60
ROLES = ("voice", "music", "noise")
HEADS = ("music_fake", "voice_fake", "file_fake", "music_present", "voice_present")

SIGNAL_COLS = ["rms_dbfs_chain", "peak_dbfs_chain", "crest_factor_db_chain",
               "clipping_ratio_chain", "lead_silence_s_chain", "tail_silence_s_chain",
               "silence_ratio_chain", "effective_bandwidth_hz_chain",
               "near_nyquist_ratio_chain", "dc_offset_chain", "band_energy_0_chain",
               "duration_s_decoded_chain"]
ENVELOPE_COLS = ["onset_level_deficit_db", "offset_level_deficit_db"]


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def build_manifest(signal: pd.DataFrame) -> pd.DataFrame:
    """A `training.manifest` over the S-tier rows, so every drawn component has
    measurements. Labels from POOL_LABELS / CELL_TABLE; families from the
    generator directory (a placeholder for §3.6's table)."""
    df = signal.copy()
    gen = generator_key(df)
    comp = df["row_kind"] == "component"
    whole = ~comp
    fake_comp = comp & df["pool"].map(POOL_IS_FAKE).fillna(False).astype(bool)
    cell = pd.to_numeric(df["cell"], errors="coerce")
    fake_whole = whole & cell.map(lambda c: pd.notna(c) and is_fake_cell(int(c))).fillna(False)

    m = pd.DataFrame({
        "file_id": df["file_id"], "path": df["path"], "sha256": df["sha256"],
        "row_kind": df["row_kind"], "pool": df["pool"].where(comp),
        "cell": cell.where(whole).astype("Int64"),
        "duration_s": df["duration_s"].astype(float),
        "orig_sr": df["orig_sr"], "orig_channels": df["orig_channels"],
        "container": df["container"],
    })
    labels = np.array([POOL_LABELS[p] if pd.notna(p) else CELL_TABLE[int(c)]
                       for p, c in zip(m["pool"], m["cell"])], dtype=object)
    for i, col in enumerate(("label_voice_present", "label_music_present",
                             "label_voice_fake", "label_music_fake")):
        m[col] = pd.array([lab[i] for lab in labels], dtype="Int64")
    family = gen.where(gen.notna(), df["group_key"])
    m["artifact_family"] = family.where(fake_comp | fake_whole)
    m["source_name"] = df["source_name"]
    m["speaker_ref_id"] = df["group_key"]
    m["pair_id"] = pd.NA
    m["dup_group"] = pd.NA
    m["domain_key"] = (df["source_name"] + "|" + family.astype(str)).where(fake_comp | fake_whole)
    m["slice"] = "train"
    m["fold"] = pd.array([pd.NA] * len(m), dtype="Int64")
    m["scheme_version"] = "strategy-v0"
    m["validity_mask_ref"] = pd.NA
    m["label_confidence"] = "reported"
    m["aug_strength"] = 0.0
    m = m[list(REQUIRED_COLUMNS)].reset_index(drop=True)
    return validate_manifest(m)


def apply_reassignment(manifest: pd.DataFrame, worklist: pd.DataFrame) -> pd.DataFrame:
    """§3.5 / D3 as a manifest edit: `reassign_cell` rows become whole_file rows
    of their target cell; `degenerate` rows are dropped; `restrict_noise` rows
    are dropped from pool E (option ii -- option i needs a sampler feature)."""
    m = manifest.set_index("file_id")
    wl = worklist.set_index("file_id")
    wl = wl[wl.index.isin(m.index)]
    re_ = wl[wl.action == "reassign_cell"]
    for fid, row in re_.iterrows():
        cell = int(row.target_cell)
        m.at[fid, "row_kind"] = "whole_file"
        m.at[fid, "cell"] = cell
        m.at[fid, "pool"] = pd.NA
        for i, col in enumerate(("label_voice_present", "label_music_present",
                                 "label_voice_fake", "label_music_fake")):
            m.at[fid, col] = CELL_TABLE[cell][i]
        if not is_fake_cell(cell):
            m.at[fid, "artifact_family"] = pd.NA
            m.at[fid, "domain_key"] = pd.NA
    drop = wl[wl.action.isin(["degenerate", "restrict_noise"])].index
    m = m.drop(index=drop)
    m["cell"] = m["cell"].astype("Int64")
    out = m.reset_index()[list(REQUIRED_COLUMNS)]
    print(f"reassigned {len(re_)} rows ({re_.target_cell.value_counts().to_dict()}), dropped {len(drop)}")
    return validate_manifest(out)


# --------------------------------------------------------------------------- #
# draw and flatten
# --------------------------------------------------------------------------- #

def draw_stream(manifest: pd.DataFrame, sampler_cfg, n: int, seed: int) -> pd.DataFrame:
    sampler = Sampler(manifest, sampler_cfg, slice_="train")
    rows = []
    for spec in sampler.epoch_specs(n, epoch=0, seed=seed):
        r = {"sample_id": spec.sample_id, "cell": spec.cell,
             "duration_s": spec.duration_s, "composed": spec.render_mode == "composed",
             "sequential": spec.structure == "sequential",
             "crossfade_ms": spec.crossfade_ms, "n_components": len(spec.components)}
        for role in ROLES:
            r.update({f"{role}_file": None, f"{role}_take": 0.0, f"{role}_offset": np.nan,
                      f"{role}_start": np.nan, f"{role}_gain": 0.0})
        for c in spec.components:
            r[f"{c.role}_file"] = c.file_id
            r[f"{c.role}_take"] = c.duration_s
            r[f"{c.role}_offset"] = c.source_offset_s
            r[f"{c.role}_start"] = c.target_start_s
            r[f"{c.role}_gain"] = c.gain_db
        # A whole-file row asserting both components (cell 5 / 8) is ONE file
        # carrying voice and music at once: it is both roles, or the music head
        # sees `music_take = 0` on every AI song and reads that as the label.
        vp, mp = CELL_TABLE[spec.cell][:2]
        if spec.render_mode == "whole_file" and vp and mp:
            for k in ("file", "take", "offset", "start", "gain"):
                r[f"music_{k}"] = r[f"voice_{k}"]
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["n_dropped_short"] = sampler.n_dropped_short
    vp, mp, vf, mf = zip(*(CELL_TABLE[c] for c in out["cell"]))
    out["voice_present"] = list(vp)
    out["music_present"] = list(mp)
    out["voice_fake"] = pd.array(list(vf), dtype="Int64")
    out["music_fake"] = pd.array(list(mf), dtype="Int64")
    out["file_fake"] = [int(is_fake_cell(c)) for c in out["cell"]]
    out["stratum"] = [stratum_of(c) for c in out["cell"]]
    return out


def attach_measurements(stream: pd.DataFrame, signal: pd.DataFrame,
                        envelope: pd.DataFrame) -> pd.DataFrame:
    """Per role, join the drawn file's S-tier scalars and its envelope."""
    meas = signal.set_index("file_id")[["source_name", "pool", "duration_s"] + SIGNAL_COLS]
    env = envelope.set_index("file_id")[ENVELOPE_COLS]
    meas = meas.join(env, how="left")
    out = stream.copy()
    for role in ROLES:
        j = meas.reindex(out[f"{role}_file"].values)
        j.index = out.index
        for col in j.columns:
            out[f"{role}__{col}"] = j[col].values
    return out


# --------------------------------------------------------------------------- #
# effective features
# --------------------------------------------------------------------------- #

def effective_features(s: pd.DataFrame, drawn_lead: pd.Series | None = None) -> pd.DataFrame:
    """The sample as the model receives it. `drawn_lead` is the A-A11 lead
    silence per sample (0 today), added to every present role's exposed lead."""
    f = pd.DataFrame(index=s.index)
    f["sample_duration_s"] = s["duration_s"]
    f["composed"] = s["composed"].astype(float)
    f["sequential"] = s["sequential"].astype(float)
    dur = s["duration_s"]
    covered = np.zeros(len(s))
    for role in ROLES:
        take = s[f"{role}_take"].fillna(0.0)
        present = take > 0
        offset = s[f"{role}_offset"]
        filedur = s[f"{role}__duration_s_decoded_chain"].fillna(s[f"{role}__duration_s"])
        exposed_onset = (present & (offset < HOP_S)).astype(float)
        exposed_end = (present & (offset + take >= filedur - HOP_S)).astype(float)
        if f"{role}_tiled" in s.columns:
            # tiled from random-offset crops: the file's end lands at a join, if
            # at all, never at the sample's end -- the first tile's onset can
            exposed_end = exposed_end.where(~s[f"{role}_tiled"], 0.0)
        f[f"{role}_take_s"] = take
        f[f"{role}_coverage"] = take / dur
        f[f"{role}_onset_exposed"] = exposed_onset
        f[f"{role}_onset_deficit_x"] = s[f"{role}__onset_level_deficit_db"].fillna(0.0) * exposed_onset
        f[f"{role}_end_exposed"] = exposed_end
        f[f"{role}_offset_deficit_x"] = s[f"{role}__offset_level_deficit_db"].fillna(0.0) * exposed_end
        f[f"{role}_rms_dbfs"] = (s[f"{role}__rms_dbfs_chain"] + s[f"{role}_gain"]).where(present, np.nan)
        f[f"{role}_crest_db"] = s[f"{role}__crest_factor_db_chain"].where(present, np.nan)
        f[f"{role}_bandwidth_hz"] = s[f"{role}__effective_bandwidth_hz_chain"].where(present, np.nan)
        f[f"{role}_near_nyquist"] = s[f"{role}__near_nyquist_ratio_chain"].where(present, np.nan)
        f[f"{role}_dc_offset"] = s[f"{role}__dc_offset_chain"].abs().where(present, np.nan)
        f[f"{role}_band0"] = s[f"{role}__band_energy_0_chain"].where(present, np.nan)
        lead_file = s[f"{role}__lead_silence_s_chain"].fillna(0.0) * exposed_onset
        f[f"{role}_lead_silence_s"] = lead_file.where(present, np.nan)
        # trailing silence of the file, exposed only when its end is in the sample
        f[f"{role}_tail_silence_s"] = (s[f"{role}__tail_silence_s_chain"].fillna(0.0)
                                       * exposed_end).where(present, np.nan)
        # union coverage: overlap components share a start; sequential ones abut
        covered = np.where(s["sequential"], covered + take, np.maximum(covered, take))
    if drawn_lead is not None:
        # A-A11 is drawn per SAMPLE, before the cell decides which roles exist
        # (sampler.py draws `lead` ahead of `wanted`), so it lands on every
        # present role -- adding it to the voice role alone made `any_lead`
        # a presence cue (0.968) in the first run. Measured, then fixed.
        for role in ROLES:
            f[f"{role}_lead_silence_s"] = f[f"{role}_lead_silence_s"] + drawn_lead.where(f[f"{role}_take_s"] > 0, 0.0)
    f["uncovered_frac"] = np.clip(1.0 - covered / dur, 0.0, 1.0)
    # role-agnostic aggregates, for the presence heads
    f["first_onset_exposed"] = f[[f"{r}_onset_exposed" for r in ROLES]].max(axis=1)
    f["first_onset_deficit_x"] = f[[f"{r}_onset_deficit_x" for r in ROLES]].max(axis=1)
    f["mean_rms_dbfs"] = f[[f"{r}_rms_dbfs" for r in ROLES]].mean(axis=1)
    f["min_bandwidth_hz"] = f[[f"{r}_bandwidth_hz" for r in ROLES]].min(axis=1)
    f["any_lead_silence_s"] = f[[f"{r}_lead_silence_s" for r in ROLES]].max(axis=1)
    return f


#: Two feature families, audited separately, because they answer different
#: questions. DRAW features are created by the sampler (lengths, coverage,
#: whether a file's first frame is in the sample, join counts, composedness):
#: they carry no acoustic content, so any AUC above chance -- ungrouped -- is a
#: cue the pipeline manufactured. ACOUSTIC residues are per-file measurements the
#: model legitimately receives; for those the archive-grouped AUC is the
#: question (does it transfer across publishers, or is it a fingerprint?).
#: A presence head cannot be audited on `music_take_s` -- it is 0 exactly when
#: music is absent, which is the label -- so it gets role-agnostic aggregates.
DRAW_SUFFIXES = ("_take_s", "_coverage", "_onset_exposed", "_end_exposed",
                 "_lead_silence_s", "_tail_silence_s", "_joins")
ACOUSTIC_SUFFIXES = ("_onset_deficit_x", "_offset_deficit_x", "_rms_dbfs", "_crest_db",
                     "_bandwidth_hz", "_near_nyquist", "_dc_offset", "_band0")


def feature_sets(f: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    role = [c for c in f.columns if c.startswith(("voice_", "music_"))]
    draw_role = [c for c in role if c.endswith(DRAW_SUFFIXES)]
    ac_role = [c for c in role if c.endswith(ACOUSTIC_SUFFIXES)]
    draw_common = ["sample_duration_s", "composed", "sequential", "uncovered_frac"]
    draw_presence = draw_common + ["first_onset_exposed", "any_lead_silence_s"]
    ac_presence = ["first_onset_deficit_x", "mean_rms_dbfs", "min_bandwidth_hz"]
    out = {}
    for head in ("music_fake", "voice_fake", "file_fake"):
        out[(head, "draw")] = draw_common + draw_role
        out[(head, "acoustic")] = ac_role
    for head in ("music_present", "voice_present"):
        out[(head, "draw")] = draw_presence
        out[(head, "acoustic")] = ac_presence
    return out


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #

def _cv_auc(x, y, groups, grouped: bool, n_splits: int = 5) -> float:
    minority = int(np.bincount(y).min())
    k = max(2, min(n_splits, minority))
    if grouped:
        ng = len(np.unique(groups))
        if ng < 2:
            return float("nan")
        # Four music sources: holding one out leaves a single-class fold, so the
        # only measurable archive holdout is 2-fold (one real + one fake source
        # per side). Reported as such -- it is the weaker measurement.
        splitter, args = StratifiedGroupKFold(min(k, ng) if ng >= 5 else 2), (x, y, groups)
    else:
        splitter, args = StratifiedKFold(k, shuffle=True, random_state=SEED), (x, y)
    scores = []
    for tr, te in splitter.split(*args):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
        model.fit(x[tr], y[tr])
        scores.append(roc_auc_score(y[te], model.predict_proba(x[te])[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def _group_for(s: pd.DataFrame, head: str) -> pd.Series:
    """The archive holdout axis: the source of the component the head is about;
    for the file head and the presence heads, the music source when music is
    present, else the voice source, else the noise source."""
    if head.startswith("music"):
        primary = s["music__source_name"]
    elif head.startswith("voice"):
        primary = s["voice__source_name"]
    else:
        primary = s["music__source_name"].fillna(s["voice__source_name"])
    return primary.fillna(s["noise__source_name"]).fillna("none")


def audit(s: pd.DataFrame, f: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows = []
    sets = feature_sets(f)
    for (head, family), cols in sets.items():
        y_all = pd.to_numeric(s[head], errors="coerce")
        scored = y_all.notna()
        if head == "file_fake":
            scored &= s["cell"] != 9            # I1b: cell 9 is always real, never composed
        strata = ["pooled"] + (sorted(s.loc[scored, "stratum"].unique())
                               if head.endswith("fake") else [])
        for stratum in strata:
            mask = scored & ((s["stratum"] == stratum) if stratum != "pooled" else True)
            sub = f.loc[mask, [c for c in cols if c in f.columns]]
            keep = [c for c in sub.columns if sub[c].notna().any() and sub[c].nunique() > 1]
            sub = sub[keep]
            if not keep:
                continue
            x = sub.fillna(sub.median()).to_numpy(dtype=float)
            y = y_all[mask].astype(int).to_numpy()
            if len(np.unique(y)) < 2 or len(y) < 50:
                continue
            groups = _group_for(s, head)[mask].to_numpy()
            uni = sorted(((univariate_auc(x[:, i], y), keep[i]) for i in range(x.shape[1])),
                         reverse=True)[:5]
            rows.append({
                "policy": policy, "head": head, "family": family, "stratum": stratum,
                "n": int(len(y)), "positive_rate": float(y.mean()),
                "auc": _cv_auc(x, y, groups, grouped=False),
                "auc_source_grouped": _cv_auc(x, y, groups, grouped=True),
                "n_groups": int(len(np.unique(groups))),
                "top_features": "; ".join(f"{n}={a:.3f}" for a, n in uni),
            })
    out = pd.DataFrame(rows)
    # the gate: draw features must be at chance ungrouped; acoustic residues are
    # judged on whether they survive the archive holdout
    judged = np.where(out["family"] == "draw", out["auc"],
                      out["auc_source_grouped"].fillna(out["auc"]))
    out["gate"] = np.where(judged < GATE, "pass", "fail")
    return out


# --------------------------------------------------------------------------- #
# policies, simulated on the stream
# --------------------------------------------------------------------------- #

def _redraw_music(s: pd.DataFrame, rng, take_lo: float, take_hi: float,
                  timeline: str = "keep") -> pd.DataFrame:
    """P1: music take ~ U(lo, hi) capped by the file, offset U(0, file - take).
    `timeline='cover'` (P2) shrinks the sample timeline to the longest take when
    that is shorter than the draw; `'reject'` (P4) keeps coverage >= 0.8 by
    re-drawing the timeline in [4, take/0.8]."""
    s = s.copy()
    present = (s["music_take"] > 0).to_numpy()
    filedur = s["music__duration_s_decoded_chain"].fillna(s["music__duration_s"]).fillna(0.0).to_numpy()
    take = np.minimum(rng.uniform(take_lo, take_hi, len(s)), filedur)
    span = np.where(s["sequential"], s["duration_s"] / s["n_components"].clip(lower=1), s["duration_s"])
    take = np.minimum(take, span)
    high = np.maximum(0.0, filedur - take)
    offset = rng.uniform(0.0, 1.0, len(s)) * high
    s.loc[present, "music_take"] = take[present]
    s.loc[present, "music_offset"] = offset[present]
    if timeline == "cover":
        longest = s[[f"{r}_take" for r in ROLES]].max(axis=1)
        s["duration_s"] = np.minimum(s["duration_s"], np.maximum(4.0, longest))
    elif timeline == "reject":
        longest = s[[f"{r}_take" for r in ROLES]].max(axis=1)
        cap = np.clip(longest / 0.8, 4.0, 60.0)
        s["duration_s"] = np.where(s["duration_s"] > cap, rng.uniform(4.0, cap), s["duration_s"])
    return s


def _redraw_all(s: pd.DataFrame, rng, take_lo: float, take_hi: float) -> pd.DataFrame:
    """P5/P6: every role's take ~ U(lo, hi) capped by its file, offset random.
    A whole-file row that is both roles is redrawn once and copied."""
    s = s.copy()
    both = (s["music_file"].notna() & (s["music_file"] == s["voice_file"])).to_numpy()
    for role in ROLES:
        present = (s[f"{role}_take"] > 0).to_numpy()
        filedur = s[f"{role}__duration_s_decoded_chain"].fillna(s[f"{role}__duration_s"]).fillna(0.0).to_numpy()
        take = np.minimum(rng.uniform(take_lo, take_hi, len(s)), filedur)
        span = np.where(s["sequential"], s["duration_s"] / s["n_components"].clip(lower=1), s["duration_s"])
        take = np.minimum(take, span)
        offset = rng.uniform(0.0, 1.0, len(s)) * np.maximum(0.0, filedur - take)
        s.loc[present, f"{role}_take"] = take[present]
        s.loc[present, f"{role}_offset"] = offset[present]
    for k in ("take", "offset"):
        s.loc[both, f"music_{k}"] = s.loc[both, f"voice_{k}"]
    return s


def _tile_all(s: pd.DataFrame) -> pd.DataFrame:
    """Every present component tiled to its span (random-offset crops of the
    same file, sigmoid joins). Coverage is 1 by construction; the join count
    is a feature and must depend on the timeline only."""
    s = s.copy()
    for role in ROLES:
        present = s[f"{role}_take"] > 0
        span = np.where(s["sequential"], s["duration_s"] / s["n_components"].clip(lower=1), s["duration_s"])
        joins = np.ceil(span / s[f"{role}_take"].where(present, np.inf)) - 1
        s[f"{role}_joins"] = joins.where(present, 0.0).clip(lower=0)
        s.loc[present, f"{role}_take"] = span[present.to_numpy()]
        s[f"{role}_tiled"] = present
    return s


def _tile_music(s: pd.DataFrame) -> pd.DataFrame:
    """P3: the music component is tiled to the sample timeline, so its coverage
    is 1 and the number of joins becomes a feature."""
    s = s.copy()
    present = s["music_take"] > 0
    joins = np.ceil(s["duration_s"] / s["music_take"].where(present, np.inf)) - 1
    s["music_joins"] = joins.where(present, 0.0).clip(lower=0)
    s.loc[present, "music_take"] = s.loc[present, "duration_s"]
    return s


def policies(stream: pd.DataFrame, tag: str = "") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 1)
    out = {f"{tag}P0_baseline": stream}
    if not tag:
        out["P1_take_U4-9.5"] = _redraw_music(stream, rng, 4.0, 9.5)
        out["P1x_take_U4-30_(why_t_hi<10)"] = _redraw_music(stream, rng, 4.0, 30.0)
        out["P1+P2_cover_timeline"] = _redraw_music(stream, rng, 4.0, 9.5, timeline="cover")
        out["P0+P3_tile_(why_P1_first)"] = _tile_music(stream)
        out["P5_all_take4_tile"] = _tile_all(_redraw_all(stream, rng, 4.0, 4.0))
        out["P8_all_take_U2-6_tile"] = _tile_all(_redraw_all(stream, rng, 2.0, 6.0))
    out[f"{tag}P7_all_take_U3-8_tile"] = _tile_all(_redraw_all(stream, rng, 3.0, 8.0))
    return out


def silence_policies(s: pd.DataFrame, f0: pd.DataFrame) -> pd.DataFrame:
    """§3.4: the effective leading silence of voice samples under (p, L) draws,
    under A6b's three controls."""
    rng = np.random.default_rng(SEED + 2)
    voice = s["voice_take"] > 0
    y = pd.to_numeric(s["voice_fake"], errors="coerce")
    src = s["voice__source_name"]
    controls = {
        "corpus-wide": voice,
        "within_cfad": voice & src.isin(["cfad-real", "cfad-fake"]),
        "ljspeech_vs_wavefake": voice & src.isin(["ljspeech", "wavefake"]),
    }
    rows = []
    for p in (0.0, 0.2, 0.5, 1.0, "trim+redraw"):
        for L in ((0.0,) if p == 0.0 else (0.5, 1.0, 2.0, 3.0, 4.0)):
            if p == "trim+redraw":
                # P-A2's other arm: strip the file's own lead, then draw one.
                # Symmetric by construction; costs the vocoder's silence-floor cue.
                eff = pd.Series(rng.uniform(0.0, L, len(s)), index=s.index)
            else:
                on = rng.random(len(s)) < p
                drawn = pd.Series(np.where(on, rng.uniform(0.0, L, len(s)), 0.0), index=s.index)
                eff = f0["voice_lead_silence_s"] + drawn
            for name, mask in controls.items():
                m = mask & y.notna()
                yy = y[m].astype(int).to_numpy()
                if len(np.unique(yy)) < 2:
                    continue
                rows.append({"p": str(p), "lead_max_s": L, "control": name, "n": int(m.sum()),
                             "auc_effective_lead": univariate_auc(eff[m].to_numpy(), yy)})
    out = pd.DataFrame(rows)
    out["gate"] = np.where(out["auc_effective_lead"] < GATE, "pass", "fail")
    return out


def level_policies(s: pd.DataFrame, f0: pd.DataFrame) -> pd.DataFrame:
    """§3.2: the residual level cue under each candidate normalisation, as a
    univariate AUC per head under the archive-grouped logistic audit."""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for role, head in (("music", "music_fake"), ("voice", "voice_fake")):
        present = s[f"{role}_take"] > 0
        y = pd.to_numeric(s[head], errors="coerce")
        m = present & y.notna()
        yy = y[m].astype(int).to_numpy()
        groups = s.loc[m, f"{role}__source_name"].to_numpy()
        cands = {
            "none": f0.loc[m, f"{role}_rms_dbfs"],
            "peak_normalise (residual = -crest)": -f0.loc[m, f"{role}_crest_db"],
            "rms_normalise (residual = crest)": f0.loc[m, f"{role}_crest_db"],
            "none + gain jitter U(-6,6)": f0.loc[m, f"{role}_rms_dbfs"] + rng.uniform(-6, 6, int(m.sum())),
            "none + gain jitter U(-12,12)": f0.loc[m, f"{role}_rms_dbfs"] + rng.uniform(-12, 12, int(m.sum())),
        }
        for name, col in cands.items():
            x = col.fillna(col.median()).to_numpy(dtype=float)[:, None]
            rows.append({"head": head, "candidate": name, "n": int(m.sum()),
                         "auc_univariate": univariate_auc(x[:, 0], yy),
                         "auc_source_grouped": _cv_auc(x, yy, groups, grouped=True)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_eda_config(ROOT / "configs" / "eda.yaml")
    signal = load_signal(cfg)
    envelope = pd.read_parquet(cfg.out / "_shared" / "envelope.parquet")
    manifest = build_manifest(signal)
    manifest.to_parquet(OUT / "manifest_stier.parquet", index=False)
    run = load_run_config(ROOT / "configs" / "run_default.yaml")
    print(f"manifest: {len(manifest)} rows; component {int((manifest.row_kind=='component').sum())}, "
          f"whole_file {int((manifest.row_kind=='whole_file').sum())}")

    from training.audit import run_audit
    report = run_audit(Sampler(manifest, run.sampler, slice_="train"), n=N_SPECS,
                       seed=SEED, manifest=manifest)
    (OUT / "run_audit_baseline.txt").write_text(str(report))
    print("== training.audit.run_audit on the baseline stream\n", report)

    stream = draw_stream(manifest, run.sampler, N_SPECS, SEED)
    print(f"drew {len(stream)} specs; sampler dropped {stream.attrs['n_dropped_short']} rows under 4 s")
    stream = attach_measurements(stream, signal, envelope)

    audits, summaries = [], {}
    worklist = pd.read_parquet(cfg.out / "_shared" / "reassignment_worklist.parquet")
    import dataclasses
    variants = {
        "": (manifest, run.sampler),
        "M1reassigned_": (apply_reassignment(manifest, worklist), run.sampler),
        "F8strict_": (manifest, dataclasses.replace(run.sampler, f8=1.0)),
    }
    for tag, (man, scfg) in variants.items():
        st = stream if not tag else attach_measurements(draw_stream(man, scfg, N_SPECS, SEED), signal, envelope)
        for name, s in policies(st, tag).items():
            variants_of = {name: (s, None)}
            if name.endswith("P7_all_take_U3-8_tile"):
                lead = pd.Series(np.random.default_rng(SEED + 5).uniform(0.0, 3.0, len(s)), index=s.index)
                variants_of[name + "+lead_U0-3"] = (s, lead)
            for pname, (sp, lead) in variants_of.items():
                f = effective_features(sp, drawn_lead=lead)
                for role in ROLES:
                    if f"{role}_joins" in sp.columns:
                        f[f"{role}_joins"] = sp[f"{role}_joins"]
                sp.assign(**{f"eff__{c}": f[c] for c in f.columns}).to_parquet(
                    OUT / f"stream_{pname.split('_(')[0]}.parquet", index=False)
                a = audit(sp, f, pname)
                audits.append(a)
                m = sp["music_take"] > 0
                by_pool = sp[m].groupby("music__pool").apply(
                    lambda g: pd.Series({"p_offset0": float((g["music_offset"] < HOP_S).mean()),
                                         "take_median": float(g["music_take"].median()),
                                         "uncovered_mean": float(f.loc[g.index, "uncovered_frac"].mean()),
                                         "n": len(g)}))
                summaries[pname] = by_pool.to_dict(orient="index")
                print(f"\n== {pname}\n{by_pool.round(3).to_string()}")
                print(a[["head", "family", "stratum", "n", "auc", "auc_source_grouped", "gate", "top_features"]]
                      .to_string(index=False, max_colwidth=80))
        if tag:
            from training.audit import run_audit
            rep = run_audit(Sampler(man, scfg, slice_="train"), n=N_SPECS, seed=SEED, manifest=man)
            (OUT / f"run_audit_{tag.strip('_')}.txt").write_text(str(rep))
            print(f"== training.audit.run_audit on {tag}\n", rep)
    pd.concat(audits).to_parquet(OUT / "stream_audit.parquet", index=False)

    f0 = effective_features(stream)
    sil = silence_policies(stream, f0)
    sil.to_parquet(OUT / "silence_policies.parquet", index=False)
    print("\n== silence policies\n", sil.pivot_table(index=["p", "lead_max_s"], columns="control",
                                                  values="auc_effective_lead").round(3).to_string())
    lev = level_policies(stream, f0)
    lev.to_parquet(OUT / "level_policies.parquet", index=False)
    print("\n== level policies\n", lev.round(3).to_string(index=False))

    (OUT / "summary.json").write_text(json.dumps({
        "n_specs": N_SPECS, "seed": SEED, "manifest_rows": len(manifest),
        "n_dropped_short": stream.attrs["n_dropped_short"], "by_pool": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
