"""VERIFY -- the shortcut audit over the processing pipeline's drawn stream.

Two things ``training.audit`` cannot see, and this module adds:

* **The harness's draw features** (docs/processing/02 §2-§3, ``scripts/
  strategy/stream_harness.py``): what the *rendered* sample exposes, computed
  from the spec and the manifest without decoding -- per role the take, the
  coverage, whether the file's onset or end lands inside the sample, the
  number of joins, the effective lead silence; per sample the uncovered
  fraction and the drawn lead/tail. H1 (pool D's offset 0), H2 (the
  zero-filled canvas) and the raw-lead cue are all visible here and invisible
  to ``training.audit._feature_frame``, which was built for a sampler that
  had none of them. They are audited **per head**: the fake heads inside
  their presence strata and the presence heads themselves, because a cue the
  presence head can read is a shortcut too (``first_onset_exposed`` was the
  top presence-head feature under P7).
* **Tiles are not draws.** A component placed as ``n`` tiles is one draw of
  one file; ``training.audit`` counts it ``n`` times (I3 read 0.125 on a
  stream whose collapsed value is 0.488) and reads ``n_components`` as the
  tile count. The training invariants therefore run over ``collapse_tiles``
  of each spec -- one ``ComponentDraw`` per (role, file) spanning its tiles --
  and the draw features over the tiled specs.

Gate, estimator and noise rule are ``training.audit``'s, imported: CV AUC,
floor 0.60, four standard errors under H0.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from training import audit as _ta
from training.audit import NOISE_K, SHORTCUT_AUC_GATE, AuditReport, _auc_or_none, _auc_se
from training.spec import ComponentDraw, SampleSpec

__all__ = ["DRAW_FEATURES", "HEADS", "HOP_S", "POOLED_COMPONENT_GATE", "audit_specs",
           "collapse_tiles", "draw_features", "run_audit"]

#: docs/processing/03 §7 item 2: the documented D-2 presence-pattern residual
#: on the component heads, POOLED over strata -- inside a stratum the gate is
#: the E-S2 0.60.
POOLED_COMPONENT_GATE = 0.61

#: The harness's exposure hop (docs/processing/02 §2): an onset is "exposed"
#: when the take starts within one hop of the file's first sample, an end when
#: it reaches within one hop of the file's last.
HOP_S = 0.05

ROLES = ("voice", "music", "noise")

#: `head -> (label attribute, which specs are in the head's population)`.
#: A fake head is judged inside its presence stratum -- the model is asked for
#: it only where the component is present; a presence head over every spec.
HEADS: dict[str, tuple[str, str | None]] = {
    "file_fake": ("file_fake", None),
    "voice_fake": ("voice_fake", "voice_present"),
    "music_fake": ("music_fake", "music_present"),
    "voice_present": ("voice_present", None),
    "music_present": ("music_present", None),
}


# --------------------------------------------------------------------------- #
# tiles -> draws


def collapse_tiles(spec: SampleSpec) -> SampleSpec:
    """One ``ComponentDraw`` per (role, file), spanning its tiles.

    The offset kept is the first tile's; the span is ``[first start, last
    end)``; the gain is shared by construction. Labels, transforms and the
    normalize draw are untouched.
    """
    groups: dict[tuple[str, str, float | None], list[ComponentDraw]] = {}
    for c in spec.components:
        groups.setdefault((c.role, c.file_id, c.snr_db), []).append(c)
    merged = []
    for (role, fid, snr), tiles in groups.items():
        tiles.sort(key=lambda c: c.target_start_s)
        start = tiles[0].target_start_s
        end = max(c.target_start_s + c.duration_s for c in tiles)
        merged.append(ComponentDraw(
            file_id=fid, role=role, source_offset_s=tiles[0].source_offset_s,
            duration_s=end - start, target_start_s=start, gain_db=tiles[0].gain_db,
            is_mixup_partner=tiles[0].is_mixup_partner, snr_db=snr))
    return replace(spec, components=tuple(merged))


# --------------------------------------------------------------------------- #
# the draw features


DRAW_FEATURES = ("take_s", "coverage", "onset_exposed", "end_exposed", "joins",
                 "lead_silence_s", "tail_silence_s")

#: Optional manifest columns: a file's own leading / trailing silence
#: (``eda.extract.timing``). When present, the EFFECTIVE lead a sample carries
#: is the file's lead where the onset is exposed, plus the drawn lead.
_LEAD_COL, _TAIL_COL = "lead_silence_s", "tail_silence_s"


def draw_features(specs: Sequence[SampleSpec], manifest: pd.DataFrame) -> pd.DataFrame:
    """The harness's ``effective_features``, from specs and the manifest alone."""
    by_id = manifest.set_index("file_id")
    dur = by_id["duration_s"].astype(float).to_dict()
    lead_file = by_id[_LEAD_COL].astype(float).to_dict() if _LEAD_COL in by_id else {}
    tail_file = by_id[_TAIL_COL].astype(float).to_dict() if _TAIL_COL in by_id else {}

    rows: list[dict[str, float]] = []
    for s in specs:
        f: dict[str, float] = {
            "sample_duration_s": s.duration_s,
            "composed": float(s.render_mode == "composed"),
            "sequential": float(s.structure == "sequential"),
        }
        own = [c for c in s.components if c.snr_db is None]
        layer = [c for c in s.components if c.snr_db is not None]
        f["noise_layer"] = float(bool(layer))
        f["noise_layer_snr_db"] = float(layer[0].snr_db) if layer else 0.0
        f["noise_layer_joins"] = float(max(0, len(layer) - 1))
        drawn_lead = min(c.target_start_s for c in own)
        drawn_tail = s.duration_s - max(c.target_start_s + c.duration_s for c in own)
        f["drawn_lead_s"], f["drawn_tail_s"] = drawn_lead, drawn_tail
        covered = np.zeros(int(np.ceil(s.duration_s / HOP_S)) + 1, dtype=bool)
        groups: dict[str, list[ComponentDraw]] = {}
        for c in own:
            groups.setdefault(c.role, []).append(c)
        # a whole-file row is every role its cell says is present
        roles_of = {role: groups.get(role, []) for role in ROLES}
        if s.render_mode == "whole_file":
            only = next(iter(groups.values()))
            roles_of = {"voice": only if s.voice_present else [],
                        "music": only if s.music_present else [],
                        "noise": only if not (s.voice_present or s.music_present) else []}
        for role in ROLES:
            tiles = sorted(roles_of[role], key=lambda c: c.target_start_s)
            if not tiles:
                for name in DRAW_FEATURES:
                    f[f"{role}_{name}"] = 0.0
                continue
            span = sum(c.duration_s for c in tiles)
            fdur = dur[tiles[0].file_id]
            onset = any(c.source_offset_s < HOP_S for c in tiles)
            end = any(c.source_offset_s + c.duration_s >= fdur - HOP_S for c in tiles)
            f[f"{role}_take_s"] = span
            f[f"{role}_coverage"] = span / s.duration_s
            f[f"{role}_onset_exposed"] = float(onset)
            f[f"{role}_end_exposed"] = float(end)
            f[f"{role}_joins"] = float(len(tiles) - 1)
            f[f"{role}_lead_silence_s"] = (
                lead_file.get(tiles[0].file_id, 0.0) * float(onset) + drawn_lead)
            f[f"{role}_tail_silence_s"] = (
                tail_file.get(tiles[-1].file_id, 0.0) * float(end) + drawn_tail)
            for c in tiles:
                a = int(c.target_start_s / HOP_S)
                b = int(np.ceil((c.target_start_s + c.duration_s) / HOP_S))
                covered[a:b] = True
        inner = covered[:max(1, int(s.duration_s / HOP_S))]
        f["uncovered_frac"] = float(1.0 - inner.mean())
        f["first_onset_exposed"] = max(f[f"{r}_onset_exposed"] for r in ROLES)
        f["any_lead_silence_s"] = max(f[f"{r}_lead_silence_s"] for r in ROLES)
        f["max_joins"] = max(f[f"{r}_joins"] for r in ROLES)
        rows.append(f)
    return pd.DataFrame(rows)


def _head_frame(features: pd.DataFrame, head: str) -> pd.DataFrame:
    """The columns a head may read (the harness's ``feature_sets``).

    A component head reads the sample-level features plus its own role's. A
    presence head reads the sample-level features and the role-agnostic
    aggregates ONLY: a role's own columns (``voice_take_s > 0``) are the
    presence label itself, not a shortcut to it. Critical: ``n_draws`` and the
    per-role columns are excluded for the same reason; ``sequential`` and
    ``composed`` stay, and what they carry is the documented presence-pattern
    residual of the cell mix (02 §4.3, D-2 OPEN).
    """
    common = ["sample_duration_s", "composed", "sequential", "uncovered_frac",
              "drawn_lead_s", "drawn_tail_s", "max_joins",
              "noise_layer", "noise_layer_snr_db", "noise_layer_joins"]
    if head in ("voice_fake", "music_fake"):
        role = head.split("_")[0]
        cols = common + [c for c in features.columns if c.startswith(f"{role}_")]
    else:
        cols = common + ["first_onset_exposed", "any_lead_silence_s"]
    return features[cols]


# --------------------------------------------------------------------------- #
# the audit


def _draw_shortcuts(specs: Sequence[SampleSpec], manifest: pd.DataFrame
                    ) -> dict[str, tuple[bool, str]]:
    """I1c: can the draw features alone predict a head? Per head, inside the
    head's population, pooled and per stratum, worst taken, noise-aware."""
    specs = [s for s in specs if s.cell != 9]
    out: dict[str, tuple[bool, str]] = {}
    if not specs:
        out["I1c_draw_shortcut_auc"] = (True, AuditReport.SKIP + "no cells 1-8 in the stream")
        return out
    features = draw_features(specs, manifest)
    for head, (label, population) in HEADS.items():
        keep = [i for i, s in enumerate(specs)
                if population is None or getattr(s, population)]
        sub = [specs[i] for i in keep]
        y_all = np.array([int(getattr(s, label) or 0) for s in sub])
        X_all = _head_frame(features.iloc[keep].reset_index(drop=True), head)
        X_all = X_all.loc[:, X_all.nunique() > 1].to_numpy(dtype=float)
        scored: list[tuple[float, str]] = []
        counts: dict[str, tuple[int, int]] = {}
        probes = [("pooled", list(range(len(sub))))]
        if head not in ("voice_present", "music_present"):
            probes += [(st, [i for i, s in enumerate(sub) if s.stratum == st])
                       for st in ("voice-only", "music-only", "mixed")]
        for where, idx in probes:
            if len(idx) < 200 or X_all.shape[1] == 0:
                continue
            auc = _auc_or_none(X_all[idx], y_all[idx])
            if auc is not None:
                scored.append((auc, where))
                ys = y_all[idx]
                counts[where] = (int((ys == 1).sum()), int((ys == 0).sum()))
        key = f"I1c_draw_shortcut_auc_{head}"
        if not scored:
            out[key] = (True, AuditReport.SKIP + "not enough samples to estimate an AUC")
            continue
        # Critical: a component head is judged INSIDE its strata, where the
        # model is asked for it; the pooled probe on a component head mixes
        # strata whose fake rates differ under the cell mix (music-only 0.69
        # vs mixed 0.44 under the reference mix), and what it reads is the
        # documented D-2 presence-pattern residual (02 §4.3) -- allowed up to
        # 0.61 pooled by docs/processing/03 §7 item 2, never inside a stratum.
        component = head in ("voice_fake", "music_fake")
        verdicts = []
        for auc, where in scored:
            se = _auc_se(*counts[where])
            floor = POOLED_COMPONENT_GATE if (component and where == "pooled") \
                else SHORTCUT_AUC_GATE
            verdicts.append((auc < max(floor, 0.5 + NOISE_K * se), auc, where, floor, se))
        passed = all(v[0] for v in verdicts)
        _, worst, where, floor, se = min(verdicts, key=lambda v: (v[0], -v[1]))
        gate = max(floor, 0.5 + NOISE_K * se)
        detail = ", ".join(f"{w}={a:.4f}" for a, w in sorted(scored, key=lambda t: -t[0]))
        out[key] = (passed,
                    f"{'worst' if passed else 'failing'} draw-feature CV AUC = "
                    f"{worst:.4f} in {where!r} over {X_all.shape[1]} feature(s); gate < "
                    f"{gate:.4f} (floor {floor}, {NOISE_K:g} SE = {NOISE_K * se:.4f} at "
                    f"n={counts[where]}) [{detail}]"
                    + ("; the pooled component-head floor is the D-2 residual "
                       "allowance (03 §7)" if component else ""))
    return out


def _exposure(specs: Sequence[SampleSpec], manifest: pd.DataFrame) -> tuple[bool, str]:
    """I1d: the onset-exposure rate per pool -- the direct reading of H1. Every
    pool below 1 % (docs/processing/03 DRAW-3 Verify)."""
    pool = manifest.set_index("file_id")["pool"].to_dict()
    dur = manifest.set_index("file_id")["duration_s"].astype(float).to_dict()
    seen: Counter = Counter()
    early: Counter = Counter()
    for s in specs:
        for c in s.components:
            p = pool.get(c.file_id)
            if p is None or pd.isna(p):         # a whole-file row: by its cell
                p = f"cell{s.cell}"
            seen[p] += 1
            early[p] += c.source_offset_s < HOP_S or \
                c.source_offset_s + c.duration_s >= dur[c.file_id] - HOP_S
    rates = {p: early[p] / seen[p] for p in sorted(seen)}
    worst = max(rates.values()) if rates else 0.0
    return (worst < 0.01,
            "edge exposure per pool (onset within 50 ms or end within 50 ms): "
            + ", ".join(f"{p}={r:.4f}" for p, r in rates.items())
            + "; gate < 0.01 (the training draw exposes pool D at 0.87)")


def audit_specs(specs: Sequence[SampleSpec], manifest: pd.DataFrame,
                **kw: Any) -> AuditReport:
    """``training.audit.audit_specs`` over the collapsed specs, plus I1c (the
    draw features, per head) and I1d (edge exposure per pool) over the tiled
    ones. The manifest is required: the draw features read file durations."""
    collapsed = [collapse_tiles(s) for s in specs]
    base = _ta.audit_specs(collapsed, manifest=manifest, **kw)
    results = dict(base.results)
    results.update(_draw_shortcuts(specs, manifest))
    results["I1d_edge_exposure"] = _exposure(specs, manifest)
    return AuditReport(results)


def run_audit(sampler: Any, manifest: pd.DataFrame, n: int = 20_000, epoch: int = 0,
              seed: int = 0, **kw: Any) -> AuditReport:
    """Draw ``n`` specs from any sampler with ``epoch_specs`` and audit them."""
    specs = list(sampler.epoch_specs(n, epoch=epoch, seed=seed))
    return audit_specs(specs, manifest, slice_=getattr(sampler, "slice_", "train"),
                       fold=getattr(sampler, "fold", None), **kw)
