"""Drawing a ``SampleSpec`` under the strategy -- DRAW-1 to DRAW-4 of
docs/processing/03 §3. Pure: reads the manifest, touches no audio.

What differs from ``training.sampler.Sampler``, and why (02 §2-§3):

* **The take/offset/tile rule (D-3, D-4).** The training sampler's
  ``take = min(span, file)`` exposes a file's onset whenever the file is
  shorter than the timeline, and pool D is 10 s files against a 4-60 s
  timeline: 87 % onset exposure, 53 % silence, and a music-head draw AUC of
  0.995 from the draw alone. Here every component role takes ``U(3, 8)`` s from
  an offset drawn so the take lies *strictly inside* the file, and the take is
  tiled to its span with independent offsets per tile. Music-only draw AUC
  0.993 -> 0.497; onset exposure < 1 %.
* **A 2 s floor (D-5)**, not 4 s: under the tile rule a 2 s file is usable, and
  the 4 s floor cost 22.5 % of pool B's hours.
* **Whole-file rows go through the same rule (D-3, D-16).** The training
  sampler uses a whole-file row as-is from offset 0, capped at the row's
  length -- which made the row's length a duration cue and its onset a
  silence cue. Here a whole-file draw is DOSS-weighted, the timeline is the
  timeline, and the row is cropped inside and tiled over the span like any
  component. Under ``f8 = 1`` the branch is drawn only in the ``f8`` sweep.
* **Lead/tail silence on every sample (D-6)** at the measured values -- drawn
  before the branch, so the whole-file branch carries the same lead.
* **Duration-matched file weights (D-21, measured in step 3).** The take is
  capped by the file, so a short file means more tiles; pool B is short and
  the tile count became a voice-fake cue (I1b 0.65). Within each role the two
  sides are re-weighted to the same usable-duration histogram
  (``_match_durations``).

Everything else -- the cell mix, ``composed_fractions``, the gain ratio, the
sequential structure -- is the training sampler's, imported not copied.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from processing.config import DrawConfig
from training.manifest import POOL_IS_FAKE, ROLE_POOLS
from training.spec import CELL_TABLE, ComponentDraw, SampleSpec, spec_rng

__all__ = ["Sampler"]


class Sampler:
    """Draws specs from one slice of one fold. Same surface as
    ``training.sampler.Sampler`` (``sample_spec``, ``epoch_specs``, ``slice_``,
    ``fold``), so ``training.audit.run_audit`` accepts it unchanged."""

    def __init__(self, manifest: pd.DataFrame, cfg: DrawConfig | None = None,
                 slice_: str = "train", fold: int | None = None):
        self.cfg = cfg or DrawConfig()
        df = manifest[manifest["slice"] == slice_]
        if fold is not None:
            df = df[df["fold"] == fold]
        if df.empty:
            raise ValueError(f"no manifest rows for slice={slice_!r} fold={fold!r}")
        self.slice_, self.fold = slice_, fold

        # D-5: the component floor. Below it a row has no usable interior once
        # the edge margin is taken off both ends (`DrawConfig` asserts
        # `floor > 2 * margin`).
        comp = df[df.row_kind == "component"]
        usable = comp[comp.duration_s >= self.cfg.component_floor_s]
        self.n_dropped_short = int(len(comp) - len(usable))
        if usable.empty:
            raise ValueError(
                f"every component row in slice={slice_!r} fold={fold!r} is shorter "
                f"than component_floor_s={self.cfg.component_floor_s}s")
        comp = usable

        self._by_role_fake: dict[tuple[str, bool], pd.DataFrame] = {}
        self._weights: dict[tuple[str, bool], np.ndarray] = {}
        for role, pools in ROLE_POOLS.items():
            sub = comp[comp.pool.isin(pools)]
            for fake in (False, True):
                rows = sub[sub.pool.map(POOL_IS_FAKE) == fake]
                self._by_role_fake[(role, fake)] = rows
                self._weights[(role, fake)] = self._doss_weights(
                    rows.domain_key if not rows.empty else pd.Series(dtype=object))
            self._match_durations(role)

        # Whole-file rows go through the same take/offset/tile rule (D-3:
        # "every role incl. whole-file"), so the same floor applies.
        whole = df[df.row_kind == "whole_file"]
        kept = whole[whole.duration_s >= self.cfg.component_floor_s]
        self.n_dropped_short_whole = int(len(whole) - len(kept))
        self._whole_by_cell: dict[int, pd.DataFrame] = {}
        self._whole_weights: dict[int, np.ndarray] = {}
        for c, g in kept.groupby("cell"):
            self._whole_by_cell[int(c)] = g
            # D-16: the DOSS key of a whole-file row is its generator domain
            # (`domain_key`, falling back to `artifact_family`); real rows
            # share the one uncapped bucket, as real component rows do.
            key = g.domain_key.where(g.domain_key.notna(), g.artifact_family)
            self._whole_weights[int(c)] = self._doss_weights(key)

    # -- DOSS ---------------------------------------------------------------- #

    def _doss_weights(self, domain: pd.Series) -> np.ndarray:
        """``w(file) = min(count(domain), cap) / count(domain)``, normalised."""
        if domain.empty:
            return np.empty(0)
        domain = domain.fillna("__real__")
        counts = domain.map(domain.value_counts())
        w = np.minimum(counts, self.cfg.domain_cap) / counts
        return (w / w.sum()).to_numpy(dtype=float)

    # -- D-21: duration matching --------------------------------------------- #

    def _usable_bins(self, rows: pd.DataFrame) -> np.ndarray:
        edges = np.asarray(self.cfg.duration_match_edges_s, dtype=float)
        usable = rows.duration_s.to_numpy(dtype=float) - 2.0 * self.cfg.edge_margin_s
        return np.searchsorted(edges, usable, side="right")

    def _match_durations(self, role: str) -> None:
        """Re-balance the two sides of ``role`` so they draw the same histogram
        of usable duration (``DrawConfig.duration_match_edges_s``).

        Critical: the take is capped by the file, so the tile count -- a join
        count the model can hear -- follows the file's length, and the length
        distribution differs between a role's real and fake pools. Matching the
        two histograms makes ``P(n_tiles | label)`` flat by construction while
        keeping every row drawable. A bin only one side populates cannot be
        matched and is given zero mass on both sides.
        """
        if self.cfg.duration_match_edges_s is None:
            return
        real, fake = self._by_role_fake[(role, False)], self._by_role_fake[(role, True)]
        if real.empty or fake.empty:
            return
        n_bins = len(self.cfg.duration_match_edges_s) + 1
        bins = {side: self._usable_bins(rows) for side, rows in (("r", real), ("f", fake))}
        mass = {side: np.bincount(bins[side], weights=self._weights[(role, side == "f")],
                                  minlength=n_bins) for side in bins}
        target = np.where((mass["r"] > 0) & (mass["f"] > 0), 0.5 * (mass["r"] + mass["f"]), 0.0)
        if target.sum() <= 0:
            raise ValueError(
                f"{role}: the real and fake pools share no usable-duration bin under "
                f"duration_match_edges_s={self.cfg.duration_match_edges_s}; nothing "
                f"can be matched")
        target /= target.sum()
        for side, fake_flag in (("r", False), ("f", True)):
            ratio = np.divide(target, mass[side], out=np.zeros(n_bins), where=mass[side] > 0)
            w = self._weights[(role, fake_flag)] * ratio[bins[side]]
            self._weights[(role, fake_flag)] = w / w.sum()

    # -- drawing ------------------------------------------------------------- #

    def _draw_component(self, rng: np.random.Generator, role: str, fake: bool) -> pd.Series:
        rows = self._by_role_fake[(role, fake)]
        if rows.empty:
            raise ValueError(
                f"slice={self.slice_!r} fold={self.fold!r} has no "
                f"{'fake' if fake else 'real'} {role} components")
        return rows.iloc[int(rng.choice(len(rows), p=self._weights[(role, fake)]))]

    def _gain_db(self, rng: np.random.Generator) -> float:
        lo, hi = self.cfg.gain_db_range
        return float(np.clip(rng.normal(self.cfg.gain_db_mean, self.cfg.gain_db_sigma), lo, hi))

    def _tiles(self, rng: np.random.Generator, file_id: str, role: str,
               file_duration: float, slot_start: float, span: float,
               gain_db: float) -> list[ComponentDraw]:
        """DRAW-3: one take, tiled over ``[slot_start, slot_start + span)``.

        ::

            take   ~ U(take_lo, take_hi)
            take    = min(take, file - 2 * margin, span)
            n       = ceil(span / take);  tile = span / n        # in (take/2, take]
            offset_i ~ U(margin, file - margin - tile)          # per tile

        Critical: the file's edges are never inside a tile. ``offset >= margin``
        and ``offset + tile <= file - margin`` hold for every tile by
        construction, so the onset of the file -- pool D's silence, the
        vocoder's first frame, CompSpoof's 4.00 s clip boundary -- is exposed
        in 0 % of draws rather than the training sampler's 87 %.

        Caveat: tiles are equal-length ``span / n`` rather than ``take`` with a
        short remainder, so no tile is shorter than half a take and the join
        count is still a function of ``(span, take)`` only.
        """
        cfg = self.cfg
        margin = cfg.edge_margin_s
        usable = file_duration - 2.0 * margin           # > 0 by the floor
        take = min(float(rng.uniform(*cfg.take_range_s)), usable, span)
        n = max(1, math.ceil(span / take - 1e-9))
        tile = span / n
        high = file_duration - margin - tile              # >= margin
        return [
            ComponentDraw(
                file_id=file_id, role=role,
                source_offset_s=float(rng.uniform(margin, high)),
                duration_s=tile, target_start_s=slot_start + i * tile,
                gain_db=gain_db)
            for i in range(n)
        ]

    def sample_spec(self, sample_id: int, epoch: int = 0, seed: int = 0) -> SampleSpec:
        rng = spec_rng(sample_id, epoch, seed)
        cfg = self.cfg

        # DRAW-1: the timeline, drawn first -- and it IS the timeline for every
        # branch. The training sampler caps a whole-file draw at the row's
        # length, which made pool D's 10 s files a duration cue.
        lo, hi = cfg.duration_range
        duration = float(rng.uniform(lo, hi))

        # DRAW-2: cell and composition. Labels are fixed here.
        cell = cfg.cell_mix.draw(rng)
        composed = bool(rng.random() < cfg.f[cell])
        if cell in (6, 7):
            composed = True                       # cannot be scraped
        if not composed and cell not in self._whole_by_cell:
            composed = True                       # no whole-file row available

        vp, mp, vf, mf = CELL_TABLE[cell]
        wanted: list[tuple[str, bool]] = []
        if vp:
            wanted.append(("voice", bool(vf)))
        if mp:
            wanted.append(("music", bool(mf)))
        if not wanted:
            wanted.append(("noise", False))

        # DRAW-4: structure, the taper, lead/tail. Drawn for EVERY sample,
        # before any file is chosen, so none can depend on which file was
        # drawn -- and the whole-file branch gets the same lead the composed
        # branch does (a lead only composed samples carried would be a
        # composedness cue).
        sequential = composed and len(wanted) > 1 and rng.random() < cfg.sequential_prob
        crossfade_ms = float(rng.uniform(*cfg.crossfade_ms_range))
        lead = float(rng.uniform(0.0, cfg.silence_lead_s)) if cfg.silence_lead_s else 0.0
        tail = float(rng.uniform(0.0, cfg.silence_tail_s)) if cfg.silence_tail_s else 0.0
        if lead + tail > 0.5 * duration:          # never silence half the sample
            scale = 0.5 * duration / (lead + tail)
            lead, tail = lead * scale, tail * scale
        span = duration - lead - tail

        if not composed:
            # One row used as-is, DOSS-weighted (D-16), tiled over the span
            # under the same rule as a component (D-3, D-4). Its role names the
            # component the cell says is present ("noise" for cell 9); the
            # labels come from the cell, not the role.
            rows = self._whole_by_cell[cell]
            row = rows.iloc[int(rng.choice(len(rows), p=self._whole_weights[cell]))]
            role = wanted[0][0]
            tiles = self._tiles(rng, str(row.file_id), role, float(row.duration_s),
                                lead, span, 0.0)
            return SampleSpec(
                sample_id=sample_id, epoch=epoch, seed=seed,
                scheme_version=cfg.scheme_version,
                duration_s=duration, cell=cell, render_mode="whole_file",
                structure="overlap", components=tuple(tiles),
                crossfade_ms=crossfade_ms,
            )

        is_ratio = len(wanted) > 1
        draws: list[ComponentDraw] = []
        for i, (role, fake) in enumerate(wanted):
            row = self._draw_component(rng, role, fake)
            if sequential:
                slot = span / len(wanted)
                start = lead + i * slot
            else:
                slot, start = span, lead
            gain = self._gain_db(rng) if (is_ratio and role == "voice") else 0.0
            draws.extend(self._tiles(rng, str(row.file_id), role, float(row.duration_s),
                                     start, slot, gain))

        return SampleSpec(
            sample_id=sample_id, epoch=epoch, seed=seed,
            scheme_version=cfg.scheme_version,
            duration_s=duration, cell=cell, render_mode="composed",
            structure="sequential" if sequential else "overlap",
            components=tuple(draws), crossfade_ms=crossfade_ms,
        )

    def epoch_specs(self, n: int, epoch: int = 0, seed: int = 0):
        """An epoch is a fixed count of drawn specs (docs/pipelines/02 §6)."""
        for i in range(n):
            yield self.sample_spec(i, epoch=epoch, seed=seed)
