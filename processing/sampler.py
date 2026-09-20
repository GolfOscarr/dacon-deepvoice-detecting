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
* **DOSS on whole-file rows too (D-16).** The training sampler draws them
  uniformly; the SONICS generators are 400 rows each today but will not stay
  balanced.
* **Lead/tail silence on every sample (D-6)** at the measured values.

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

        # Whole-file rows. Caveat (step 1 of the change list): until the
        # whole-file branch gets the full DRAW-3/DRAW-4 treatment (tiles, lead,
        # an uncapped timeline -- docs/processing/03 §6 step 3), a whole-file
        # draw is one inside-crop whose length IS the timeline, so the row
        # must cover the timeline floor plus both margins.
        whole = df[df.row_kind == "whole_file"]
        whole_floor = self.cfg.duration_range[0] + 2.0 * self.cfg.edge_margin_s
        kept = whole[whole.duration_s >= whole_floor]
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

        # DRAW-1: the timeline, drawn first.
        lo, hi = cfg.duration_range
        duration = float(rng.uniform(lo, hi))

        # DRAW-2: cell and composition. Labels are fixed here.
        cell = cfg.cell_mix.draw(rng)
        composed = bool(rng.random() < cfg.f[cell])
        if cell in (6, 7):
            composed = True                       # cannot be scraped
        if not composed and cell not in self._whole_by_cell:
            composed = True                       # no whole-file row available

        if not composed:
            return self._whole_file_spec(rng, sample_id, epoch, seed, cell, duration)

        vp, mp, vf, mf = CELL_TABLE[cell]
        wanted: list[tuple[str, bool]] = []
        if vp:
            wanted.append(("voice", bool(vf)))
        if mp:
            wanted.append(("music", bool(mf)))
        if not wanted:
            wanted.append(("noise", False))

        # DRAW-4: structure, lead/tail, the taper. All drawn before any
        # component is chosen, so none can depend on which file was drawn.
        sequential = len(wanted) > 1 and rng.random() < cfg.sequential_prob
        crossfade_ms = float(rng.uniform(*cfg.crossfade_ms_range))
        lead = float(rng.uniform(0.0, cfg.silence_lead_s)) if cfg.silence_lead_s else 0.0
        tail = float(rng.uniform(0.0, cfg.silence_tail_s)) if cfg.silence_tail_s else 0.0
        if lead + tail > 0.5 * duration:          # never silence half the sample
            scale = 0.5 * duration / (lead + tail)
            lead, tail = lead * scale, tail * scale
        span = duration - lead - tail
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

    def _whole_file_spec(self, rng: np.random.Generator, sample_id: int, epoch: int,
                         seed: int, cell: int, duration: float) -> SampleSpec:
        """A whole-file row used as-is: DOSS-weighted (D-16), cropped strictly
        inside the file (D-3's margin).

        Caveat: interim. ``SampleSpec`` requires a whole-file spec to be exactly
        one component, so the crop is a single take whose length is the
        timeline (capped by the row) -- no tiles, no lead. Step 3 of
        docs/processing/03 §6 replaces this with the full rule.
        """
        cfg = self.cfg
        rows = self._whole_by_cell[cell]
        row = rows.iloc[int(rng.choice(len(rows), p=self._whole_weights[cell]))]
        file_duration = float(row.duration_s)
        margin = cfg.edge_margin_s
        take = min(duration, file_duration - 2.0 * margin)
        role = ("voice" if CELL_TABLE[cell][0] else
                "music" if CELL_TABLE[cell][1] else "noise")
        return SampleSpec(
            sample_id=sample_id, epoch=epoch, seed=seed,
            scheme_version=cfg.scheme_version,
            duration_s=take, cell=cell, render_mode="whole_file",
            structure="overlap",
            components=(ComponentDraw(
                file_id=str(row.file_id), role=role,
                source_offset_s=float(rng.uniform(margin, file_duration - margin - take)),
                duration_s=take, target_start_s=0.0, gain_db=0.0),),
        )

    def epoch_specs(self, n: int, epoch: int = 0, seed: int = 0):
        """An epoch is a fixed count of drawn specs (docs/pipelines/02 §6)."""
        for i in range(n):
            yield self.sample_spec(i, epoch=epoch, seed=seed)
