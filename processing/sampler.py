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
* **The floor is bound to the take (D-5 revised)**: ``floor - 2 * margin``
  must hold the longest take. 4.0 s with a 2-3 s take keeps 80 % of pool B's
  hours; the spec's 2 s floor is reachable only with a 1 s take.
* **Whole-file rows go through the same rule (D-3, D-16).** The training
  sampler uses a whole-file row as-is from offset 0, capped at the row's
  length -- which made the row's length a duration cue and its onset a
  silence cue. Here a whole-file draw is DOSS-weighted, the timeline is the
  timeline, and the row is cropped inside and tiled over the span like any
  component. Under ``f8 = 1`` the branch is drawn only in the ``f8`` sweep.
* **Lead/tail silence on every sample (D-6)** at the measured values -- drawn
  before the branch, so the whole-file branch carries the same lead.
* **The noise layer (DRAW-5).** With ``p_noise_layer`` a pool-E row goes
  under the composite of ANY cell at ``U(10, 30)`` dB SNR, tiled like a
  component; rows flagged ``noise_has_speech`` never go under a
  ``voice_present = 0`` cell. The layer decision and its SNR are drawn before
  the cell; the SNR is resolved at render (``ComponentDraw.snr_db``).
* **The augment and test-chain draws exist (DRAW-6, DRAW-7).** The training
  sampler never fills ``spec.transforms`` or ``spec.normalize``; here both are
  drawn per sample from the config's menus, before the cell.
* **One take per sample, never capped by a file (D-21, measured in steps 3
  and 6).** A take capped by a short file means more tiles, and pool B is
  short (voice_fake 0.65); a take drawn per role makes the larger of two
  join counts read as "two components" (voice_present 0.68). So the take is
  drawn once, shared by every role, and ``take_hi <= floor - 2 * margin`` --
  the join count is a function of (span, take) alone.

Everything else -- the cell mix, ``composed_fractions``, the gain ratio, the
sequential structure -- is the training sampler's, imported not copied.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from typing import Any

from processing.config import DrawConfig
from training.manifest import POOL_IS_FAKE, ROLE_POOLS
from training.registries import AUGMENT
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

        # DRAW-5: the pool-E rows a layer may draw from. `noise_has_speech`
        # (D-14 `restrict_noise`) is an optional manifest column; absent, no
        # row is restricted.
        noise = comp[comp.pool == "E"]
        flagged = (noise["noise_has_speech"].fillna(False).astype(bool)
                   if "noise_has_speech" in noise.columns
                   else pd.Series(False, index=noise.index))
        self._layer_rows = {
            True: noise, False: noise[~flagged]}          # keyed by voice_present
        self._layer_weights = {k: self._doss_weights(v.domain_key) if len(v) else np.empty(0)
                               for k, v in self._layer_rows.items()}
        self.n_noise_restricted = int(flagged.sum())

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

    # -- DRAW-6 / DRAW-7 ----------------------------------------------------- #

    def _draw_transforms(self, rng: np.random.Generator) -> tuple[tuple[str, dict[str, Any]], ...]:
        """DRAW-6: with probability ``p`` per entry, ``(name, params)`` with every
        drawable ``*_range`` drawn to its scalar. Drawn BEFORE the cell."""
        out: list[tuple[str, dict[str, Any]]] = []
        for entry in self.cfg.augments:
            take = rng.random() < entry.p
            params: dict[str, Any] = {}
            accepted = AUGMENT.params_of(entry.name)
            for key, value in entry.params.items():
                scalar = key[:-len("_range")] if key.endswith("_range") else None
                if scalar is not None and scalar in accepted:
                    # Critical: the draw happens whether or not the entry is
                    # taken, so the later stream does not shift with `p`.
                    drawn = float(rng.uniform(*value))
                    params[scalar] = drawn
                else:
                    params[key] = value
            if take:
                out.append((entry.name, params))
        return tuple(out)

    def _draw_normalize(self, rng: np.random.Generator) -> dict[str, Any]:
        """DRAW-7: one test chain per sample from the menu, keys a subset of
        ``training.render.NORMALIZE_KEYS``. Drawn BEFORE the cell."""
        menu = self.cfg.normalize_menu
        if menu is None:
            return {}

        def pick(d: dict[str, float]) -> str:
            keys = sorted(d)
            return str(keys[int(rng.choice(len(keys), p=[d[k] for k in keys]))])

        out: dict[str, Any] = {}
        container = pick(menu.container)
        base, _, rate = container.partition("_")
        out["container"] = base
        if rate:
            out["bitrate"] = int(rate)
        out["channels"] = pick(menu.channels)
        telephone = pick(menu.telephone)
        if telephone != "none":
            out["telephone_hz"] = int(menu.telephone_hz)
            if telephone != "plain":
                out["companding"] = telephone
        return out

    # -- DRAW-5 ---------------------------------------------------------------- #

    def _draw_layer(self, rng: np.random.Generator, voice_present: bool,
                    exclude: str | None, lead: float, span: float, take: float,
                    snr_db: float) -> list[ComponentDraw]:
        """One pool-E row, tiled over the span at ``snr_db``. Rows flagged
        ``noise_has_speech`` are excluded under a ``voice_present = 0`` cell
        (they would mislabel a music-only composite); ``exclude`` keeps a
        cell-9 sample from layering a file under itself."""
        rows, w = self._layer_rows[voice_present], self._layer_weights[voice_present]
        if exclude is not None and len(rows):
            keep = (rows["file_id"] != exclude).to_numpy()
            rows, w = rows[keep], w[keep]
            w = w / w.sum() if w.sum() > 0 else w
        if not len(rows):
            return []
        row = rows.iloc[int(rng.choice(len(rows), p=w))]
        tiles = self._tiles(rng, str(row.file_id), "noise", float(row.duration_s),
                            lead, span, 0.0, take)
        return [replace(t, snr_db=snr_db) for t in tiles]

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
               gain_db: float, take: float) -> list[ComponentDraw]:
        """DRAW-3: the sample's take, tiled over ``[slot_start, slot_start + span)``.

        ::

            take     = min(take, file - 2 * margin, span)     # never binds on the file
            n        = ceil(span / take);  tile = span / n     # in (take/2, take]
            offset_i ~ U(margin, file - margin - tile)         # per tile

        Critical: the file's edges are never inside a tile. ``offset >= margin``
        and ``offset + tile <= file - margin`` hold for every tile by
        construction, so the onset of the file -- pool D's silence, the
        vocoder's first frame, CompSpoof's 4.00 s clip boundary -- is exposed
        in 0 % of draws rather than the training sampler's 87 %.

        Critical: ``take`` is drawn ONCE per sample and shared by every role
        (D-21). Drawn per role, the larger of two independent join counts
        exceeds one alone, so the join count read as "two components are
        present" (voice_present 0.68 on the S-tier stream). And the cap on the
        file never binds -- ``DrawConfig`` asserts ``take_hi <= floor - 2 *
        margin`` -- so the join count is a function of ``(span, take)`` only,
        for every role and row kind.

        Caveat: tiles are equal-length ``span / n`` rather than ``take`` with a
        short remainder, so no tile is shorter than half a take.
        """
        cfg = self.cfg
        margin = cfg.edge_margin_s
        usable = file_duration - 2.0 * margin           # >= take_hi by the floor
        take = min(take, usable, span)
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

        # DRAW-6 / DRAW-7: the augment and test-chain draws, BEFORE the cell,
        # so nothing about the labels can reach them (docs/processing/03).
        transforms = self._draw_transforms(rng)
        normalize = self._draw_normalize(rng)
        # DRAW-5: whether a noise layer is added, and at what SNR -- BEFORE the
        # cell (R2: per sample, never per cell); the row is drawn after it.
        layer = rng.random() < cfg.p_noise_layer
        layer_snr = float(rng.uniform(*cfg.noise_snr_db_range))

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
        # DRAW-3: the take, once per sample, shared by every role (D-21).
        take = float(rng.uniform(*cfg.take_range_s))

        if not composed:
            # One row used as-is, DOSS-weighted (D-16), tiled over the span
            # under the same rule as a component (D-3, D-4). Its role names the
            # component the cell says is present ("noise" for cell 9); the
            # labels come from the cell, not the role.
            rows = self._whole_by_cell[cell]
            row = rows.iloc[int(rng.choice(len(rows), p=self._whole_weights[cell]))]
            role = wanted[0][0]
            tiles = self._tiles(rng, str(row.file_id), role, float(row.duration_s),
                                lead, span, 0.0, take)
            if layer:
                tiles += self._draw_layer(rng, bool(vp), str(row.file_id), lead, span, take,
                                          layer_snr)
            return SampleSpec(
                sample_id=sample_id, epoch=epoch, seed=seed,
                scheme_version=cfg.scheme_version,
                duration_s=duration, cell=cell, render_mode="whole_file",
                structure="overlap", components=tuple(tiles),
                crossfade_ms=crossfade_ms, transforms=transforms, normalize=normalize,
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
                                     start, slot, gain, take))
        if layer:
            own = draws[0].file_id if wanted[0][0] == "noise" else None
            draws.extend(self._draw_layer(rng, bool(vp), own, lead, span, take, layer_snr))

        return SampleSpec(
            sample_id=sample_id, epoch=epoch, seed=seed,
            scheme_version=cfg.scheme_version,
            duration_s=duration, cell=cell, render_mode="composed",
            structure="sequential" if sequential else "overlap",
            components=tuple(draws), crossfade_ms=crossfade_ms,
            transforms=transforms, normalize=normalize,
        )

    def epoch_specs(self, n: int, epoch: int = 0, seed: int = 0):
        """An epoch is a fixed count of drawn specs (docs/pipelines/02 §6)."""
        for i in range(n):
            yield self.sample_spec(i, epoch=epoch, seed=seed)
