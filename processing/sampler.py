"""Drawing a ``SampleSpec`` under the strategy -- DRAW-1 to DRAW-7 of
docs/processing/03 §3, revised by docs/processing/06. Pure: reads the
manifest, touches no audio.

What differs from ``training.sampler.Sampler``, and why (02 §2-§3, 05, 06):

* **The take/offset/tile rule (D-3, D-4).** The training sampler's
  ``take = min(span, file)`` exposes a file's onset whenever the file is
  shorter than the timeline, and pool D is 10 s files against a 4-60 s
  timeline: 87 % onset exposure, 53 % silence, and a music-head draw AUC of
  0.995 from the draw alone. Here every component takes ``take`` seconds from
  an offset drawn so the take lies *strictly inside* the file, and the take is
  tiled to its span with independent offsets per tile.
* **One take per sample, never capped by a file (D-21).** A take capped by a
  short file means more tiles, and pool B is short (voice_fake 0.65); a take
  drawn per role makes the larger of two join counts read as "two components"
  (voice_present 0.68). The take is drawn once and shared by every role, and
  the join count is a function of ``(span, take)`` alone.
* **Bucket tiling (06 D5).** The tiles of one component may come from
  DIFFERENT files of the same bucket -- the speaker for voice, the whole pool
  for music and for noise (one bucket per side, symmetric by construction). A file
  is eligible for a tile iff it can hold it (``duration >= tile + xfade + 2 *
  margin``), so a 2.5-4 s file serves the tiles it can hold and no file ever
  caps the take. That is what admits the 2.5 s floor (pool B 235 -> 271 h) and
  what removes the repetition cue (05 A6): a 10 s fake-music file tiled over
  60 s repeated itself, and "repeated music" read ``music_fake`` at 0.86.
* **Overlapping tiles (06 D8).** Every tile but the last of its slot reads
  ``tile + xfade`` and the next tile starts ``tile`` later, so the renderer
  crossfades over real audio instead of dipping to silence at every joint.
  The last tile of a sequential slot extends too, into the next slot.
* **Whole-file rows go through the same rule (D-3, D-16)**, tiled from the
  one row; under ``f8 = 1`` the branch is never drawn.
* **Lead/tail silence on every sample (D-6)**, drawn before the branch.
* **The noise layer (DRAW-5)** under any cell at ``U(10, 30)`` dB, tiled like
  a component from pool E; rows flagged ``noise_has_speech`` never go under a
  ``voice_present = 0`` cell -- as a layer OR as cell 9's primary (05 A3).
* **The augment and test-chain draws (DRAW-6, DRAW-7)** per sample, before
  the cell; every drawable range is drawn to its scalar at spec time, the RIR
  choice included (05 B12), so the audit sees them.
* **Real rows are domain-capped too (06 D2)**: the publisher atom is the
  domain, under the same cap as a generator, so zeroth-korean cannot be 39 %
  of real-voice draws (05 B1).
* **The stream does not depend on the manifest's row order (05 B16)**: rows
  are sorted by ``file_id`` first.
* **A frame without folds is refused (05 A7)**: the built manifest carries no
  fold, so it must go through ``training.folds.apply_folds`` first.

Everything else -- the cell mix, ``composed_fractions``, the gain ratio, the
sequential structure -- is the training sampler's, imported not copied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from processing.config import DrawConfig
from training.manifest import POOL_IS_FAKE, ROLE_POOLS
from training.registries import AUGMENT
from training.spec import CELL_TABLE, ComponentDraw, SampleSpec, spec_rng

__all__ = ["Sampler", "bucket_keys"]

#: The slot a noise layer occupies: after every component slot.
LAYER_SLOT = 8


def bucket_keys(rows: pd.DataFrame, role: str, fake: bool) -> pd.Series:
    """The bucket a tile's file is drawn from (06 D5). Voice: the speaker,
    the publisher atom where no speaker is known -- a slot stays one voice.
    Music and noise: ONE bucket per side. Real music by artist would be 2,356
    atoms of ~3 tracks and repeat itself (05 A6); fake music by generator
    family is 600-5,500 files on the built corpus but 25 on a synthetic one,
    and whichever side's bucket exhausts first under a 40-tile slot reads
    the label through the repetition. One bucket per side is symmetric by
    construction; a music slot then mixes artists (real) or generators
    (fake) at its joints, on both sides alike."""
    if role == "voice":
        return rows["speaker_ref_id"].where(rows["speaker_ref_id"].notna(),
                                            rows["source_name"]).astype(str)
    return pd.Series("*", index=rows.index)


@dataclass
class _Pool:
    """The rows of one (role, fake), as arrays, with per-bucket duration
    order so a tile's eligible files are one ``searchsorted`` away."""

    file_id: np.ndarray             # object
    duration: np.ndarray            # float
    bucket: np.ndarray              # int codes
    weights: np.ndarray             # DOSS, sums to 1
    flagged: np.ndarray             # bool: noise_has_speech
    #: bucket code -> (row indices sorted by duration, those durations)
    members: dict[int, tuple[np.ndarray, np.ndarray]]
    #: every row sorted by duration, for the fallback
    all_sorted: tuple[np.ndarray, np.ndarray]

    def __len__(self) -> int:
        return len(self.file_id)

    def eligible(self, need_s: float, bucket: int | None, *, exclude_flagged: bool,
                 exclude: frozenset[str] | None) -> np.ndarray:
        """Row indices whose file can hold ``need_s`` seconds, inside the
        bucket (``None`` = any), minus the flagged rows and the excluded ids."""
        idx, dur = self.members[bucket] if bucket is not None else self.all_sorted
        k = int(np.searchsorted(dur, need_s - 1e-9, side="left"))
        out = idx[k:]
        if exclude_flagged and out.size:
            out = out[~self.flagged[out]]
        if exclude and out.size:
            out = out[~np.isin(self.file_id[out], list(exclude))]
        return out


class Sampler:
    """Draws specs from one slice of one fold view. Same surface as
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
        # Critical: fail closed. The built manifest marks every row `train`
        # with no fold; drawn as it is, PROBE and every VAL fold are training
        # data (05 A7). A frame reaches the sampler through
        # `training.folds.apply_folds(manifest, folds, k)`, which sets both.
        if slice_ in ("train", "val") and df["fold"].isna().any():
            raise ValueError(
                f"{int(df['fold'].isna().sum())} row(s) of slice={slice_!r} carry no fold: "
                f"materialise a fold view with training.folds.apply_folds(manifest, "
                f"folds, k) first (docs/processing/05 A7)")
        df = df.sort_values("file_id", kind="stable").reset_index(drop=True)      # 05 B16
        self.slice_, self.fold = slice_, fold

        # D-5: the component floor -- a row must hold at least the shortest
        # take (`DrawConfig` asserts `floor >= take_lo + 2 * margin`).
        comp = df[df.row_kind == "component"]
        usable = comp[comp.duration_s >= self.cfg.component_floor_s]
        self.n_dropped_short = int(len(comp) - len(usable))
        if usable.empty:
            raise ValueError(
                f"every component row in slice={slice_!r} fold={fold!r} is shorter "
                f"than component_floor_s={self.cfg.component_floor_s}s")
        comp = usable

        # docs/training/07 D-d: a language keeps its share only when this view
        # has it on BOTH voice sides. Measured on strategy-v3: CFAD's Chinese
        # fakes are sealed in PROBE with their pair atoms, so every train view
        # had Chinese as real only (20 % of real draws, 0 % of fake) -- per-side
        # renormalisation cannot fix a language one side lacks. `other` is the
        # exception: it exists for MLAAD's generator diversity, fake side only.
        self._shares = None
        if self.cfg.lang_shares is not None:
            voice = comp[comp.pool.isin(ROLE_POOLS["voice"])]
            if "lang" not in voice.columns:
                raise ValueError("draw.lang_shares is set but the manifest has no `lang` column")
            fake_side = voice.pool.map(POOL_IS_FAKE).astype(bool)
            listed = [k for k, _ in self.cfg.lang_shares]
            key = voice["lang"].fillna("other").astype(str)
            key = key.where(key.isin(listed), "other")
            both = set(key[fake_side]) & set(key[~fake_side])
            self._shares = {k: (v if (k in both or k == "other") else 0.0)
                            for k, v in self.cfg.lang_shares}
            self.lang_shares_dropped = sorted(k for k, v in self.cfg.lang_shares
                                              if v > 0 and self._shares[k] == 0)

        self._pools: dict[tuple[str, bool], _Pool] = {}
        for role, pools in ROLE_POOLS.items():
            sub = comp[comp.pool.isin(pools)]
            for fake in (False, True):
                rows = sub[sub.pool.map(POOL_IS_FAKE) == fake]
                self._pools[(role, fake)] = self._make_pool(rows, role, fake)
        self.n_noise_restricted = int(self._pools[("noise", False)].flagged.sum())
        self.n_bucket_fallbacks = 0

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
            # (`domain_key`, falling back to `artifact_family`); a real row's
            # is its publisher atom (06 D2).
            key = g.domain_key.where(g.domain_key.notna(), g.artifact_family)
            key = key.where(key.notna(), g.source_name)
            self._whole_weights[int(c)] = self._doss_weights(key)

    # -- pools and DOSS ------------------------------------------------------ #

    def _make_pool(self, rows: pd.DataFrame, role: str, fake: bool) -> _Pool:
        if role == "voice" and self.cfg.lang_shares is not None and not rows.empty:
            # share 0 removes the language outright: the bucket fallback draws
            # from the whole pool, so a zero weight alone would still leak it
            rows = rows[self._lang_of(rows).map(self._shares).gt(0)]
        if rows.empty:
            empty = np.empty(0, dtype=int)
            return _Pool(np.empty(0, dtype=object), np.empty(0), empty, np.empty(0),
                         np.empty(0, dtype=bool), {}, (empty, np.empty(0)))
        # 06 D2: a real row's domain is its publisher atom, capped like a generator
        domain = rows["domain_key"].where(rows["domain_key"].notna(), rows["source_name"])
        weights = self._doss_weights(domain)
        if role == "voice" and self.cfg.lang_shares is not None:
            weights = self._lang_balanced(rows, weights)
        codes, _ = pd.factorize(bucket_keys(rows, role, fake), sort=True)
        flagged = (rows["noise_has_speech"].fillna(False).astype(bool).to_numpy()
                   if role == "noise" and "noise_has_speech" in rows.columns
                   else np.zeros(len(rows), dtype=bool))
        duration = rows["duration_s"].to_numpy(dtype=float)
        order = np.argsort(duration, kind="stable")
        members: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for code in np.unique(codes):
            idx = order[codes[order] == code]
            members[int(code)] = (idx, duration[idx])
        return _Pool(rows["file_id"].astype(str).to_numpy(dtype=object), duration,
                     codes.astype(int), weights, flagged, members, (order, duration[order]))

    def _lang_of(self, rows: pd.DataFrame) -> pd.Series:
        """Each row's language key: its `lang`, or `other` when unlisted."""
        if "lang" not in rows.columns:
            raise ValueError("draw.lang_shares is set but the manifest has no `lang` column")
        listed = [k for k, _ in self.cfg.lang_shares]
        lang = rows["lang"].fillna("other").astype(str)
        return lang.where(lang.isin(listed), "other")

    def _lang_balanced(self, rows: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        """docs/training/07 D-d: rescale so each language holds its configured
        share of the pool, after the DOSS cap and within it."""
        shares = self._shares
        lang = self._lang_of(rows).to_numpy()
        out = np.zeros_like(weights)
        for code in np.unique(lang):
            m = lang == code
            mass = weights[m].sum()
            if mass > 0:
                out[m] = weights[m] / mass * shares.get(code, 0.0)
        if out.sum() <= 0:
            raise ValueError("lang_shares gives every language of a voice pool share 0")
        return out / out.sum()

    def _doss_weights(self, domain: pd.Series) -> np.ndarray:
        """``w(file) = min(count(domain), cap) / count(domain)``, normalised."""
        if domain.empty:
            return np.empty(0)
        domain = domain.fillna("__real__")
        counts = domain.map(domain.value_counts())
        w = np.minimum(counts, self.cfg.domain_cap) / counts
        if self.cfg.domain_weights is not None:
            # docs/training/10 F1: the longest matching prefix sets the multiplier
            mult = pd.Series(1.0, index=domain.index)
            best = pd.Series(-1, index=domain.index)
            for prefix, m in self.cfg.domain_weights:
                hit = domain.str.startswith(prefix) & (best < len(prefix))
                mult[hit], best[hit] = m, len(prefix)
            w = w * mult
            if w.sum() <= 0:
                raise ValueError("draw.domain_weights gives every domain of a pool weight 0")
        return (w / w.sum()).to_numpy(dtype=float)

    # -- DRAW-6 / DRAW-7 ----------------------------------------------------- #

    def _draw_transforms(self, rng: np.random.Generator) -> tuple[tuple[str, dict[str, Any]], ...]:
        """DRAW-6: with probability ``p`` per entry, ``(name, params)`` with every
        drawable ``*_range`` drawn to its scalar, and ``pick`` (the RIR choice)
        drawn where the augment accepts it. Drawn BEFORE the cell."""
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
                    params[scalar] = float(rng.uniform(*value))
                else:
                    params[key] = value
            if "pick" in accepted and "pick" not in params:
                params["pick"] = float(rng.random())
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

    # -- drawing ------------------------------------------------------------- #

    def _anchor(self, rng: np.random.Generator, role: str, fake: bool,
                *, exclude_flagged: bool) -> int:
        """The DOSS-weighted row that names the bucket a component draws from."""
        pool = self._pools[(role, fake)]
        if not len(pool):
            raise ValueError(
                f"slice={self.slice_!r} fold={self.fold!r} has no "
                f"{'fake' if fake else 'real'} {role} components")
        w = pool.weights
        if exclude_flagged and pool.flagged.any():
            w = np.where(pool.flagged, 0.0, w)
            if w.sum() <= 0:
                raise ValueError("every noise row is flagged noise_has_speech; cell 9 "
                                 "has nothing to draw")
            w = w / w.sum()
        return int(rng.choice(len(pool), p=w))

    def _gain_db(self, rng: np.random.Generator) -> float:
        lo, hi = self.cfg.gain_db_range
        return float(np.clip(rng.normal(self.cfg.gain_db_mean, self.cfg.gain_db_sigma), lo, hi))

    def _tiles(self, rng: np.random.Generator, pool: _Pool, bucket: int | None, role: str,
               slot: int, slot_start: float, span: float, gain_db: float, take: float,
               xfade: float, *, extend_last: bool, exclude_flagged: bool = False,
               exclude: frozenset[str] | None = None, single: int | None = None
               ) -> list[ComponentDraw]:
        """DRAW-3 under bucket tiling: the sample's take, tiled over
        ``[slot_start, slot_start + span)``, each tile from a file of the
        bucket that can hold it.

        ::

            n        = ceil(span / take);  tile = span / n          # in (take/2, take]
            dur_i    = tile + xfade   (every tile but the slot's last, unless extend_last)
            file_i   ~ Uniform{ bucket files with duration >= dur_i + 2 * margin }
            offset_i ~ U(margin, file_i - margin - dur_i)
            start_i  = slot_start + i * tile

        Critical: the file's edges are never inside a tile, the take is never
        capped by a file (D-21), and consecutive tiles overlap by ``xfade`` so
        the renderer crossfades over audio (06 D8). ``single`` pins every tile
        to one row (the whole-file branch).

        Critical: within a slot, files are drawn without replacement from
        the bucket; an exhausted bucket is reused round-robin (least-used
        file first, a fresh offset); the pool is reached only when the bucket
        has no file that can hold the tile. Measured on the way here: with
        replacement, ``n_files`` read the bucket's size and bucket sizes read
        the label (voice_fake 0.997 on the synthetic corpus); leaving an
        exhausted bucket for the pool made "speakers per slot" read it
        (0.81); a fixed 2-4 file budget made the repetition read the files'
        lengths (pool B is short). Round-robin reuse keeps a slot on one
        speaker and repeats a file only with a new offset, so what a small
        bucket leaves behind is ``unique_fraction`` slightly under 1 -- on
        the built corpus 8 % of the files of EITHER voice pool sit in buckets
        under 40 files, so the two sides match, and the audit reads it."""
        cfg = self.cfg
        margin = cfg.edge_margin_s
        take = min(take, span)                       # a short slot, never a file
        n = max(1, math.ceil(span / take - 1e-9))
        tile = span / n
        out: list[ComponentDraw] = []
        used: list[int] = []
        uses: dict[int, int] = {}
        for i in range(n):
            last = i == n - 1
            dur = tile + (0.0 if (last and not extend_last) else xfade)
            need = dur + 2.0 * margin
            if single is not None:
                fits = pool.duration[single] >= need - 1e-9
                cand = np.array([single]) if fits else np.empty(0, int)
            else:
                in_bucket = pool.eligible(need, bucket, exclude_flagged=exclude_flagged,
                                          exclude=exclude)
                if not in_bucket.size and exclude:
                    # a pool too small to keep the layer off every primary
                    # file (a test corpus of 5 noise files): the exclusion is
                    # a preference, the tile is not optional
                    in_bucket = pool.eligible(need, bucket, exclude_flagged=exclude_flagged,
                                              exclude=None)
                cand = in_bucket[~np.isin(in_bucket, used)] if used else in_bucket
                if not cand.size and in_bucket.size:
                    counts = np.array([uses.get(int(j), 0) for j in in_bucket])
                    cand = in_bucket[counts == counts.min()]      # round-robin reuse
                elif not cand.size:
                    self.n_bucket_fallbacks += 1
                    cand = pool.eligible(need, None, exclude_flagged=exclude_flagged,
                                         exclude=exclude)
            if not cand.size:
                raise ValueError(
                    f"no {role} file can hold a {dur:.2f}s tile plus 2 x {margin}s margin "
                    f"(slice={self.slice_!r} fold={self.fold!r})")
            j = int(cand[int(rng.integers(len(cand)))])
            used.append(j)
            uses[j] = uses.get(j, 0) + 1
            high = float(pool.duration[j]) - margin - dur
            out.append(ComponentDraw(
                file_id=str(pool.file_id[j]), role=role,
                source_offset_s=float(rng.uniform(margin, high)),
                duration_s=dur, target_start_s=slot_start + i * tile,
                gain_db=gain_db, slot=slot))
        return out

    def _draw_layer(self, rng: np.random.Generator, voice_present: bool,
                    exclude: frozenset[str] | None, lead: float, span: float, take: float,
                    xfade: float, snr_db: float) -> list[ComponentDraw]:
        """DRAW-5: pool-E tiles over the span at ``snr_db``. Rows flagged
        ``noise_has_speech`` are excluded under a ``voice_present = 0`` cell;
        ``exclude`` keeps a cell-9 sample from layering its own files under
        itself."""
        pool = self._pools[("noise", False)]
        if not len(pool):
            return []
        tiles = self._tiles(rng, pool, None, "noise", LAYER_SLOT, lead, span, 0.0, take,
                            xfade, extend_last=False, exclude_flagged=not voice_present,
                            exclude=exclude)
        return [replace(t, snr_db=snr_db) for t in tiles]

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
        # cell (R2: per sample, never per cell); the rows are drawn after it.
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

        # DRAW-4: structure, the crossfade, lead/tail. Drawn for EVERY sample,
        # before any file is chosen, so none can depend on which file was
        # drawn -- and the whole-file branch gets the same lead the composed
        # branch does (a lead only composed samples carried would be a
        # composedness cue).
        sequential = composed and len(wanted) > 1 and rng.random() < cfg.sequential_prob
        crossfade_ms = float(rng.uniform(*cfg.crossfade_ms_range))
        xfade = crossfade_ms / 1000.0
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
            one = _Pool(np.array([str(row.file_id)], dtype=object),
                        np.array([float(row.duration_s)]), np.zeros(1, dtype=int),
                        np.ones(1), np.zeros(1, dtype=bool), {}, (np.zeros(1, int), None))
            tiles = self._tiles(rng, one, None, role, 0, lead, span, 0.0, take, xfade,
                                extend_last=False, single=0)
            if layer:
                tiles += self._draw_layer(rng, bool(vp), frozenset({str(row.file_id)}), lead,
                                          span, take, xfade, layer_snr)
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
            # cell 9's primary noise never carries speech (05 A3)
            exclude_flagged = role == "noise" and not vp
            pool = self._pools[(role, fake)]
            anchor = self._anchor(rng, role, fake, exclude_flagged=exclude_flagged)
            if sequential:
                slot_len = span / len(wanted)
                start = lead + i * slot_len
            else:
                slot_len, start = span, lead
            gain = self._gain_db(rng) if (is_ratio and role == "voice") else 0.0
            draws.extend(self._tiles(
                rng, pool, int(pool.bucket[anchor]), role, i, start, slot_len, gain, take,
                xfade, extend_last=sequential and i < len(wanted) - 1,
                exclude_flagged=exclude_flagged))
        if layer:
            own = (frozenset(c.file_id for c in draws) if wanted[0][0] == "noise" else None)
            draws.extend(self._draw_layer(rng, bool(vp), own, lead, span, take, xfade,
                                          layer_snr))

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
