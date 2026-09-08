"""Drawing a `SampleSpec`. Pure: reads the manifest, touches no audio.

Strong evidence: the Kaggle ordering puts this stage *second*, ahead of
architecture -- `data split -> sampler/loss -> architecture -> optimizer` --
with a named warning against over-searching architecture before solving data
shift (docs/kaggle/05 E8).

Three constraints from the objective live here, because the loss is unsound
without them (docs/pipelines/02 §4, docs/training/02 §8):

* **C1** per-head positive rate in [0.2, 0.8], measured *after masking*.
* **C2** a floor on the per-head present-count per batch.
* **C3** composedness label-independent *within each presence stratum*, and
  mixedness label-independent across them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from training.manifest import POOL_IS_FAKE, ROLE_POOLS
from training.spec import (CELL_TABLE, ComponentDraw, SampleSpec, is_fake_cell,
                           spec_rng, stratum_of)

__all__ = ["C1_BOUNDS", "REFERENCE_MIX", "CellMix", "SamplerConfig", "Sampler",
           "composed_fractions", "head_positive_rates", "mixedness_balance"]

#: docs/training/01 §2.4. Below ~0.10 is where published AUC-surrogate gains
#: start to exceed our noise floor; inside this band a pairwise term is worth 0.
C1_BOUNDS = (0.2, 0.8)

#: Verified against C1 and C3 (docs/pipelines/02 §4). Over-weights cells 6/7
#: above their uniform 1/9 share while keeping every head inside its bound.
REFERENCE_MIX: dict[int, float] = {
    1: 0.060, 2: 0.130, 3: 0.060, 4: 0.135, 5: 0.155,
    6: 0.125, 7: 0.125, 8: 0.095, 9: 0.115,
}


@dataclass(frozen=True)
class CellMix:
    """A distribution over the nine cells."""

    p: dict[int, float] = field(default_factory=lambda: dict(REFERENCE_MIX))

    def __post_init__(self) -> None:
        if set(self.p) != set(CELL_TABLE):
            raise ValueError(f"a cell mix needs all of 1-9, got {sorted(self.p)}")
        if any(v < 0 for v in self.p.values()):
            raise ValueError("cell probabilities must be >= 0")
        total = sum(self.p.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"cell mix must sum to 1, got {total}")

    def draw(self, rng: np.random.Generator) -> int:
        cells = sorted(self.p)
        return int(rng.choice(cells, p=[self.p[c] for c in cells]))


def composed_fractions(mix: CellMix, f8: float, a: float = 0.0, b: float = 0.0,
                       f9: float = 0.0, balance_marginal: bool = True) -> dict[int, float]:
    """The per-cell composed fraction implied by `f8`.

    Critical: `f8` is the only knob, and the two policies are its endpoints:
    `f8 = 0` is label-conditional (natural songs and AI songs usable, ~13.8%
    of the corpus genuine whole-file audio), `f8 = 1` is strict (every mixed
    file composed). `f6 = f7 = 1` is forced -- those cells cannot be scraped.

    Caveat: pairwise equality `f5 = f6 = f7 = f8` is *too strong*: it drives
    `f5 = f8 = 1` and bans both pools. Marginal balance is *too weak*: it can
    hold while composedness predicts the label inside a stratum. The stratified
    form below is neither (docs/data/02).
    """
    if not 0.0 <= f8 <= 1.0:
        raise ValueError(f"f8 must be in [0, 1], got {f8}")
    for name, v in (("a", a), ("b", b), ("f9", f9)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    p = mix.p
    mixed_mass = p[6] + p[7] + p[8]
    if mixed_mass <= 0:
        raise ValueError("the mixed-fake cells 6/7/8 cannot all have zero mass")
    f5 = (p[6] * 1.0 + p[7] * 1.0 + p[8] * f8) / mixed_mass
    f = {1: a, 2: a, 3: b, 4: b, 5: f5, 6: 1.0, 7: 1.0, 8: f8, 9: f9}
    if balance_marginal:
        f[9] = _f9_for_marginal_balance(p, f)
    return f


def _f9_for_marginal_balance(p: dict[int, float], f: dict[int, float]) -> float:
    """The cell-9 composed fraction that removes the *marginal* composedness gap.

    Critical: the stratified constraint leaves a marginal residual, because
    cell 9 is its own presence stratum with no FAKE counterpart to balance
    against. It is emitted at p≈0.115 with f₉=0, so "not composed" skews REAL:
    measured **P(composed|REAL) 0.2875 vs P(composed|FAKE) 0.4090**, a gap of
    0.121, against 0.0001 over cells 1-8.

    Caveat: the defence -- composedness is uninformative *given* presence, and
    the model is supervised on presence -- is sound but was asserted in
    docstrings and measured nowhere. Closing it costs nothing: a composed
    cell-9 sample is a pool-E noise draw the sampler already supports.

    Caveat: `f₉ = f₅` does **not** close it, it overshoots (0.5015 against a
    0.4090 target). Solve instead.
    """
    fake = [c for c in p if is_fake_cell(c)]
    real = [c for c in p if not is_fake_cell(c)]
    m_fake, m_real = sum(p[c] for c in fake), sum(p[c] for c in real)
    if not m_fake or not m_real or p[9] <= 0:
        return f[9]
    target = sum(p[c] * f[c] for c in fake) / m_fake
    without_9 = sum(p[c] * f[c] for c in real if c != 9)
    return float(min(1.0, max(0.0, (target * m_real - without_9) / p[9])))


def head_positive_rates(mix: CellMix) -> dict[str, float]:
    """C1's five quantities, each measured over its own masked pool."""
    p = mix.p
    v_pool = sum(p[c] for c in p if CELL_TABLE[c][0])
    m_pool = sum(p[c] for c in p if CELL_TABLE[c][1])
    return {
        "file": sum(p[c] for c in p if is_fake_cell(c)),
        "voice": sum(p[c] for c in p if CELL_TABLE[c][0] and CELL_TABLE[c][2]) / v_pool,
        "music": sum(p[c] for c in p if CELL_TABLE[c][1] and CELL_TABLE[c][3]) / m_pool,
        "v_pres": v_pool,
        "m_pres": m_pool,
    }


def mixedness_balance(mix: CellMix) -> tuple[float, float]:
    """`P(mixed | FAKE)` and `P(mixed | REAL)` over **cells 1-8**.

    Caveat: cell 9 is excluded deliberately: a file with no components cannot
    be fake, so `P(FAKE | cell 9) = 0` by definition. That is a true property
    of the label space -- it holds at test time too -- and including cell 9
    lets a mix look balanced for the wrong reason.
    """
    p = {c: v for c, v in mix.p.items() if c != 9}
    fake = [c for c in p if is_fake_cell(c)]
    real = [c for c in p if not is_fake_cell(c)]
    mixed = lambda c: CELL_TABLE[c][0] and CELL_TABLE[c][1]      # noqa: E731
    return (sum(p[c] for c in fake if mixed(c)) / sum(p[c] for c in fake),
            sum(p[c] for c in real if mixed(c)) / sum(p[c] for c in real))


@dataclass(frozen=True)
class SamplerConfig:
    """Everything the sampler needs, and nothing it does not."""

    cell_mix: CellMix = field(default_factory=CellMix)
    #: 0.0 = label-conditional (primary), 1.0 = strict. See `composed_fractions`.
    f8: float = 0.0
    #: Composed rate for the single-component cells. `f1 = f2` and `f3 = f4`
    #: keep composedness label-independent inside those strata.
    single_composed_rate: float = 0.0
    noise_composed_rate: float = 0.0
    #: Critical: solve `f9` so the *marginal* composedness gap closes. The
    #: stratified constraint leaves one, because cell 9 is its own presence
    #: stratum with no FAKE counterpart: measured 0.121 without this. Costs
    #: nothing -- a composed cell-9 sample is a pool-E noise draw the sampler
    #: already supports.
    balance_marginal_composedness: bool = True
    #: DOSS per-domain cap. Strong evidence: 0.2k h domain-balanced -> 2.77%
    #: EER vs 6.4k h naive -> 3.29% (docs/papers/05). A weight, so nothing is
    #: discarded.
    domain_cap: int = 500
    #: The test duration range (competition/01); AudioConfig agrees.
    duration_range: tuple[float, float] = (4.0, 60.0)
    #: Strong evidence: G2Net -- models generalise low-SNR -> high-SNR but not
    #: the reverse, so the voice/music gain ratio is skewed toward the quiet
    #: end (A-A3).
    gain_db_range: tuple[float, float] = (-15.0, 15.0)
    gain_db_mean: float = -3.6
    gain_db_sigma: float = 4.0
    sequential_prob: float = 0.25
    crossfade_ms_range: tuple[float, float] = (10.0, 200.0)
    #: Critical: A-A8 (±0.5 s temporal misalignment) and A-A11 (silence
    #: edits), as *draws* rather than step-4 augments -- they move audio along
    #: the timeline `frame_intervals` describe, and only a draw is recorded in
    #: the spec (docs/pipelines/03 §4). Both are the same mechanism: the
    #: components occupy `duration_s - lead - tail`, starting at `lead`.
    #: Caveat: default 0.0. Switching them on changes the drawn stream and
    #: needs an I1b re-run; at 0.0 no RNG draw happens, so the stream is
    #: unchanged.
    silence_lead_s: float = 0.0
    silence_tail_s: float = 0.0
    scheme_version: str = "synthetic-v1"

    def __post_init__(self) -> None:
        lo, hi = self.duration_range
        if not 0 < lo < hi:
            raise ValueError(f"duration_range must be 0 < lo < hi, got {self.duration_range}")
        if self.domain_cap < 1:
            raise ValueError(f"domain_cap must be >= 1, got {self.domain_cap}")
        for name in ("silence_lead_s", "silence_tail_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.sequential_prob <= 1.0:
            raise ValueError(f"sequential_prob must be in [0, 1], got {self.sequential_prob}")
        composed_fractions(                       # validates the knobs
            self.cell_mix, self.f8,
            a=self.single_composed_rate, b=self.single_composed_rate,
            f9=self.noise_composed_rate,
            balance_marginal=self.balance_marginal_composedness)

    @property
    def f(self) -> dict[int, float]:
        # Caveat: keywords, not position. `noise_composed_rate` is f9; passing
        # it positionally put it in `b` -- the MUSIC-ONLY composed rate -- so
        # the two single-component strata silently disagreed (f1=f2=a but
        # f3=f4=f9).
        return composed_fractions(
            self.cell_mix, self.f8,
            a=self.single_composed_rate, b=self.single_composed_rate,
            f9=self.noise_composed_rate,
            balance_marginal=self.balance_marginal_composedness)


class Sampler:
    """Draws specs from one slice of one fold. Pure, and cheap enough to audit."""

    def __init__(self, manifest: pd.DataFrame, cfg: SamplerConfig | None = None,
                 slice_: str = "train", fold: int | None = None):
        self.cfg = cfg or SamplerConfig()
        df = manifest[manifest["slice"] == slice_]
        if fold is not None:
            df = df[df["fold"] == fold]
        if df.empty:
            raise ValueError(f"no manifest rows for slice={slice_!r} fold={fold!r}")
        self.slice_, self.fold = slice_, fold

        # Critical: a source shorter than the minimum duration would produce
        # a spec below `AudioConfig.min_seconds`:
        # `take = min(duration, row.duration_s)` silently shortens the
        # timeline. Measured on a corpus of 1.5 s scraped clips: 2,580 of
        # 4,000 specs came out under the competition's own 4 s floor. Drop
        # those rows here rather than emit an out-of-range sample.
        min_seconds = self.cfg.duration_range[0]
        usable = df[df.duration_s >= min_seconds]
        self.n_dropped_short = int(len(df) - len(usable))
        if usable.empty:
            raise ValueError(
                f"every row in slice={slice_!r} fold={fold!r} is shorter than "
                f"{min_seconds}s; nothing can be drawn")
        df = usable

        comp = df[df.row_kind == "component"]
        self._by_role = {
            role: comp[comp.pool.isin(pools)]
            for role, pools in ROLE_POOLS.items()
        }
        #: Component rows split by fake status, so a cell can ask for exactly
        #: the kind of component it needs.
        self._by_role_fake: dict[tuple[str, bool], pd.DataFrame] = {}
        self._weights: dict[tuple[str, bool], np.ndarray] = {}
        for role, sub in self._by_role.items():
            for fake in (False, True):
                rows = sub[sub.pool.map(POOL_IS_FAKE) == fake]
                self._by_role_fake[(role, fake)] = rows
                self._weights[(role, fake)] = self._doss_weights(rows)

        self._whole_by_cell = {
            int(c): g for c, g in df[df.row_kind == "whole_file"].groupby("cell")
        }

    # -- DOSS ---------------------------------------------------------------- #

    def _doss_weights(self, rows: pd.DataFrame) -> np.ndarray:
        """`w(file) = min(count(domain), N_c) / count(domain)`, normalised.

        The paper caps files per domain at dataset-construction time; we compose
        an unbounded stream, so the analogue is a sampling weight. Same
        balancing effect, nothing discarded, and `N_c` becomes a sweepable float.
        """
        if rows.empty:
            return np.empty(0)
        domain = rows.domain_key.fillna("__real__")
        counts = domain.map(domain.value_counts())
        w = np.minimum(counts, self.cfg.domain_cap) / counts
        return (w / w.sum()).to_numpy(dtype=float)

    # -- drawing ------------------------------------------------------------- #

    def _draw_component(self, rng, role: str, fake: bool) -> pd.Series:
        rows = self._by_role_fake[(role, fake)]
        if rows.empty:
            raise ValueError(
                f"slice={self.slice_!r} fold={self.fold!r} has no "
                f"{'fake' if fake else 'real'} {role} components")
        return rows.iloc[int(rng.choice(len(rows), p=self._weights[(role, fake)]))]

    def _gain_db(self, rng) -> float:
        """Skewed toward the quiet end, then clipped to the swept range."""
        lo, hi = self.cfg.gain_db_range
        return float(np.clip(rng.normal(self.cfg.gain_db_mean, self.cfg.gain_db_sigma), lo, hi))

    def sample_spec(self, sample_id: int, epoch: int = 0, seed: int = 0) -> SampleSpec:
        rng = spec_rng(sample_id, epoch, seed)
        cfg = self.cfg

        # step 0: the timeline, drawn first so nothing is rendered then discarded
        lo, hi = cfg.duration_range
        duration = float(rng.uniform(lo, hi))

        # steps 1-2: cell, render mode, components. Labels are fixed here.
        cell = cfg.cell_mix.draw(rng)
        f = cfg.f[cell]
        composed = bool(rng.random() < f)
        if cell in (6, 7):
            composed = True                       # cannot be scraped
        if not composed and cell not in self._whole_by_cell:
            composed = True                       # no whole-file row available

        if not composed:
            rows = self._whole_by_cell[cell]
            row = rows.iloc[int(rng.integers(len(rows)))]
            take = min(duration, float(row.duration_s))
            return SampleSpec(
                sample_id=sample_id, epoch=epoch, seed=seed,
                scheme_version=cfg.scheme_version,
                duration_s=take, cell=cell, render_mode="whole_file",
                structure="overlap",
                components=(ComponentDraw(
                    file_id=str(row.file_id),
                    role="voice" if CELL_TABLE[cell][0] else
                         "music" if CELL_TABLE[cell][1] else "noise",
                    source_offset_s=float(rng.uniform(
                        0.0, max(0.0, float(row.duration_s) - take))),
                    duration_s=take, target_start_s=0.0, gain_db=0.0),),
            )

        vp, mp, vf, mf = CELL_TABLE[cell]
        wanted: list[tuple[str, bool]] = []
        if vp:
            wanted.append(("voice", bool(vf)))
        if mp:
            wanted.append(("music", bool(mf)))
        if not wanted:
            wanted.append(("noise", False))

        sequential = len(wanted) > 1 and rng.random() < cfg.sequential_prob
        # Caveat: `gain_db` is the voice/music LEVEL RATIO (docs/data/02, A-A3)
        # -- the quantity G2Net's low-SNR-generalisation result is about. It is
        # only meaningful against something, so it is drawn only when both
        # components are present. Applied to a solo voice component it silently
        # became an absolute level shift: that is A-A7 gain jitter, a
        # label-independent augment which belongs in the augment registry, not
        # in the component draw, and which nothing renormalises
        # (`render._normalize` models the test chain only -- there is no
        # loudness stage), so it reached the waveform as a composedness cue
        # inside the voice-only stratum.
        # Critical: a no-op on the shipped stream: `single_composed_rate = 0.0`
        # emits no composed voice-only samples at all, so only `a`/`b` sweeps
        # change.
        is_ratio = len(wanted) > 1
        # Critical: A-A8 (temporal misalignment, ±0.5 s) and A-A11
        # (leading/trailing silence) are drawn HERE, not applied as step-4
        # augments. They move audio along the timeline, and `frame_intervals`
        # are intervals on that timeline: an augment returns only a waveform,
        # so it has no way to say the timeline moved, and the frame labels
        # would silently describe audio that is no longer there
        # (docs/pipelines/03 §4). Drawn, they are just a placement -- the spec
        # records it and the frame targets follow it for free, because they are
        # computed from the placement.
        # Caveat: both default to 0.0: switching them on changes the drawn
        # stream, so it needs an I1b re-run rather than a default flip. At 0.0
        # no RNG draw happens at all, so the shipped stream is byte-identical.
        lead = float(rng.uniform(0.0, cfg.silence_lead_s)) if cfg.silence_lead_s else 0.0
        tail = float(rng.uniform(0.0, cfg.silence_tail_s)) if cfg.silence_tail_s else 0.0
        if lead + tail > 0.5 * duration:          # never silence half the sample
            scale = 0.5 * duration / (lead + tail)
            lead, tail = lead * scale, tail * scale
        span = duration - lead - tail

        draws: list[ComponentDraw] = []
        for i, (role, fake) in enumerate(wanted):
            row = self._draw_component(rng, role, fake)
            if sequential:
                seg = span / len(wanted)
                start, take = lead + i * seg, seg
            else:
                start, take = lead, span
            take = min(take, float(row.duration_s))
            draws.append(ComponentDraw(
                file_id=str(row.file_id), role=role,
                source_offset_s=float(rng.uniform(
                    0.0, max(0.0, float(row.duration_s) - take))),
                duration_s=take, target_start_s=start,
                gain_db=self._gain_db(rng) if (is_ratio and role == "voice") else 0.0))

        return SampleSpec(
            sample_id=sample_id, epoch=epoch, seed=seed,
            scheme_version=cfg.scheme_version,
            duration_s=duration, cell=cell, render_mode="composed",
            structure="sequential" if sequential else "overlap",
            components=tuple(draws),
            crossfade_ms=float(rng.uniform(*cfg.crossfade_ms_range)) if sequential else 0.0,
        )

    def epoch_specs(self, n: int, epoch: int = 0, seed: int = 0):
        """An epoch is a fixed count of drawn specs -- otherwise `sample_id` is
        undefined and reproducibility is nominal (docs/pipelines/02 §6)."""
        # Caveat: `i`, not `epoch * n + i`. `epoch` is already in the RNG key,
        # and a global index makes the whole corpus depend on
        # `steps_per_epoch`: changing the batch size would silently re-roll
        # every sample.
        for i in range(n):
            yield self.sample_spec(i, epoch=epoch, seed=seed)
