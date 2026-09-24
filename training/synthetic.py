"""A synthetic manifest, and the synthetic corpus it names.

Critical: not test scaffolding -- a first-class deliverable. The corpus does
not exist and will not for days (docs/data/08), while the sampler, the audit
and the fold builder are all buildable now. This module is what lets M1-M4 be
written and fully tested before a single real file is downloaded.

Caveat: it deliberately generates a corpus with **realistic pathologies**:
unequal family sizes, over-represented generator domains, duplicate groups and
resynthesis twins. A generator that produced a perfectly balanced corpus would
make the audit vacuous.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from training.manifest import POOL_IS_FAKE, REQUIRED_COLUMNS, validate_manifest

__all__ = ["synthetic_manifest", "write_synthetic_corpus"]

_CONTAINERS = ("wav", "mp3", "flac")
_REAL_SOURCES = {"A": ("libritts", "commonvoice", "aihub_kr"),
                 "C": ("musdb18", "jamendo", "fma"),
                 "E": ("musan_noise", "esc50", "audioset_bg")}
_FAKE_FAMILIES = {"B": ("hifigan", "bigvgan", "encodec", "vocos", "dac",
                        "diffusion_tts", "flow_tts", "unknown_commercial"),
                  "D": ("suno_v3", "udio_v1", "musicgen", "stable_audio",
                        "riffusion", "boomy", "audioldm", "unknown_music")}

#: Critical: which (fake family, real source) combinations were resynthesized
#: into T1/T2/T3 twins -- `(fake pool, real pool, family index, source index)`.
#:
#: Twins are **not** a property of every fake row. A twin is a file *we* generate
#: (docs/data/05 S-S1), so the set of (corpus, vocoder) combinations that carry
#: one is a deliberate, bounded choice. The earlier version drew `pair_id` at
#: random and independently per pool, which meant a real *instrumental* and a
#: real *voice* row could share a pair id by index collision, and every family
#: was twinned to every real corpus. Since `pair_id` binds its members to one
#: slice (docs/validation/01 §1), that fused the entire corpus into a single
#: inseparable group -- 1,088 of 1,200 rows -- and no family-disjoint split
#: existed at all. See `training.folds.grouping_atoms`.
_TWIN_COMBINATIONS = (("B", "A", 0, 0), ("B", "A", 1, 1), ("D", "C", 0, 0))


def _widen(names: tuple[str, ...], n: int | None, what: str) -> tuple[str, ...]:
    """`names`, cycled with a numeric suffix out to `n` entries. `None` = as-is.

    Caveat: exists so a test can build a corpus that *can* support the fold
    count it asks for -- never so the builder can assume one does. Under the
    rotating scheme (docs/validation/01 §3) the fold count is bounded by the
    number of real source corpora per role as well as by the family count: each
    of the k VAL sides needs its own, and PROBE needs one more.
    """
    if n is None:
        return names
    if n < 1:
        raise ValueError(f"{what} must be >= 1, got {n}")
    return tuple(names[i % len(names)]
                 + ("" if i < len(names) else f"_v{i // len(names) + 1}")
                 for i in range(n))


def _assign_twins(rows: list[dict], fam_names: dict[str, tuple[str, ...]],
                  src_names: dict[str, tuple[str, ...]]) -> None:
    """Give each resynthesis twin the same `pair_id` as the real file it came from.

    A twin is one real component and the fake component generated *from it*, so
    the pair id is shared by exactly two rows of the same role. Mutates `rows`.
    """
    by_pool: dict[str, list[dict]] = {}
    for row in rows:
        if row["row_kind"] == "component":
            by_pool.setdefault(row["pool"], []).append(row)

    for fake_pool, real_pool, fam_i, src_i in _TWIN_COMBINATIONS:
        fams, srcs = fam_names[fake_pool], src_names[real_pool]
        if fam_i >= len(fams) or src_i >= len(srcs):
            continue
        fam, src = fams[fam_i], srcs[src_i]
        reals = [r for r in by_pool.get(real_pool, []) if r["source_name"] == src]
        fakes = [r for r in by_pool.get(fake_pool, []) if r["artifact_family"] == fam]
        for k, (real, fake) in enumerate(zip(reals, fakes)):
            real["pair_id"] = fake["pair_id"] = f"{src}__{fam}__t{k:05d}"


def _family_names(pool: str, n_families: int | None) -> tuple[str, ...]:
    """The generator families of one fake pool, optionally widened.

    Caveat: the default eight per pool is below what a 5-fold needs once ~25%
    of families are held out for VAL (docs/validation/01 §3). `n_families`
    exists so a test can build a corpus that *can* support the requested fold
    count -- never so the builder can assume one does.
    """
    return _widen(_FAKE_FAMILIES[pool], n_families, "n_families")


def synthetic_manifest(
    n_per_pool: int = 400,
    n_whole_file: int = 300,
    seed: int = 0,
    scheme_version: str = "synthetic-v1",
    n_families: int | None = None,
    n_sources: int | None = None,
    duration_range: tuple[float, float] = (5.0, 240.0),
) -> pd.DataFrame:
    """A manifest with the shape and the pathologies of the real thing.

    ``n_per_pool`` component rows in each of A-E, plus ``n_whole_file`` rows
    spread over the cells that can be scraped whole (1-5, 8, 9). Caveat: cells
    6 and 7 are deliberately absent from the whole-file rows: they cannot be
    scraped, which is the entire reason two fake heads exist.

    Critical: reproducible **across processes**, not only within one. Every
    column is a pure function of ``(seed, shape)``: the numeric draws come from
    a seeded `np.random.default_rng` and ``sha256`` from blake2b, because
    Python's `hash()` is salted per process and the column it produced differed
    between runs. `tests/test_synthetic.py` runs two interpreters under
    different ``PYTHONHASHSEED`` values and compares.

    ``n_families`` widens each fake pool beyond its default eight generators, and
    ``n_sources`` each real pool beyond its default three corpora.
    ``duration_range`` bounds the declared source durations -- shrink it before
    calling `write_synthetic_corpus`, which writes real audio for every row.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    fam_names = {p: _family_names(p, n_families) for p in ("B", "D")}
    src_names = {p: _widen(v, n_sources, "n_sources") for p, v in _REAL_SOURCES.items()}

    def add(file_id, row_kind, pool, cell, labels, family, source, domain, extra=None):
        vp, mp, vf, mf = labels
        container = str(rng.choice(_CONTAINERS))
        rows.append({
            "file_id": file_id,
            # Caveat: the extension agrees with `container` here, but nothing
            # in the pipeline may rely on that: P-S1 says "never assume the
            # extension", and the eval server hands us mixed containers.
            "path": f"pools/{pool or 'whole'}/{file_id}.{container}",
            # Critical: blake2b, **not** Python's `hash()`. String hashing is
            # salted per process, so a `hash()`-keyed digest is stable within a
            # run and different across runs -- which contradicts
            # `write_synthetic_corpus`'s "two runs produce byte-identical
            # corpora" and would make any manifest a run diffed against another
            # run's differ in a column nothing controls. Same hazard, same
            # answer as `training.spec.spec_rng`.
            "sha256": hashlib.blake2b(file_id.encode(), digest_size=6).hexdigest(),
            "row_kind": row_kind,
            "pool": pool,
            "cell": cell,
            "duration_s": float(rng.uniform(*duration_range)),
            "orig_sr": int(rng.choice([16_000, 22_050, 44_100, 48_000])),
            "orig_channels": int(rng.choice([1, 2])),
            "container": container,
            "label_voice_present": vp, "label_music_present": mp,
            "label_voice_fake": vf, "label_music_fake": mf,
            "artifact_family": family,
            "source_name": source,
            "speaker_ref_id": (extra or {}).get("speaker"),
            "pair_id": (extra or {}).get("pair"),
            "dup_group": (extra or {}).get("dup"),
            "domain_key": domain,
            "slice": "train", "fold": 0, "scheme_version": scheme_version,
            "validity_mask_ref": None,
            "label_confidence": "exact" if (family is not None) else "reported",
            "aug_strength": float(rng.uniform(0.5, 1.5)),
        })

    # -- component rows ---------------------------------------------------- #
    for pool in ("A", "B", "C", "D", "E"):
        fake = POOL_IS_FAKE[pool]
        if fake:
            fams = fam_names[pool]
            # Caveat: Zipf-ish family sizes: a handful of generators dominate,
            # which is exactly the imbalance DOSS capping exists to flatten.
            weights = np.array([1.0 / (i + 1) for i in range(len(fams))])
            weights /= weights.sum()
            picks = rng.choice(len(fams), size=n_per_pool, p=weights)
        else:
            srcs = src_names[pool]
            picks = rng.integers(0, len(srcs), size=n_per_pool)

        for i in range(n_per_pool):
            fid = f"{pool}{i:05d}"
            if fake:
                fam = fams[picks[i]]
                src = f"gen_{fam}"
                dom = f"{src}::{fam}"
                labels = ((1, 0, 1, None) if pool == "B" else (0, 1, None, 1))
                extra = None            # pairs are assigned by `_assign_twins`
            else:
                fam, src, dom = None, srcs[picks[i]], None
                labels = ((1, 0, 0, None) if pool == "A" else
                          (0, 1, None, 0) if pool == "C" else (0, 0, None, None))
                # Caveat: `dup_group` is namespaced by source. A near-duplicate
                # hash group spanning LibriTTS and MUSDB18 is not a thing, and
                # an unnamespaced `dup{k}` fused every real corpus into one
                # group.
                extra = {"speaker": f"{src}_spk{rng.integers(0, 60)}",
                         "dup": (f"{src}_dup{rng.integers(0, 40)}"
                                 if rng.random() < 0.08 else None)}
            add(fid, "component", pool, None, labels, fam, src, dom, extra)

    # -- whole-file rows: only the cells that can be scraped or generated whole #
    from training.spec import CELL_TABLE
    whole_cells = (1, 2, 3, 4, 5, 8, 9)
    for i in range(n_whole_file):
        cell = int(rng.choice(whole_cells))
        vp, mp, vf, mf = CELL_TABLE[cell]
        if cell in (2, 4, 8):                       # generated whole files
            pool_for_fam = "B" if cell == 2 else "D"
            fam = str(rng.choice(fam_names[pool_for_fam]))
            src, dom = f"gen_{fam}", f"gen_{fam}::{fam}"
        else:
            fam, src, dom = None, str(rng.choice(
                src_names["C"][:2] + src_names["A"][2:3])), None
        add(f"W{i:05d}", "whole_file", None, cell, (vp, mp, vf, mf), fam, src, dom,
            {"speaker": None, "pair": None, "dup": None})

    _assign_twins(rows, fam_names, src_names)

    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    df["cell"] = df["cell"].astype("Int64")
    df["fold"] = df["fold"].astype("Int64")
    for c in ("label_voice_present", "label_music_present",
              "label_voice_fake", "label_music_fake"):
        df[c] = df[c].astype("Int64")
    return validate_manifest(df)


# --------------------------------------------------------------------------- #
# A synthetic *corpus* -- the audio the manifest promises exists
#
# Critical: same reason as the manifest -- `render`, the collator and the
# rule-2.4 suite are all buildable now, and the corpus will not exist for days
# (docs/data/08). Without this, every render test would be a mock, and a mock
# cannot catch a decode, resample or codec defect -- which is where the
# failures actually are.


def _tone_bank(rng: np.random.Generator, n: int, sr: int, kind: str) -> np.ndarray:
    """One channel of plausible audio for a pool, deterministic given ``rng``."""
    t = np.arange(n, dtype=np.float64) / sr
    if kind == "voice":
        # A harmonic stack with vibrato and an amplitude envelope: enough
        # structure that a composition or a fade is visible in the waveform.
        f0 = float(rng.uniform(90.0, 240.0))
        vib = 1.0 + 0.01 * np.sin(2 * np.pi * float(rng.uniform(3.0, 7.0)) * t)
        x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t * vib) for k in range(1, 8))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * float(rng.uniform(0.3, 1.5)) * t)
        x = x * env
    elif kind == "music":
        root = float(rng.uniform(110.0, 330.0))
        x = sum(np.sin(2 * np.pi * root * r * t + float(rng.uniform(0, 6.28)))
                for r in (1.0, 1.26, 1.5, 2.0))
    else:                                            # noise
        x = np.cumsum(rng.standard_normal(n))        # pink-ish, not flat
        x = x - x.mean()
    x = x + 0.02 * rng.standard_normal(n)
    peak = float(np.max(np.abs(x))) or 1.0
    return (0.2 * x / peak).astype(np.float32)


def write_synthetic_corpus(manifest: pd.DataFrame, root: Path | str,
                           seed: int = 0) -> Path:
    """Write every file the manifest names, at its declared sr/channels/container.

    Deterministic per ``file_id``, so two runs produce byte-identical corpora and
    a reproducibility test is testing the pipeline rather than the fixture.
    Caveat: sizes follow ``duration_s`` -- pass
    ``synthetic_manifest(duration_range=...)`` something small before calling
    this.
    """
    import soundfile as sf

    from training.spec import CELL_TABLE

    root = Path(root)
    kind_for_pool = {"A": "voice", "B": "voice", "C": "music", "D": "music",
                     "E": "noise"}
    for row in manifest.to_dict(orient="records"):
        path = root / str(row["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.blake2b(f"{seed}:{row['file_id']}".encode(),
                                 digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        sr, n = int(row["orig_sr"]), int(round(float(row["duration_s"]) * int(row["orig_sr"])))
        if row["row_kind"] == "component":
            kind = kind_for_pool[str(row["pool"])]
        else:
            vp, mp, _, _ = CELL_TABLE[int(row["cell"])]
            kind = "voice" if vp else "music" if mp else "noise"
        channels = int(row["orig_channels"])
        data = np.stack([_tone_bank(rng, n, sr, kind) for _ in range(channels)], axis=1)
        sf.write(str(path), data, sr, format=str(row["container"]).upper())
    return root
