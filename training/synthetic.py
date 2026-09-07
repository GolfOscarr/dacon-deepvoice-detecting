"""A synthetic manifest, and later a synthetic corpus.

🔴 Not test scaffolding -- a first-class deliverable. The corpus does not exist
and will not for days (docs/data/08), while the sampler, the audit and the fold
builder are all buildable now. This module is what lets M1-M4 be written and
fully tested before a single real file is downloaded.

⚠️ It deliberately generates a corpus with **realistic pathologies**: unequal
family sizes, over-represented generator domains, duplicate groups and
resynthesis twins. A generator that produced a perfectly balanced corpus would
make the audit vacuous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.manifest import POOL_IS_FAKE, REQUIRED_COLUMNS, validate_manifest

__all__ = ["synthetic_manifest"]

_CONTAINERS = ("wav", "mp3", "flac")
_REAL_SOURCES = {"A": ("libritts", "commonvoice", "aihub_kr"),
                 "C": ("musdb18", "jamendo", "fma"),
                 "E": ("musan_noise", "esc50", "audioset_bg")}
_FAKE_FAMILIES = {"B": ("hifigan", "bigvgan", "encodec", "vocos", "dac",
                        "diffusion_tts", "flow_tts", "unknown_commercial"),
                  "D": ("suno_v3", "udio_v1", "musicgen", "stable_audio",
                        "riffusion", "boomy", "audioldm", "unknown_music")}

#: 🔴 Which (fake family, real source) combinations were resynthesized into
#: T1/T2/T3 twins -- `(fake pool, real pool, family index, source index)`.
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


def _assign_twins(rows: list[dict], fam_names: dict[str, tuple[str, ...]]) -> None:
    """Give each resynthesis twin the same `pair_id` as the real file it came from.

    A twin is one real component and the fake component generated *from it*, so
    the pair id is shared by exactly two rows of the same role. Mutates `rows`.
    """
    by_pool: dict[str, list[dict]] = {}
    for row in rows:
        if row["row_kind"] == "component":
            by_pool.setdefault(row["pool"], []).append(row)

    for fake_pool, real_pool, fam_i, src_i in _TWIN_COMBINATIONS:
        fams, srcs = fam_names[fake_pool], _REAL_SOURCES[real_pool]
        if fam_i >= len(fams) or src_i >= len(srcs):
            continue
        fam, src = fams[fam_i], srcs[src_i]
        reals = [r for r in by_pool.get(real_pool, []) if r["source_name"] == src]
        fakes = [r for r in by_pool.get(fake_pool, []) if r["artifact_family"] == fam]
        for k, (real, fake) in enumerate(zip(reals, fakes)):
            real["pair_id"] = fake["pair_id"] = f"{src}__{fam}__t{k:05d}"


def _family_names(pool: str, n_families: int | None) -> tuple[str, ...]:
    """The generator families of one fake pool, optionally widened.

    ⚠️ The default eight per pool is below what a 5-fold needs once ~25% of
    families are held out for VAL (docs/validation/01 §3). `n_families` exists so
    a test can build a corpus that *can* support the requested fold count --
    never so the builder can assume one does.
    """
    base = _FAKE_FAMILIES[pool]
    if n_families is None:
        return base
    if n_families < 1:
        raise ValueError(f"n_families must be >= 1, got {n_families}")
    return tuple(base[i % len(base)] + ("" if i < len(base) else f"_v{i // len(base) + 1}")
                 for i in range(n_families))


def synthetic_manifest(
    n_per_pool: int = 400,
    n_whole_file: int = 300,
    seed: int = 0,
    scheme_version: str = "synthetic-v1",
    n_families: int | None = None,
) -> pd.DataFrame:
    """A manifest with the shape and the pathologies of the real thing.

    ``n_per_pool`` component rows in each of A-E, plus ``n_whole_file`` rows
    spread over the cells that can be scraped whole (1-5, 8, 9). ⚠️ Cells 6 and
    7 are deliberately absent from the whole-file rows: they cannot be scraped,
    which is the entire reason two fake heads exist.

    ``n_families`` widens each fake pool beyond its default eight generators.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    fam_names = {p: _family_names(p, n_families) for p in ("B", "D")}

    def add(file_id, row_kind, pool, cell, labels, family, source, domain, extra=None):
        vp, mp, vf, mf = labels
        rows.append({
            "file_id": file_id,
            "path": f"pools/{pool or 'whole'}/{file_id}.wav",
            "sha256": f"{abs(hash(file_id)) & 0xFFFFFFFFFFFF:012x}",
            "row_kind": row_kind,
            "pool": pool,
            "cell": cell,
            "duration_s": float(rng.uniform(5.0, 240.0)),
            "orig_sr": int(rng.choice([16_000, 22_050, 44_100, 48_000])),
            "orig_channels": int(rng.choice([1, 2])),
            "container": str(rng.choice(_CONTAINERS)),
            "label_voice_present": vp, "label_music_present": mp,
            "label_voice_fake": vf, "label_music_fake": mf,
            "artifact_family": family,
            "source_name": source,
            "speaker_ref_id": (extra or {}).get("speaker"),
            "pair_id": (extra or {}).get("pair"),
            "dup_group": (extra or {}).get("dup"),
            "domain_key": domain,
            "slice": "train", "fold": None, "scheme_version": scheme_version,
            "validity_mask_ref": None,
            "label_confidence": "exact" if (family is not None) else "reported",
            "aug_strength": float(rng.uniform(0.5, 1.5)),
        })

    # -- component rows ---------------------------------------------------- #
    for pool in ("A", "B", "C", "D", "E"):
        fake = POOL_IS_FAKE[pool]
        if fake:
            fams = fam_names[pool]
            # ⚠️ Zipf-ish family sizes: a handful of generators dominate, which is
            # exactly the imbalance DOSS capping exists to flatten.
            weights = np.array([1.0 / (i + 1) for i in range(len(fams))])
            weights /= weights.sum()
            picks = rng.choice(len(fams), size=n_per_pool, p=weights)
        else:
            srcs = _REAL_SOURCES[pool]
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
                # ⚠️ `dup_group` is namespaced by source. A near-duplicate hash
                # group spanning LibriTTS and MUSDB18 is not a thing, and an
                # unnamespaced `dup{k}` fused every real corpus into one group.
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
            fam, src, dom = None, str(rng.choice(("jamendo", "fma", "aihub_kr"))), None
        add(f"W{i:05d}", "whole_file", None, cell, (vp, mp, vf, mf), fam, src, dom,
            {"speaker": None, "pair": None, "dup": None})

    _assign_twins(rows, fam_names)

    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    df["cell"] = df["cell"].astype("Int64")
    df["fold"] = df["fold"].astype("Int64")
    for c in ("label_voice_present", "label_music_present",
              "label_voice_fake", "label_music_fake"):
        df[c] = df[c].astype("Int64")
    return validate_manifest(df)
