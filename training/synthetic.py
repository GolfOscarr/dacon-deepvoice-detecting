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
                 "E": ("musan_noise", "esc50")}
_FAKE_FAMILIES = {"B": ("hifigan", "bigvgan", "encodec", "vocos", "dac",
                        "diffusion_tts", "flow_tts", "unknown_commercial"),
                  "D": ("suno_v3", "udio_v1", "musicgen", "stable_audio",
                        "riffusion", "boomy", "audioldm", "unknown_music")}


def synthetic_manifest(
    n_per_pool: int = 400,
    n_whole_file: int = 300,
    seed: int = 0,
    scheme_version: str = "synthetic-v1",
) -> pd.DataFrame:
    """A manifest with the shape and the pathologies of the real thing.

    ``n_per_pool`` component rows in each of A-E, plus ``n_whole_file`` rows
    spread over the cells that can be scraped whole (1-5, 8, 9). ⚠️ Cells 6 and
    7 are deliberately absent from the whole-file rows: they cannot be scraped,
    which is the entire reason two fake heads exist.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

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
            fams = _FAKE_FAMILIES[pool]
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
                extra = {"pair": f"twin{i}" if rng.random() < 0.15 else None}
            else:
                fam, src, dom = None, srcs[picks[i]], None
                labels = ((1, 0, 0, None) if pool == "A" else
                          (0, 1, None, 0) if pool == "C" else (0, 0, None, None))
                extra = {"speaker": f"{src}_spk{rng.integers(0, 60)}",
                         "pair": f"twin{i}" if rng.random() < 0.15 else None,
                         "dup": f"dup{rng.integers(0, 40)}" if rng.random() < 0.08 else None}
            add(fid, "component", pool, None, labels, fam, src, dom, extra)

    # -- whole-file rows: only the cells that can be scraped or generated whole #
    from training.spec import CELL_TABLE
    whole_cells = (1, 2, 3, 4, 5, 8, 9)
    for i in range(n_whole_file):
        cell = int(rng.choice(whole_cells))
        vp, mp, vf, mf = CELL_TABLE[cell]
        if cell in (2, 4, 8):                       # generated whole files
            pool_for_fam = "B" if cell == 2 else "D"
            fam = str(rng.choice(_FAKE_FAMILIES[pool_for_fam]))
            src, dom = f"gen_{fam}", f"gen_{fam}::{fam}"
        else:
            fam, src, dom = None, str(rng.choice(("jamendo", "fma", "aihub_kr"))), None
        add(f"W{i:05d}", "whole_file", None, cell, (vp, mp, vf, mf), fam, src, dom,
            {"speaker": None, "pair": None, "dup": None})

    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    df["cell"] = df["cell"].astype("Int64")
    df["fold"] = df["fold"].astype("Int64")
    for c in ("label_voice_present", "label_music_present",
              "label_voice_fake", "label_music_fake"):
        df[c] = df[c].astype("Int64")
    return validate_manifest(df)
