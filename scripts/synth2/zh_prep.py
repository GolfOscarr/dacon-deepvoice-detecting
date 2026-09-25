"""S4 (docs/training/09 §1): build the text and prompt indexes for the Mandarin fakes.

Two subcommands, run once on CPU; each writes a CSV under
``interim/zh-synth/_index/`` that the family scripts read with the stdlib only.

``texts``   (synth-cosyvoice venv, needs wetext)
    google/fleurs ``cmn_hans_cn`` transcripts (CC-BY-4.0; train/dev/test TSVs saved as
    ``_index/fleurs_cmn_<split>.tsv``), one row per FLEURS sentence id. Parenthesised
    glosses are cut, the text is normalised with WeTextProcessing zh TN (digits, %, dates
    -> hanzi, the same front end CosyVoice applies), and any sentence that still holds a
    Latin letter or digit, or has < 8 or > 60 hanzi, is dropped. No local corpus carries
    Mandarin transcripts (CFAD ships aishell1/aishell3/thchs30 wavs without text), so FLEURS
    is the text source; it never overlaps a prompt utterance.

``prompts`` (dacon311 venv, needs pandas)
    Chinese REAL speakers from manifests/strategy-v3: ``lang == "zh"``, ``pool == "A"``,
    folds.parquet ``slice == "train_val"`` (never ``probe``), with a real speaker id
    (``aishell1_*`` / ``thchs30_*``; the magicread/selfrecording rows share one id per
    source and are not speakers). Per speaker, up to 4 utterances of 3-10 s, drawn with a
    fixed seed. Output ``_index/prompts.csv``: prompt_id, speaker_ref_id, path
    (corpus-relative), duration_s, file_id.
"""
from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path

CORPUS = Path("/data/project/private/dacon-corpus")
IDX = CORPUS / "interim" / "zh-synth" / "_index"
WETEXT = Path("/data/project/private/dacon-weights/synth2/wetext")

HAN = re.compile(r"[一-鿿]")
BAD = re.compile(r"[A-Za-z0-9０-９Ａ-Ｚａ-ｚ]")
PAREN = re.compile(r"（[^（）]*）|\([^()]*\)")


def texts() -> None:
    from wetext import Normalizer
    tn = Normalizer(tagger_path=str(WETEXT / "zh/tn/tagger.fst"),
                    verbalizer_path=str(WETEXT / "zh/tn/verbalizer.fst"), lang="zh", operator="tn")
    seen: dict[str, tuple[str, str]] = {}
    for split in ("train", "dev", "test"):
        with (IDX / f"fleurs_cmn_{split}.tsv").open(encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                sid, raw = row[0], row[2]
                if sid in seen:
                    continue
                seen[sid] = (split, raw)
    kept, dropped = [], 0
    for sid, (split, raw) in sorted(seen.items(), key=lambda kv: int(kv[0])):
        t = PAREN.sub("", raw).strip()
        t = re.sub(r"\s+", "", tn.normalize(t))
        n_han = len(HAN.findall(t))
        if BAD.search(t) or not 8 <= n_han <= 60:
            dropped += 1
            continue
        kept.append((f"fleurs-cmn/{split}/{sid}", split, n_han, t, raw))
    with (IDX / "texts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "split", "n_han", "text", "raw_text"])
        w.writerows(kept)
    print(f"texts: {len(seen)} FLEURS sentences, kept {len(kept)}, dropped {dropped}")


def prompts() -> None:
    import pandas as pd
    m = pd.read_parquet(CORPUS / "manifests/strategy-v3/manifest.parquet",
                        columns=["file_id", "path", "pool", "lang", "speaker_ref_id", "duration_s"])
    f = pd.read_parquet(CORPUS / "manifests/strategy-v3/folds.parquet", columns=["file_id", "slice"])
    z = m[(m.lang == "zh") & (m.pool == "A")].merge(f, on="file_id", how="inner")
    probe_spk = set(z.loc[z["slice"] == "probe", "speaker_ref_id"])
    z = z[z["slice"] == "train_val"]
    z = z[z.speaker_ref_id.str.match(r"^(aishell1|thchs30)_")]
    assert not (set(z.speaker_ref_id) & probe_spk), "a train_val speaker also appears in probe"
    z = z[(z.duration_s >= 3.0) & (z.duration_s <= 10.0)].sort_values("file_id")
    rng = random.Random(4242)
    rows = []
    for spk, g in z.groupby("speaker_ref_id", sort=True):
        pick = rng.sample(list(g.itertuples(index=False)), min(4, len(g)))
        for k, r in enumerate(pick):
            if not (CORPUS / r.path).exists():
                continue
            rows.append((f"{spk}#{k}", spk, r.path, f"{r.duration_s:.3f}", r.file_id))
    with (IDX / "prompts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "speaker_ref_id", "path", "duration_s", "file_id"])
        w.writerows(rows)
    print(f"prompts: {len(rows)} utterances over {len({r[1] for r in rows})} speakers "
          f"(probe speakers excluded: {len(probe_spk)})")


if __name__ == "__main__":
    {"texts": texts, "prompts": prompts}[sys.argv[1]]()
