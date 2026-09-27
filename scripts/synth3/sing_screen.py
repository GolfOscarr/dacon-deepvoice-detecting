#!/usr/bin/env python3
"""N2: measure and screen the vocal chunks; write sing-fake / sing-real metadata.csv.

    # 1. measure (GPU for Whisper LID, CPU threads for Silero); cached, rerunnable
    python scripts/synth3/sing_screen.py measure --work <work> [--threads 6]
    # 2. apply the rule, write metadata (dry run unless --apply moves drops)
    python scripts/synth3/sing_screen.py apply --work <work> [--apply]

Every chunk logged in `<work>/sep_*_*.csv` is measured the same way on both
sides (16 kHz mono chunks already):

  speech_ratio     fraction of 32 ms frames with Silero P(speech) >= 0.5
                   (the repo's vendored model, as scripts/data_extra/screen_stems.py)
  speech_prob_max  max Silero P(speech)
  lid, lid_p       Whisper large-v3 language id on the first 30 s
  no_speech_p      Whisper's <|nospeech|> probability at the same step

The SAME rule on both sides (`RULE` below):
  rms_dbfs >= -40         near-silent stem
  vox_rel_db >= -15       stem is not just leakage (energy vs the canonical mix)
  speech_ratio >= 0.20    Silero hears a voice in >= 20 % of frames
  lid_p >= 0.50           Whisper is sure of a language (a sung voice, not bleed)
  lid not in {zh, ja}     owner rule: no Chinese / Japanese audio

Whisper's no_speech_p is recorded but not used: on sung vocals it is high for
Suno stems far more often than Silero misses them (not a like-for-like screen).
Then balance: each SONICS version keeps at most CAP_H hours, whole source songs
taken in a seeded order ("cap"); ACE-Step and real stems are not capped.

Output: `<corpus>/interim/sing-fake/metadata.csv` (generator_version column)
and `<corpus>/interim/sing-real/metadata.csv` (corpus column), the
sonics-sep / realmusic-sep schema plus: seg_start_s, excerpt_s, active_ratio,
vox_rel_db, lid, lid_p, lang, source_name. With `--apply`, dropped chunks move
to `<root>/_dropped/<same rel path>`.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
ROOTS = {"fake": CORPUS / "interim/sing-fake", "real": CORPUS / "interim/sing-real"}
GROUP_COL = {"fake": "generator_version", "real": "corpus"}
RMS_MIN, REL_MIN, SPEECH_MIN, LIDP_MIN, LID_DENY = -40.0, -15.0, 0.20, 0.50, ("zh", "ja")
#: kept hours per SONICS version (balance; whole songs beyond it drop as "cap")
CAP_H = 4.0
WHISPER = "openai/whisper-large-v3"
LID_LANGS = ["en", "ko", "zh", "ja", "es", "fr", "de", "pt", "it", "ru", "nl", "pl", "sv", "tr",
             "ar", "hi", "id", "vi", "th", "uk", "cs", "el", "fi", "he", "hu", "ro", "da", "no"]
SONICS_FAMILY = {"chirp-v2-xxl-alpha": "suno_chirp", "chirp-v3": "suno_chirp",
                 "chirp-v3.5": "suno_chirp", "udio-30s": "udio", "udio-120s": "udio"}
_tl = threading.local()


def locate(root: pathlib.Path, rel: str) -> pathlib.Path | None:
    for p in (root / rel, root / "_dropped" / rel):
        if p.exists():
            return p
    return None


def chunks(work: pathlib.Path) -> pd.DataFrame:
    sep = pd.concat([pd.read_csv(p) for p in sorted(work.glob("sep_*_*.csv"))], ignore_index=True)
    sep = sep[sep["sep_ok"].astype(str) == "True"]
    sep = sep[sep["file"].notna()].drop_duplicates(["side", "file"], keep="last")
    # a re-separated source may leave log rows for chunks that no longer exist
    ok = [locate(ROOTS[sd], f) is not None for sd, f in zip(sep["side"], sep["file"])]
    return sep[ok]


def silero(path: pathlib.Path) -> dict:
    from models.vendor.silero_vad import SAMPLE_RATE, speech_probabilities
    y, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE, (path, sr)
    p = speech_probabilities(y, SAMPLE_RATE)
    return {"speech_ratio": float((p >= 0.5).mean()) if p.size else 0.0,
            "speech_prob_max": float(p.max()) if p.size else 0.0}


def measure(args) -> int:
    torch.set_num_threads(1)
    ch = chunks(args.work)
    cache_path = args.work / "screen_cache.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["side", "file"])
    have = set(zip(cache["side"], cache["file"]))
    todo = [r for r in ch.itertuples(index=False) if (r.side, r.file) not in have]
    print(f"{len(ch)} chunks, {len(todo)} to measure", flush=True)
    if not todo:
        return 0
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
    dt = torch.float16 if args.device == "cuda" else torch.float32
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
    tok = WhisperTokenizer.from_pretrained(WHISPER)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER, dtype=dt).to(args.device).eval()
    nospeech = tok.convert_tokens_to_ids("<|nospeech|>")
    lang_ids = [tok.convert_tokens_to_ids(f"<|{l}|>") for l in LID_LANGS]
    start = torch.tensor([[model.config.decoder_start_token_id]], device=args.device)

    def load(r):
        p = locate(ROOTS[r.side], r.file)
        y, _ = sf.read(str(p), dtype="float32")
        return r, p, y

    pool = ThreadPoolExecutor(args.threads)
    rows = []
    B = 32
    for i in range(0, len(todo), B):
        part = list(pool.map(load, todo[i:i + B]))
        vads = pool.map(lambda t: silero(t[1]), part)
        feats = fe([y[:30 * 16000] for _, _, y in part], sampling_rate=16000,
                   return_tensors="pt").input_features.to(args.device, dt)
        with torch.no_grad():
            enc = model.model.encoder(feats)
            logits = model(encoder_outputs=enc, decoder_input_ids=start.expand(len(part), 1)).logits[:, -1]
            pr = logits[:, lang_ids].float().softmax(-1).cpu().numpy()
            nsp = logits.float().softmax(-1)[:, nospeech].cpu().numpy()
        for (r, _, _), v, p, q in zip(part, vads, pr, nsp):
            rows.append({"side": r.side, "file": r.file, **v, "no_speech_p": round(float(q), 4),
                         "lid": LID_LANGS[int(p.argmax())], "lid_p": round(float(p.max()), 3)})
        if (i // B) % 50 == 0:
            print(f"  {i + len(part)}/{len(todo)}", flush=True)
            pd.concat([cache, pd.DataFrame(rows)]).to_csv(cache_path, index=False)
    pool.shutdown()
    pd.concat([cache, pd.DataFrame(rows)]).to_csv(cache_path, index=False)
    print(f"measured {len(rows)}")
    return 0


def fma_lang() -> pd.Series:
    t = pd.read_csv(CORPUS / "interim/fma/fma_small/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
    return t[("track", "language_code")]


def apply(args) -> int:
    ch = chunks(args.work)
    cache = pd.read_csv(args.work / "screen_cache.csv").drop_duplicates(["side", "file"], keep="last")
    m = ch.merge(cache, on=["side", "file"], how="left")
    sel = pd.concat([pd.read_csv(p).assign(_sel=p.name) for p in sorted(args.work.glob("selection*.csv"))])
    sel = sel.drop_duplicates("file_id")[["file_id", "speaker_ref_id", "source_name"]]
    m = m.merge(sel, left_on="source_file_id", right_on="file_id", how="left").drop(columns="file_id")
    reasons = [
        (m["speech_ratio"].isna(), "unmeasured"),
        (m["rms_dbfs"] < RMS_MIN, f"rms<{RMS_MIN:g}"),
        (m["vox_rel_db"] < REL_MIN, f"vox_rel<{REL_MIN:g}"),
        (m["speech_ratio"] < SPEECH_MIN, f"speech_ratio<{SPEECH_MIN:g}"),
        (m["lid_p"] < LIDP_MIN, f"lid_p<{LIDP_MIN:g}"),
        (m["lid"].isin(LID_DENY), "lid_zh_ja"),
    ]
    m["drop_reason"] = ""
    for mask, why in reasons:
        m.loc[mask.fillna(False).to_numpy(), "drop_reason"] += ";" + why
    m["drop_reason"] = m["drop_reason"].str.strip(";")
    m["kept"] = m["drop_reason"] == ""
    sonics = (m["side"] == "fake") & ~m["group"].astype(str).str.startswith("acestep-")
    for g, d in m[sonics & m["kept"]].groupby("group"):
        songs = d.groupby("source_file_id")["duration_s"].sum()
        songs = songs.iloc[np.random.default_rng(0).permutation(len(songs))]
        over = songs.index[(songs.cumsum() / 3600) > CAP_H]
        cap = m["source_file_id"].isin(over) & (m["group"] == g) & m["kept"]
        m.loc[cap, "drop_reason"] = "cap"
        m.loc[cap, "kept"] = False

    for c in ("speaker_ref_id", "source_name", "lid"):
        m[c] = m[c].astype(object)
    fake = m["side"] == "fake"
    # an artist name with no [a-z0-9] has no name atom: processing.corpus falls back to the id atom
    m.loc[~fake, "speaker_ref_id"] = m.loc[~fake, "speaker_ref_id"].fillna(m.loc[~fake, "source_name"])
    grp = m["group"].astype(str)
    ace = grp.str.startswith("acestep-")
    m["artifact_family"] = np.where(ace, "acestep15", grp.map(SONICS_FAMILY))
    m.loc[~fake, "artifact_family"] = None
    m.loc[fake & ~ace, "speaker_ref_id"] = "sonics/" + grp[fake & ~ace]
    m.loc[ace, "speaker_ref_id"] = "aisong-kr-en/acestep15"
    # lang: ACE-Step by its lyrics, FMA by its track metadata, else Whisper LID
    m["lang"] = m["lid"]
    m.loc[ace, "lang"] = grp[ace].str.replace("acestep-", "", regex=False)
    tid = pd.to_numeric(m["source_file_id"].str.extract(r"(\d+)\.mp3$")[0], errors="coerce")
    fl = tid.map(fma_lang())
    isfma = (~fake) & (grp == "fma")
    m.loc[isfma & fl.notna(), "lang"] = fl[isfma & fl.notna()]

    cols = ["file", "source_path", "source_file_id", "offset_s", "duration_s", "sample_rate",
            "speech_ratio", "rms_dbfs", "kept", "drop_reason", "artifact_family", "speaker_ref_id",
            "src_sr", "src_channels", "peak", "speech_prob_max", "seg_start_s", "excerpt_s",
            "active_ratio", "vox_rel_db", "lid", "lid_p", "lang", "source_name"]
    for side, root in ROOTS.items():
        d = m[m["side"] == side].copy()
        if d.empty:
            continue
        files = []
        for r in d.itertuples(index=False):
            cur = locate(root, r.file)
            want = root / r.file if r.kept else root / "_dropped" / r.file
            if args.apply and cur is not None and cur != want:
                want.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cur), str(want))
                cur = want
            files.append(str(cur.relative_to(root)) if cur else "")
        d["file"] = files
        d = d.rename(columns={"group": GROUP_COL[side]})
        out = d.sort_values("file")[cols[:3] + [GROUP_COL[side]] + cols[3:]]
        out.to_csv(root / "metadata.csv", index=False)
        g = d.groupby(GROUP_COL[side])
        print(f"\n== {side}: {root}/metadata.csv ({len(d)} chunks)")
        print(pd.DataFrame({
            "chunks": g.size(), "hours": g["duration_s"].sum() / 3600,
            "kept": g["kept"].sum(), "kept_h": g.apply(lambda x: x.loc[x.kept, "duration_s"].sum() / 3600),
            **{why: g.apply(lambda x, w=why: x.drop_reason.str.contains(w, regex=False).sum())
               for _, why in reasons[1:] + [(None, "cap")]},
            "songs_kept": g.apply(lambda x: x.loc[x.kept, "source_file_id"].nunique()),
            "spk_kept": g.apply(lambda x: x.loc[x.kept, "speaker_ref_id"].nunique()),
        }).round(2).to_string())
        print("lang (kept, h):", (d[d.kept].groupby("lang")["duration_s"].sum() / 3600).round(2)
              .sort_values(ascending=False).head(8).to_dict())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["measure", "apply"])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-base", type=pathlib.Path, default=None,
                    help="stems live under <dir>/sing-fake and <dir>/sing-real (as sing_separate.py)")
    args = ap.parse_args()
    if args.out_base:
        ROOTS.update({"fake": args.out_base / "sing-fake", "real": args.out_base / "sing-real"})
    return measure(args) if args.cmd == "measure" else apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
