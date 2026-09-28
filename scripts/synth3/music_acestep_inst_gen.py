#!/usr/bin/env python3
"""Pool D: ACE-Step 1.5 INSTRUMENTAL fake music matched to the MTG-Jamendo real
instrumentals of strategy-v6 (genre / instrumentation / duration / container are
not label cues).

    OUT=/data/project/private/dacon-corpus/interim/acestep-inst
    # 1. CPU, once, before the GPU window (dacon311 venv): freeze the plan
    $DACON311 scripts/synth3/music_acestep_inst_gen.py plan --out $OUT --n 2400
    # 2. GPU (sing-acestep15 venv): scripts/synth3/music_acestep_inst.sbatch, one shard each
    $ACESTEP scripts/synth3/music_acestep_inst_gen.py gen --out $OUT --shard 0 --nshards 8 \\
        --stop-at 2026-09-28T13:20
    # 3. CPU (dacon311 venv), resumable, may run while 2 is still generating
    $DACON311 scripts/synth3/music_acestep_inst_gen.py finalize --out $OUT --workers 8 --threads 4

plan      one row per clip: a kept Jamendo instrumental (interim/mtg-jamendo/metadata.csv,
          `kept` rows) is drawn in seeded random order, cycling through the whole list,
          so the genre / instrument / mood distribution of the captions IS the Jamendo
          side's; its tag set becomes the caption (`caption()`), and the seed is
          drawn per clip. `--provisional` builds the same kept set from the Jamendo
          agent's _work files (AST < 0.03, no voice tag) for a smoke test only. The plan
          is frozen: it is never rewritten once any shard has logged a clip.
          A second batch (strategy-v6c, its own --out): `--extra-tracks` adds a later
          Jamendo batch's selected tracks without a voice tag (selection_tracks.csv,
          tags known before its screen), tracks are counted once each, `--id-prefix`
          keeps ids distinct and `--avoid-plan` excludes an earlier plan's seeds.
gen       ACE-Step 1.5 (turbo DiT + 5 Hz LM, the pipeline sing_acestep_gen.py uses:
          LM codes on, caption / language rewrite OFF so the recorded caption is what
          conditioned the clip). Instrumental = lyrics "[Instrumental]" (the marker the
          model is trained on; `GenerationParams.instrumental` is not read by
          generate_music) + vocal_language "unknown" + "instrumental, no vocals" in the
          caption. GEN_S (38 s) is generated and finalize keeps the MIDDLE 30 s: the
          Jamendo side is the middle 30 s of full tracks, never an intro or an ending.
          Writes `_master/<id>.flac` (the model's 48 kHz stereo, provenance) and
          `mp3/<id>.mp3`, encoded like Jamendo's `raw_30s` audio-low source files
          (ffmpeg libmp3lame, 44.1 kHz MONO, ~96 kbps ABR; measured on the source:
          5547/5762 are 44.1 kHz mono, the rest 48 kHz mono, 93-103 kbps, Lavf).
          One JSON line per clip in `gen_log.s<shard>.jsonl` (id, seed, caption, tags,
          Jamendo track, model revision, timings). Resumable (logged ids are skipped);
          stops cleanly at `--stop-at` (KST wall clock).
finalize  the deliverable, made EXACTLY as jamendo_prepare.py's extract makes the
          real side: torchaudio decode of the MP3, middle 30 s, channel mean, 16 kHz
          (torchaudio.functional.resample), a peak above 0.99 scaled to 0.99, PCM16 WAV
          at `clips/<id>.wav`. QC (docs/training/16 §5.1 and the Jamendo rules):
          exact-zero run > 20 ms or exact zeros > 5 % of the written PCM16 -> drop;
          rms < -40 dBFS -> drop; accidental vocals: the SAME AST AudioSet screen as the
          Jamendo side (max P(vocal classes) over 10 s windows >= 0.03 -> drop), so both
          sides of pool C / D pass one vocal rule. Writes gen_log.jsonl (all shards),
          qc.csv (resumable) and metadata.csv (every clip, `kept` + `drop_reason`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import time

ACESTEP_SRC = pathlib.Path("/data/project/private/dacon-src/acestep15")
MODEL_REV = {"repo": "ACE-Step/Ace-Step1.5", "hf_sha": "19671f406d603126926c1b7e2adc169acbcade22",
             "code": "ace-step/ACE-Step-1.5@ca1e85fe9430179831e6bc6be790c332190a3866",
             "dit": "acestep-v15-turbo", "lm": "acestep-5Hz-lm-1.7B"}
KST = dt.timezone(dt.timedelta(hours=9))
CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
JAMENDO = CORPUS / "interim/mtg-jamendo"
# the Jamendo side's constants (scripts/data_extra/jamendo_prepare.py), mirrored here so
# both sides pass the same rules
CLIP_S, SR = 30.0, 16000
ZERO_RUN_MS, ZERO_FRAC, RMS_MIN_CLIP = 20.0, 0.05, -40.0
AST_INSTR_MAX = 0.03
VOICE_TAGS = ("instrument---voice", "genre---rap", "genre---hiphop", "genre---singersongwriter")
AST_SNAPSHOT = next(iter(sorted((CORPUS / "raw/mtg-jamendo/_models/hf/hub/models--MIT--ast-finetuned-"
                                 "audioset-10-10-0.4593/snapshots").glob("*"))), None)
VOCAL_CLASSES = ("Speech", "Singing", "Male singing", "Female singing", "Child singing", "Choir",
                 "Yodeling", "Chant", "Mantra", "Rapping", "Humming", "A capella", "Vocal music",
                 "Synthetic singing", "Male speech, man speaking", "Female speech, woman speaking",
                 "Narration, monologue", "Whispering", "Shout", "Beatboxing")
GEN_S = 38.0          # generated; the middle CLIP_S is kept (4 s trimmed at each end)
MP3_ARGS = ["-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k", "-abr", "1"]

# ------------------------------------------------------------------ captions
#: Jamendo's run-together tag words -> caption words (anything else passes through)
WORDS = {
    "electricguitar": "electric guitar", "acousticguitar": "acoustic guitar",
    "classicalguitar": "classical guitar", "electricpiano": "electric piano",
    "drummachine": "drum machine", "doublebass": "double bass", "acousticbassguitar": "acoustic bass",
    "electronicorgan": "electronic organ", "pipeorgan": "pipe organ", "computer": "electronic production",
    "sampler": "samples", "easylistening": "easy listening", "newage": "new age",
    "instrumentalpop": "instrumental pop", "instrumentalrock": "instrumental rock",
    "drumnbass": "drum and bass", "triphop": "trip hop", "postrock": "post-rock",
    "darkambient": "dark ambient", "hardrock": "hard rock", "punkrock": "punk rock",
    "poprock": "pop rock", "classicrock": "classic rock", "bossanova": "bossa nova", "rnb": "R&B",
    "synthpop": "synth-pop", "deephouse": "deep house", "chillout": "chill-out", "idm": "IDM",
    "edm": "EDM", "videogame": "video game", "hiphop": "hip hop", "darkwave": "darkwave",
    "alternativerock": "alternative rock", "progressive": "progressive", "symphonic": "symphonic",
    "worldfusion": "world fusion", "breakbeat": "breakbeat", "eurodance": "eurodance",
}


def _word(tag: str) -> str:
    w = tag.split("---", 1)[1]
    return WORDS.get(w, w)


def caption(tags: str, r: random.Random) -> str:
    """A Jamendo tag set -> an ACE-Step caption. Every genre tag (max 3), up to 3
    instruments and 2 moods, shuffled; always 'instrumental ... no vocals'."""
    t = [x for x in str(tags or "").split("|") if "---" in x]
    g = [_word(x) for x in t if x.startswith("genre---")]
    i = [_word(x) for x in t if x.startswith("instrument---")]
    m = [_word(x) for x in t if x.startswith("mood/theme---")]
    for lst in (g, i, m):
        r.shuffle(lst)
    g, i, m = g[:3], i[:3], m[:2]
    style = ", ".join(g) if g else "music"
    parts = [f"instrumental {style}" if r.random() < 0.5 else f"{style}, instrumental"]
    if "instrumental" in style:
        parts = [style]
    if i:
        parts.append(" and ".join(i) if len(i) <= 2 else f"{i[0]}, {i[1]} and {i[2]}")
    if m:
        parts.append(" and ".join(m) + r.choice([" mood", "", " feel"]))
    parts.append(r.choice(["no vocals", "no vocals, instrumental only", "without vocals"]))
    return ", ".join(parts)


# ---------------------------------------------------------------------- plan
def kept_jamendo(provisional: bool):
    import pandas as pd
    meta = JAMENDO / "metadata.csv"
    if meta.exists():
        d = pd.read_csv(meta)
        if "kept" in d.columns:
            d = d[d["kept"].astype(str).str.lower().isin(["true", "1"])]
        return d[["track_id", "tags"]].assign(tags=lambda x: x["tags"].fillna("")), str(meta)
    if not provisional:
        raise SystemExit(f"FATAL: {meta} does not exist yet (the Jamendo build); "
                         "--provisional only for a smoke test")
    w = JAMENDO / "_work"
    a = pd.read_csv(w / "ast_tags.csv").drop_duplicates("file", keep="last")
    a["track_id"] = a["file"].str.rsplit("/", n=1).str[-1].str.replace(".wav", "", regex=False)
    t = pd.read_csv(w / "selection_tracks.csv", usecols=["track_id", "tags"])
    d = a.merge(t, on="track_id")
    d["tags"] = d["tags"].fillna("")
    voice = d["tags"].map(lambda s: any(v in s for v in VOICE_TAGS))
    return d[(d["vox_max"] < AST_INSTR_MAX) & ~voice][["track_id", "tags"]], "PROVISIONAL:" + str(w)


def plan(args) -> int:
    out = args.out
    if any(p.stat().st_size for p in out.glob("gen_log.s*.jsonl")) and not args.dry:
        raise SystemExit(f"FATAL: {out} already has generated clips; the plan is frozen")
    d, src = kept_jamendo(args.provisional)
    for x in args.extra_tracks:
        import pandas as pd
        t = pd.read_csv(x, usecols=["track_id", "tags", "selected"])
        t = t[t["selected"].astype(bool)].assign(tags=lambda y: y["tags"].fillna(""))
        t = t[~t["tags"].map(lambda s: any(v in s for v in VOICE_TAGS))]
        d, src = pd.concat([d, t[["track_id", "tags"]]], ignore_index=True), f"{src}+{x}"
    if args.extra_tracks:
        d = d.drop_duplicates("track_id")
    d = d.sort_values("track_id").reset_index(drop=True)
    avoid = set()
    if args.avoid_plan:
        avoid = {json.loads(ln)["seed"] for ln in pathlib.Path(args.avoid_plan).read_text().splitlines()
                 if ln.strip()}
    r = random.Random(args.seed)
    rows, order = [], []
    while len(rows) < args.n:
        if not order:
            order = list(range(len(d)))
            r.shuffle(order)
        j = d.iloc[order.pop()]
        s = r.randrange(1, 2**31 - 1)
        while s in avoid:
            s = r.randrange(1, 2**31 - 1)
        i = len(rows)
        rows.append({"id": f"{args.id_prefix}{i:05d}", "seed": s, "jamendo_track_id": j["track_id"],
                     "tags": j["tags"], "caption": caption(j["tags"], random.Random(s)),
                     "gen_s": GEN_S, "clip_s": CLIP_S})
    genres = {}
    for x in rows:
        for t in x["tags"].split("|"):
            if t.startswith("genre---"):
                genres[t[8:]] = genres.get(t[8:], 0) + 1
    top = sorted(genres.items(), key=lambda kv: -kv[1])[:12]
    print(f"{len(rows)} clips from {len(d)} kept Jamendo instrumentals ({src}); top genres {top}")
    for x in rows[:5]:
        print(" ", x["jamendo_track_id"], "|", x["caption"])
    if args.dry:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    (out / "plan.jsonl").write_text(body)
    (out / "plan_source.json").write_text(json.dumps({
        "source": src, "n_jamendo": len(d), "n": len(rows), "seed": args.seed,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "created": dt.datetime.now(KST).isoformat(timespec="seconds")}, indent=1))
    print(f"wrote {out / 'plan.jsonl'}")
    return 0


# ----------------------------------------------------------------------- gen
def gen(args) -> int:
    rows = [json.loads(ln) for ln in (args.out / "plan.jsonl").read_text().splitlines() if ln.strip()]
    mine = [x for i, x in enumerate(rows) if i % args.nshards == args.shard]
    if args.limit:
        mine = mine[:args.limit]
    log = args.out / f"gen_log.s{args.shard}.jsonl"
    have = set()
    if log.exists():
        have = {json.loads(ln)["id"] for ln in log.read_text().splitlines() if ln.strip()}
    todo = [x for x in mine if x["id"] not in have]
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} planned, {len(todo)} to do", flush=True)
    stop = dt.datetime.fromisoformat(args.stop_at).replace(tzinfo=KST)
    if not todo or dt.datetime.now(KST) >= stop:
        return 0

    sys.path.insert(0, str(ACESTEP_SRC))
    os.chdir(ACESTEP_SRC)
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
    ck = os.environ["ACESTEP_CHECKPOINTS_DIR"]
    dit, llm = AceStepHandler(), LLMHandler()
    print(dit.initialize_service(project_root=str(pathlib.Path(ck).parent),
                                 config_path=MODEL_REV["dit"], device=args.device), flush=True)
    print(llm.initialize(checkpoint_dir=ck, lm_model_path=MODEL_REV["lm"],
                         backend="vllm" if args.device == "cuda" else "pt", device=args.device), flush=True)
    tmp = args.out / "_tmp" / f"s{args.shard}"
    for x in todo:
        if dt.datetime.now(KST) >= stop:
            print("stop-at reached", flush=True)
            break
        t0 = time.time()
        dur = args.gen_s or x["gen_s"]
        params = GenerationParams(caption=x["caption"], lyrics="[Instrumental]", instrumental=True,
                                  vocal_language="unknown", duration=dur, seed=x["seed"],
                                  thinking=True, use_cot_caption=False, use_cot_language=False)
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[x["seed"]],
                               audio_format="flac")
        res = generate_music(dit, llm, params, cfg, save_dir=str(tmp))
        if not res.success or not res.audios:
            print(f"FAIL {x['id']}: {res.error}", flush=True)
            continue
        master = args.out / "_master" / f"{x['id']}.flac"
        master.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pathlib.Path(res.audios[0]["path"]), master)
        mp3 = args.out / "mp3" / f"{x['id']}.mp3"
        mp3.parent.mkdir(parents=True, exist_ok=True)
        part = mp3.with_suffix(".part.mp3")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(master), *MP3_ARGS,
                        str(part)], check=True)
        os.replace(part, mp3)
        lm = (res.extra_outputs or {}).get("lm_metadata")
        p = lm[0] if isinstance(lm, list) and lm else lm
        rec = {**x, "gen_s": dur, "shard": args.shard, "file_mp3": f"mp3/{x['id']}.mp3",
               "master": f"_master/{x['id']}.flac", "model": MODEL_REV, "lyrics": "[Instrumental]",
               "wall_s": round(time.time() - t0, 1),
               "lm_metas": {k: p.get(k) for k in ("bpm", "keyscale", "timesignature", "duration")}
               if isinstance(p, dict) else {},   # the LM's CoT metas (extra_outputs)
               "created": dt.datetime.now(KST).isoformat(timespec="seconds")}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"ok {x['id']} {dur:.0f}s in {rec['wall_s']}s", flush=True)
    return 0


# ------------------------------------------------------------------ finalize
def _canon_one(rec: dict) -> dict:
    """jamendo_prepare.py's _extract_one on our MP3 (the same decode / cut / canonical)."""
    import io

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.functional as AF
    torch.set_num_threads(1)
    out = {"id": rec["id"], "ok": False}
    try:
        root = pathlib.Path(rec["_out"])
        wav, sr = torchaudio.load(io.BytesIO((root / rec["file_mp3"]).read_bytes()), format="mp3")
        dur = wav.shape[1] / sr
        off = max(0.0, (dur - CLIP_S) / 2)
        a, n = int(round(off * sr)), int(round(CLIP_S * sr))
        seg = wav[:, a:a + n].mean(0)
        if sr != SR:
            seg = AF.resample(seg, sr, SR)
        y = seg.numpy().astype(np.float64)
        peak = float(np.abs(y).max())
        if peak > 0.99:
            y = y * (0.99 / peak)
        pcm = np.round(y * 32767).astype(np.int16)
        z = pcm == 0
        run = 0
        if z.any():
            dz = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
            run = int((np.flatnonzero(dz == -1) - np.flatnonzero(dz == 1)).max())
        spec = np.abs(np.fft.rfft(y[: len(y) // 2048 * 2048].reshape(-1, 2048) * np.hanning(2048), axis=1)) ** 2
        cum = np.cumsum(spec.sum(0))
        roll = float(np.searchsorted(cum, 0.99 * cum[-1]) * SR / 2048)
        rel = f"clips/{rec['id']}.wav"
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part.wav")
        sf.write(tmp, pcm, SR, subtype="PCM_16")
        tmp.replace(dst)
        rms = float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))
        br = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of",
                             "default=nw=1:nk=1", str(root / rec["file_mp3"])],
                            capture_output=True, text=True).stdout.strip()
        out.update(ok=True, file=rel, src_sr=sr, src_channels=int(wav.shape[0]),
                   mp3_kbps=round(int(br) / 1000, 1) if br.isdigit() else None,
                   gen_track_s=round(dur, 2), offset_s=round(off, 3), duration_s=round(len(pcm) / SR, 3),
                   rms_dbfs=round(20 * np.log10(rms + 1e-12), 2), peak_src=round(peak, 4),
                   zero_run_ms=round(run / SR * 1000, 2), zero_frac=round(float(z.mean()), 6),
                   rolloff99_hz=round(roll, 1))
    except Exception as exc:                                  # noqa: BLE001 -- recorded
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def _ast_part(files: list[str], root: str, threads: int) -> list[dict]:
    """jamendo_prepare.py's AST screen: per 10 s window the max sigmoid over VOCAL_CLASSES."""
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    torch.set_num_threads(threads)
    fe = ASTFeatureExtractor.from_pretrained(AST_SNAPSHOT)
    model = ASTForAudioClassification.from_pretrained(AST_SNAPSHOT).eval()
    lab = model.config.label2id
    vid = [lab[c] for c in VOCAL_CLASSES]
    out = []
    for f in files:
        y, _ = sf.read(pathlib.Path(root) / f, dtype="float32")
        wins = [y[k * 160000:(k + 1) * 160000] for k in range(max(1, len(y) // 160000))]
        feats = fe(wins, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            p = torch.sigmoid(model(feats).logits).numpy()
        v = p[:, vid].max(1)
        out.append({"file": f, "vox_win": ";".join(f"{x:.4f}" for x in v), "vox_max": float(v.max()),
                    "sing_max": float(p[:, lab["Singing"]].max()),
                    "speech_max": float(p[:, lab["Speech"]].max()),
                    "music_min": float(p[:, lab["Music"]].min())})
    return out


def qc_reason(r) -> str:
    """The drop rule (the Jamendo side's, in its order); '' = kept."""
    if not r.get("ok"):
        return "decode-failed"
    if r["zero_run_ms"] > ZERO_RUN_MS or r["zero_frac"] > ZERO_FRAC:
        return "exact-zero"
    if r["rms_dbfs"] < RMS_MIN_CLIP:
        return "near-silent"
    if r.get("vox_max") != r.get("vox_max") or r.get("vox_max") is None:
        return "untagged"
    if r["vox_max"] >= AST_INSTR_MAX:
        return "ast-vocal-or-ambiguous"
    return ""


def segments(kept, out: pathlib.Path):
    """Each kept 30 s clip -> consecutive 10 s rows (seg10/<id>_<k>.wav, PCM16 16 kHz),
    exactly as the Jamendo side (scripts/data_extra/jamendo_prepare.py _segments). Why:
    the music draw has ONE bucket per side and tiles it uniformly over the rows that fit
    (processing.sampler.bucket_keys / _tiles), so a corpus's share of music tiles is its
    row count; draw.domain_weights cannot move it. A near-silent segment is dropped."""
    import numpy as np
    import pandas as pd
    import soundfile as sf
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "strategy"))
    from cut_music_rows import pieces_of      # the ONE music cut rule (>= 15 s -> 10 s pieces)
    rows = []
    for r in kept.to_dict("records"):
        y, sr = sf.read(out / r["file"], dtype="int16")
        assert sr == SR, (r["file"], sr)
        for k, (off, length) in enumerate(pieces_of(len(y) / SR)):
            c = y[int(round(off * SR)):int(round((off + length) * SR))]
            rms = float(np.sqrt(np.mean((c / 32768.0) ** 2)))
            if 20 * np.log10(rms + 1e-12) < RMS_MIN_CLIP:
                continue
            rel = f"seg10/{r['id']}_{k}.wav"
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                sf.write(dst.with_suffix(".part.wav"), c, SR, subtype="PCM_16")
                dst.with_suffix(".part.wav").replace(dst)
            rows.append({**r, "clip": r["file"], "seg": k, "file": rel, "file_id": f"{r['file_id']}_{k}",
                         "duration_s": len(c) / SR, "offset_s": r["offset_s"] + off})
    return pd.DataFrame(rows)


def finalize(args) -> int:
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    out = args.out
    recs = {}
    for p in sorted(out.glob("gen_log.s*.jsonl")):
        for ln in p.read_text().splitlines():
            if ln.strip():
                x = json.loads(ln)
                recs[x["id"]] = x
    recs = [recs[k] for k in sorted(recs)]
    (out / "gen_log.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
    qc_path, ast_path = out / "qc.csv", out / "ast_tags.csv"
    have = set(pd.read_csv(qc_path)["id"]) if qc_path.exists() else set()
    todo = [{**x, "_out": str(out)} for x in recs if x["id"] not in have]
    print(f"{len(recs)} generated, {len(todo)} to canonicalise", flush=True)
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_canon_one, todo, chunksize=8))
        pd.DataFrame(res).to_csv(qc_path, mode="a", header=not qc_path.exists(), index=False)
    qc = pd.read_csv(qc_path).drop_duplicates("id", keep="last")
    tagged = set(pd.read_csv(ast_path)["file"]) if ast_path.exists() else set()
    files = [f for f in qc.loc[qc["ok"].astype(bool), "file"] if f not in tagged]
    print(f"{len(files)} clips to AST-tag", flush=True)
    if files and not args.no_ast:
        if AST_SNAPSHOT is None:
            raise SystemExit("FATAL: the AST snapshot (the Jamendo screen's model) is missing")
        procs = max(1, args.workers // args.threads)
        step = 60 * procs
        for i in range(0, len(files), step):
            chunk = files[i:i + step]
            parts = [chunk[k::procs] for k in range(procs)]
            with ProcessPoolExecutor(procs) as ex:
                res = [r for part in ex.map(_ast_part, parts, [str(out)] * procs, [args.threads] * procs)
                       for r in part]
            pd.DataFrame(res).to_csv(ast_path, mode="a", header=not ast_path.exists(), index=False)
            print(f"  {i + len(chunk)}/{len(files)}", flush=True)
    g = pd.DataFrame(recs)
    m = g.merge(qc, on="id", how="left")
    if ast_path.exists():
        m = m.merge(pd.read_csv(ast_path).drop_duplicates("file", keep="last"), on="file", how="left")
    else:
        m["vox_max"] = float("nan")
    m["drop_reason"] = [qc_reason(r) for r in m.to_dict("records")]
    m["kept"] = m["drop_reason"].eq("")
    for k in ("repo", "hf_sha", "code", "dit", "lm"):
        m[f"model_{k}"] = m["model"].map(lambda d, k=k: d.get(k) if isinstance(d, dict) else None)
    m["file_id"] = f"{out.name}:" + m["id"]      # the corpus name (acestep-inst, acestep-inst-2)
    cols = ["file", "file_id", "id", "kept", "drop_reason", "duration_s", "seed", "caption", "tags",
            "jamendo_track_id", "gen_s", "offset_s", "gen_track_s", "src_sr", "src_channels", "mp3_kbps",
            "rms_dbfs", "peak_src", "zero_run_ms", "zero_frac", "rolloff99_hz", "vox_max", "sing_max",
            "speech_max", "music_min", "file_mp3", "master", "lm_metas", "model_repo", "model_hf_sha",
            "model_code", "model_dit", "model_lm", "shard", "wall_s", "created"]
    m = m.reindex(columns=cols).sort_values("id")
    print(m["drop_reason"].replace("", "kept").value_counts().to_string())
    k = segments(m[m["kept"]], out)
    m = pd.concat([m[~m["kept"]].assign(clip=m["file"], seg=pd.NA), k], ignore_index=True)
    m = m.reindex(columns=["file", "clip", "seg"] + cols[1:]).sort_values(["id", "seg"])
    m.to_csv(out / "metadata.csv", index=False)
    print(f"kept {k['clip'].nunique()} / {m['id'].nunique()} clips -> {len(k)} 10 s rows, "
          f"{k['duration_s'].sum() / 3600:.2f} h -> {out / 'metadata.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["plan", "gen", "finalize"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=2400, help="plan: clips (2,400 x 30 s = 20 h)")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--id-prefix", default="acestep15inst_", help="plan: clip id prefix")
    ap.add_argument("--extra-tracks", nargs="*", default=[],
                    help="plan: later Jamendo batches' selection_tracks.csv (selected, no voice tag)")
    ap.add_argument("--avoid-plan", default=None, help="plan: an earlier plan.jsonl whose seeds are not reused")
    ap.add_argument("--provisional", action="store_true", help="plan: Jamendo _work files (smoke only)")
    ap.add_argument("--dry", action="store_true", help="plan: print, write nothing")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stop-at", default="2026-09-28T13:20")
    ap.add_argument("--device", default="cuda", help="gen: cuda, or cpu for a smoke test")
    ap.add_argument("--gen-s", type=float, default=0.0, help="gen: override the plan's length (smoke)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4, help="finalize: torch threads per AST process")
    ap.add_argument("--no-ast", action="store_true", help="finalize: skip the vocal screen (untagged = dropped)")
    a = ap.parse_args()
    return {"plan": plan, "gen": gen, "finalize": finalize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
