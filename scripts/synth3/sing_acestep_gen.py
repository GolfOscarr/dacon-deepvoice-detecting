#!/usr/bin/env python3
"""N2c: AI songs with Korean and English lyrics from ACE-Step 1.5 (docs/training/15 §3).

    ACESTEP_CHECKPOINTS_DIR=/data/project/private/dacon-weights/sing/acestep15/checkpoints \\
    /data/project/private/dacon-venvs/sing-acestep15/bin/python scripts/synth3/sing_acestep_gen.py \\
        --out /data/project/private/dacon-corpus/interim/aisong-kr-en --hours 10 --stop-at 2026-09-27T12:40

Lyrics are LLM-free: a seeded template grammar over fixed word banks (below),
so every song's text is reproducible from its seed and carries no third-party
copyright. The caption (style prompt) is drawn the same way. The LM's
caption / language rewrite (CoT) is OFF so the recorded caption and lyrics are
exactly what conditioned the song.

Per song: `<out>/_master/<lang>/<id>.flac` (the model's 48 kHz output, kept
for provenance) and the whole-file deliverable `<out>/<lang>/<id>.mp3`
(44.1 kHz stereo 192 kbps, the container FMA's real songs have, so the file
format is no label cue). One JSON line per song in `<out>/gen_log.jsonl`:
id, lang, seed, caption, lyrics, duration, model revision, timings. Songs are
drawn ko:en = 60:40 by hours. Rerunnable: ids already logged are skipped.
Stops cleanly at `--stop-at` (KST wall clock).
"""
from __future__ import annotations

import argparse
import datetime as dt
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

# ----------------------------------------------------------------- word banks
KO = {
    "time": ["새벽", "아침", "한낮", "저녁", "밤", "봄날", "여름밤", "가을", "겨울", "비 오는 날", "첫눈 오는 날", "주말"],
    "place": ["골목길", "바닷가", "강가", "옥상", "정류장", "창가", "기차역", "언덕", "작은 방", "공원", "교실", "횡단보도"],
    "noun": ["바람", "별빛", "노을", "꽃잎", "빗소리", "달빛", "파도", "그림자", "약속", "편지", "기억", "노래", "하늘", "구름", "눈물", "웃음"],
    "feel": ["그리움", "설렘", "외로움", "기쁨", "사랑", "용기", "희망", "추억", "미소"],
    "adj": ["따뜻한", "차가운", "눈부신", "조용한", "희미한", "푸른", "하얀", "빛나는", "작은", "깊은"],
    "verb": ["걸어가", "노래해", "달려가", "꿈을 꿔", "웃어봐", "춤을 춰", "날아가", "숨을 쉬어"],
    "tverb": ["기다려", "불러봐", "안아줘", "찾아가", "그려봐", "떠올려"],
    "you": ["너", "그대", "우리"],
}
EN = {
    "time": ["morning", "midnight", "the evening", "the summer", "the winter", "a rainy day", "the weekend", "sunrise", "the autumn", "tonight"],
    "place": ["the station", "the river", "the rooftop", "the city lights", "the empty street", "the ocean", "the old road", "your window", "the park", "the highway"],
    "noun": ["wind", "stars", "rain", "moonlight", "paper heart", "waves", "shadow", "promise", "letter", "memory", "song", "sky"],
    "feel": ["love", "hope", "courage", "feeling", "fire", "light", "silence", "dream"],
    "adj": ["golden", "quiet", "shining", "broken", "gentle", "endless", "restless", "silver", "wild", "tender"],
    "verb": ["run", "wait", "sing", "dance", "hold on", "fall", "dream", "fly", "stay", "call your name"],
}


def _batchim(word: str) -> bool:
    ch = word.strip()[-1]
    return 0xAC00 <= ord(ch) <= 0xD7A3 and (ord(ch) - 0xAC00) % 28 != 0


def _j(word: str, pair: str) -> str:
    """Korean particle: pair 'with/without batchim' given as 'with/without'."""
    a, b = pair.split("/")
    return word + (a if _batchim(word) else b)


def ko_line(r: random.Random) -> str:
    c = {k: r.choice(v) for k, v in KO.items()}
    pats = [
        lambda: f"{c['time']}의 {c['place']}에서 {_j(c['you'], '을/를')} {c['tverb']}",
        lambda: f"{c['adj']} {_j(c['noun'], '이/가')} 내게 속삭여",
        lambda: f"{_j(c['noun'], '을/를')} 따라 {c['place']}까지 {c['verb']}",
        lambda: f"{c['adj']} {_j(c['feel'], '이/가')} 가슴에 남아",
        lambda: f"{c['you']}와 함께라면 {c['time']}도 괜찮아",
        lambda: f"{_j(c['noun'], '처럼/처럼')} {c['adj']} {c['you']}의 목소리",
        lambda: f"오늘도 {c['place']}에 서서 {_j(c['feel'], '을/를')} 불러",
        lambda: f"{c['time']}이 오면 다시 {c['verb']}",
        lambda: f"{c['adj']} {c['noun']} 사이로 {c['you']}가 보여",
        lambda: f"잊지 마 우리의 {c['adj']} {c['feel']}",
    ]
    return r.choice(pats)()


def en_line(r: random.Random) -> str:
    c = {k: r.choice(v) for k, v in EN.items()}
    pats = [
        lambda: f"Every {c['time'].replace('the ', '')} by {c['place']} I {c['verb']}",
        lambda: f"The {c['noun']} is {c['adj']} tonight",
        lambda: f"I follow the {c['noun']} down to {c['place']}",
        lambda: f"There's a {c['adj']} {c['feel']} inside of me",
        lambda: f"With you even {c['time']} feels right",
        lambda: f"Your voice is {c['adj']} like the {c['noun']}",
        lambda: f"Standing at {c['place']} I {c['verb']} again",
        lambda: f"When {c['time']} comes around we {c['verb']}",
        lambda: f"Through the {c['adj']} {c['noun']} I can see you",
        lambda: f"Don't forget our {c['adj']} {c['feel']}",
    ]
    return r.choice(pats)()


def lyrics(lang: str, r: random.Random, duration: float) -> str:
    line = ko_line if lang == "ko" else en_line
    n_verse = 4 if duration <= 100 else (6 if duration <= 150 else 8)
    chorus = [line(r) for _ in range(4)]
    parts = [("[Verse 1]", [line(r) for _ in range(n_verse // 2 + 2)]), ("[Chorus]", chorus)]
    if duration > 80:
        parts += [("[Verse 2]", [line(r) for _ in range(n_verse // 2 + 2)]), ("[Chorus]", chorus)]
    if duration > 140:
        parts += [("[Bridge]", [line(r) for _ in range(2)]), ("[Chorus]", chorus)]
    return "\n\n".join(tag + "\n" + "\n".join(ls) for tag, ls in parts)


GENRES = {
    "ko": ["K-pop dance pop", "Korean ballad", "Korean indie folk", "K-pop R&B", "Korean rock",
           "Korean trot", "Korean hip-hop", "city pop", "Korean acoustic pop", "K-pop synth-pop"],
    "en": ["pop", "rock", "country", "folk", "R&B", "synth-pop", "indie rock", "soul",
           "hip-hop", "dance pop with vocals"],
}
INSTR = ["piano", "acoustic guitar", "electric guitar", "synthesizer", "strings", "drums and bass",
         "808 drums", "brass section", "electric piano", "orchestral pads"]
MOOD = ["emotional", "upbeat", "melancholic", "energetic", "dreamy", "warm", "bright", "nostalgic"]
VOICE = ["female vocal", "male vocal", "female vocal", "male vocal", "duet vocals"]
LANG_NAME = {"ko": "Korean", "en": "English"}


def caption(lang: str, r: random.Random) -> str:
    return (f"{r.choice(GENRES[lang])}, {r.choice(MOOD)}, {' and '.join(r.sample(INSTR, 2))}, "
            f"{r.choice(VOICE)}, {LANG_NAME[lang]} lyrics")


def plan(n_hours: float, seed: int) -> list[dict]:
    r = random.Random(seed)
    out, tot = [], {"ko": 0.0, "en": 0.0}
    target = {"ko": 0.6 * n_hours * 3600, "en": 0.4 * n_hours * 3600}
    i = 0
    while any(tot[l] < target[l] for l in tot):
        lang = "ko" if tot["ko"] / target["ko"] <= tot["en"] / target["en"] else "en"
        dur = float(r.choice([60, 90, 120, 150, 180]))
        s = r.randrange(1, 2**31 - 1)
        rr = random.Random(s)
        out.append({"id": f"acestep15_{lang}_{i:04d}", "lang": lang, "seed": s, "duration": dur,
                    "caption": caption(lang, rr), "lyrics": lyrics(lang, rr, dur)})
        tot[lang] += dur; i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stop-at", default="2026-09-27T12:40")
    ap.add_argument("--dry", action="store_true", help="print the plan only")
    args = ap.parse_args()
    songs = plan(args.hours, args.seed)
    if args.limit:
        songs = songs[:args.limit]
    if args.dry:
        for s in songs[:4]:
            print(json.dumps(s, ensure_ascii=False, indent=1))
        print(len(songs), "songs")
        return 0
    stop = dt.datetime.fromisoformat(args.stop_at).replace(tzinfo=KST)
    log = args.out / "gen_log.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    have = set()
    if log.exists():
        have = {json.loads(l)["id"] for l in log.read_text().splitlines() if l.strip()}
    todo = [s for s in songs if s["id"] not in have]
    print(f"{len(songs)} planned, {len(todo)} to do", flush=True)

    sys.path.insert(0, str(ACESTEP_SRC))
    os.chdir(ACESTEP_SRC)
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
    ck = os.environ["ACESTEP_CHECKPOINTS_DIR"]
    dit, llm = AceStepHandler(), LLMHandler()
    print(dit.initialize_service(project_root=str(pathlib.Path(ck).parent),
                                 config_path=MODEL_REV["dit"], device="cuda"), flush=True)
    print(llm.initialize(checkpoint_dir=ck, lm_model_path=MODEL_REV["lm"], backend="vllm",
                         device="cuda"), flush=True)
    tmp = args.out / "_tmp"
    for s in todo:
        if dt.datetime.now(KST) >= stop:
            print("stop-at reached", flush=True)
            break
        t0 = time.time()
        params = GenerationParams(caption=s["caption"], lyrics=s["lyrics"],
                                  vocal_language=s["lang"], duration=s["duration"],
                                  seed=s["seed"], thinking=True, use_cot_caption=False,
                                  use_cot_language=False)
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[s["seed"]],
                               audio_format="flac")
        res = generate_music(dit, llm, params, cfg, save_dir=str(tmp))
        if not res.success or not res.audios:
            print(f"FAIL {s['id']}: {res.error}", flush=True)
            continue
        src = pathlib.Path(res.audios[0]["path"])
        master = args.out / "_master" / s["lang"] / f"{s['id']}.flac"
        master.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, master)
        mp3 = args.out / s["lang"] / f"{s['id']}.mp3"
        mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(master),
                        "-ar", "44100", "-ac", "2", "-b:a", "192k", str(mp3)], check=True)
        p = res.audios[0].get("params", {})
        rec = {**s, "file": f"{s['lang']}/{s['id']}.mp3", "master": f"_master/{s['lang']}/{s['id']}.flac",
               "model": MODEL_REV, "gen_s": round(time.time() - t0, 1),
               "lm_metas": {k: p.get(k) for k in ("bpm", "keyscale", "timesignature", "duration")
                            if isinstance(p, dict)},
               "created": dt.datetime.now(KST).isoformat(timespec="seconds")}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"ok {s['id']} {s['duration']:.0f}s in {rec['gen_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
