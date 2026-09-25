"""Shared helper for the round-2 Korean fake-voice families (S2, docs/training/09 §2-§3).

Same contract as round 1's ``scripts/synth/synth_common.py`` (deterministic job list per
family seed, ``--shard i/n``, resumable file-locked appends), with round-2 sources:

* utterance pool (prompts AND texts), read fresh at every start so later jobs see more Emilia:
    - ``emilia``  interim/emilia-ko/metadata.csv, kept rows of part emilia|yodas (S3 output)
    - ``fleurs``  interim/emilia-ko/metadata.csv, part fleurs (read speech, CC-BY-4.0)
    - ``zeroth``  round 1's zeroth_index.csv, ONLY speakers whose strategy-v3 rows are slice
                  train_val (allow-list cached in _index/zeroth_train_val_speakers.txt)
* prompts: >= 70 % Emilia speakers once Emilia is indexed, the rest Zeroth; before that,
  FLEURS + Zeroth. Speakers are taken round-robin (seed-shuffled) so a family spans many.
* texts: dataset transcripts only (Hangul + basic punctuation; digits/Latin dropped because
  Whisper writes them as digits and the CER would measure the normaliser, not the TTS),
  1-2 utterances, 3-15 s of source speech, never the prompt utterance.

Files per family (synthesis and QC run concurrently without rewriting each other's rows):
``<family>/synth.csv``     one row per written wav (round 1's columns), appended by the synth
``<family>/qc.csv``        one row per QC'd wav, appended by ko_qc.py
``<family>/metadata.csv``  = synth ⋈ qc, rewritten atomically by ko_qc.py; round 1's columns +
                           qc_cer, kept, drop_reason (the ingester keeps kept == True).
Only stdlib + numpy + soundfile here, so every family venv can import it.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CORPUS_ROOT = Path(os.environ.get("DACON_CORPUS_ROOT", "/data/project/private/dacon-corpus"))
INTERIM = CORPUS_ROOT / "interim"
OUT_ROOT = Path(os.environ.get("KO_SYNTH2_ROOT", INTERIM / "ko-synth2"))
IDX = INTERIM / "ko-synth2" / "_index"
EMILIA_META = INTERIM / "emilia-ko" / "metadata.csv"
ZEROTH_INDEX = INTERIM / "ko-synth" / "_index" / "zeroth_index.csv"
ZEROTH_ALLOW = IDX / "zeroth_train_val_speakers.txt"
FOLDS = CORPUS_ROOT / "manifests" / "strategy-v3" / "folds.parquet"

META_COLUMNS = [
    "file", "family", "model", "model_revision", "licence", "text",
    "text_source_utts", "prompt_speaker", "prompt_files", "seed",
    "duration_s", "sample_rate",
]
QC_COLUMNS = ["file", "qc_cer", "asr_text", "duration_s", "rms_dbfs", "clip_frac", "silence_frac",
              "kept", "drop_reason"]
FINAL_COLUMNS = META_COLUMNS + ["qc_cer", "kept", "drop_reason"]
MIN_DURATION_S = 3.0
TEXT_OK = re.compile(r"^[가-힣\s.,?!~'\"…·-]+$")


@dataclass
class Utt:
    utt_id: str
    speaker: str       # "emilia-ko/<id>" | "fleurs-ko/<split>-<gender>" | "zeroth-korean/<split>/<ch>/<spk>"
    rel_path: str      # relative to interim/
    text: str
    duration_s: float
    source: str        # emilia | yodas | fleurs | zeroth

    @property
    def abs_path(self) -> Path:
        return INTERIM / self.rel_path


@dataclass
class Job:
    idx: int
    name: str
    text: str
    text_utts: list[str]
    prompt_speaker: str = ""
    prompt_utts: list[Utt] = field(default_factory=list)
    src_utts: list[Utt] = field(default_factory=list)   # VC families: the source utterance(s)
    seed: int = 0

    @property
    def prompt_files(self) -> str:
        return "|".join(u.rel_path for u in self.prompt_utts)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_row(path: Path, columns: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            if fh.tell() == 0:
                w.writeheader()
            w.writerow(row)
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- pool
def zeroth_allow() -> set[str]:
    if not ZEROTH_ALLOW.exists():
        raise SystemExit(f"{ZEROTH_ALLOW} missing: run `ko_common.py allowlist` with a parquet-capable python")
    return {s.strip() for s in ZEROTH_ALLOW.read_text().splitlines() if s.strip()}


def load_pool() -> list[Utt]:
    utts: list[Utt] = []
    allow = zeroth_allow()
    for r in read_csv(ZEROTH_INDEX):
        if r["speaker"] in allow:
            utts.append(Utt(r["utt_id"], r["speaker"], r["rel_path"], r["text"].strip(),
                            float(r["duration_s"]), "zeroth"))
    for r in read_csv(EMILIA_META):
        if str(r.get("kept", "")).lower() not in ("true", "1"):
            continue
        utts.append(Utt(Path(r["file"]).stem, r["speaker_ref_id"], "emilia-ko/" + r["file"],
                        r["text"].strip(), float(r["duration_s"]), r["part"]))
    return utts


def text_ok(t: str) -> bool:
    return bool(TEXT_OK.match(t)) and len(re.sub(r"\s", "", t)) >= 6


def by_speaker(utts: list[Utt]) -> dict[str, list[Utt]]:
    d: dict[str, list[Utt]] = {}
    for u in utts:
        d.setdefault(u.speaker, []).append(u)
    return d


def _family_seed(family: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{family}:{seed}".encode()).digest()[:8], "little")


def _pick_text(rng: random.Random, pool: list[Utt], min_s: float, max_s: float,
               avoid: set[str]) -> tuple[str, list[str]]:
    while True:
        a = rng.choice(pool)
        if a.utt_id in avoid:
            continue
        if a.duration_s >= min_s and (a.duration_s > max_s * 0.6 or rng.random() < 0.55):
            return a.text, [a.utt_id]
        b = rng.choice(pool)
        if b.utt_id in avoid or b.utt_id == a.utt_id or a.duration_s + b.duration_s > max_s:
            if a.duration_s >= min_s:
                return a.text, [a.utt_id]
            continue
        return a.text + " " + b.text, [a.utt_id, b.utt_id]


def make_jobs(family: str, n_files: int, seed: int, utts: list[Utt], *,
              prompt_min_s: float = 3.0, prompt_max_s: float = 10.0, n_prompt_files: int = 1,
              emilia_frac: float = 0.75, text_min_s: float = 3.0, text_max_s: float = 13.0,
              vc: bool = False) -> list[Job]:
    """Deterministic job list. Prompt speakers: emilia_frac of jobs from Emilia speakers
    (if any are indexed), the rest Zeroth (FLEURS stands in for Emilia before it exists).
    vc=True: the 'text' is one source utterance of ANOTHER speaker (src_utts), 3-15 s."""
    rng = random.Random(_family_seed(family, seed))
    spk = by_speaker(utts)

    def ok_prompt(u: Utt) -> bool:
        return prompt_min_s <= u.duration_s <= prompt_max_s

    def speakers_of(sources: set[str]) -> list[str]:
        s = sorted(k for k, v in spk.items() if v[0].source in sources and any(ok_prompt(u) for u in v))
        rng.shuffle(s)
        return s

    em = speakers_of({"emilia", "yodas"})
    fl = speakers_of({"fleurs"})
    ze = speakers_of({"zeroth"})
    main = em if em else fl
    text_pool = [u for u in utts if text_ok(u.text) and u.duration_s <= 15.0]
    if em:  # conversational texts preferred once Emilia exists: 70 % Emilia, rest FLEURS/Zeroth
        em_texts = [u for u in text_pool if u.source in ("emilia", "yodas")]
        other_texts = [u for u in text_pool if u.source not in ("emilia", "yodas")]
    else:
        em_texts, other_texts = [], text_pool
    jobs: list[Job] = []
    ci = {"m": 0, "z": 0}
    for i in range(n_files):
        use_main = main and (not ze or rng.random() < emilia_frac)
        key = "m" if use_main else "z"
        lst = main if use_main else ze
        s = lst[ci[key] % len(lst)]
        ci[key] += 1
        cands = [u for u in spk[s] if ok_prompt(u)]
        prompts = rng.sample(cands, min(n_prompt_files, len(cands)))
        avoid = {u.utt_id for u in spk[s]} if vc else {u.utt_id for u in prompts}
        tp = em_texts if (em_texts and rng.random() < 0.7) else other_texts
        job = Job(i, f"{family}_{i:06d}", "", [], prompt_speaker=s, prompt_utts=prompts,
                  seed=seed * 1_000_003 + i)
        if vc:
            src_pool = [u for u in tp if text_min_s <= u.duration_s <= 15.0]
            while True:
                src = rng.choice(src_pool)
                if src.speaker != s:
                    break
            job.src_utts = [src]
            job.text, job.text_utts = src.text, [src.utt_id]
        else:
            job.text, job.text_utts = _pick_text(rng, tp, text_min_s, text_max_s, avoid)
        jobs.append(job)
    return jobs


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    assert 0 <= i < n
    return i, n


def shard_jobs(jobs: list[Job], shard: str) -> list[Job]:
    i, n = parse_shard(shard)
    return [j for j in jobs if j.idx % n == i]


def in_shard(fname: str, shard: str) -> bool:
    i, n = parse_shard(shard)
    m = re.search(r"_(\d{6})\.wav$", fname)
    return bool(m) and int(m.group(1)) % n == i


# --------------------------------------------------------------------------- output
def trim_silence(audio: np.ndarray, sr: int, thresh: float = 1e-3, pad_s: float = 0.15) -> np.ndarray:
    if len(audio) == 0:
        return audio
    hop = max(1, sr // 100)
    n = len(audio) // hop
    if n == 0:
        return audio
    rms = np.sqrt((audio[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    loud = np.nonzero(rms > thresh)[0]
    if len(loud) == 0:
        return audio[:0]
    pad = int(pad_s * sr)
    return audio[max(0, loud[0] * hop - pad): min(len(audio), (loud[-1] + 1) * hop + pad)]


class FamilyWriter:
    """WAVs + synth.csv rows for one family; resumable and shard-safe."""

    def __init__(self, family: str, model: str, model_revision: str, licence: str,
                 out_root: Path = OUT_ROOT):
        self.family, self.model, self.model_revision, self.licence = family, model, model_revision, licence
        self.dir = out_root / family
        self.dir.mkdir(parents=True, exist_ok=True)
        self.synth = self.dir / "synth.csv"
        prior = read_csv(self.synth)
        self.done = {r["file"] for r in prior}
        self.prior_seconds = sum(float(r["duration_s"]) for r in prior)  # whole family, at start
        self.n_written = self.n_dropped = 0
        self.seconds_written = 0.0
        self.t0 = time.time()

    def is_done(self, job: Job) -> bool:
        return f"{job.name}.wav" in self.done

    def write(self, job: Job, audio: np.ndarray, sr: int) -> bool:
        import soundfile as sf
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio[:, 0]
        audio = trim_silence(audio, sr)
        dur = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if dur < MIN_DURATION_S or rms < 1e-3 or not np.isfinite(audio).all():
            self.n_dropped += 1
            return False
        peak = float(np.abs(audio).max())
        if peak > 0.99:
            audio = audio / peak * 0.99
        fname = f"{job.name}.wav"
        tmp = self.dir / (fname + ".tmp")
        sf.write(str(tmp), audio, sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, self.dir / fname)
        append_row(self.synth, META_COLUMNS, {
            "file": fname, "family": self.family, "model": self.model,
            "model_revision": self.model_revision, "licence": self.licence, "text": job.text,
            "text_source_utts": "|".join(job.text_utts), "prompt_speaker": job.prompt_speaker,
            "prompt_files": job.prompt_files, "seed": job.seed,
            "duration_s": f"{dur:.3f}", "sample_rate": sr})
        self.done.add(fname)
        self.n_written += 1
        self.seconds_written += dur
        return True

    def progress(self, every: int = 20) -> None:
        if self.n_written and self.n_written % every == 0:
            el = time.time() - self.t0
            log(f"[{self.family}] written={self.n_written} dropped={self.n_dropped} "
                f"audio={self.seconds_written/3600:.3f}h wall={el/60:.1f}min "
                f"x_rt={self.seconds_written/max(el,1e-6):.2f}")

    def summary(self) -> str:
        el = time.time() - self.t0
        return (f"[{self.family}] DONE written={self.n_written} dropped={self.n_dropped} "
                f"audio={self.seconds_written/3600:.3f}h wall={el/60:.1f}min "
                f"x_rt={self.seconds_written/max(el,1e-6):.2f}")


def should_stop(args, w: FamilyWriter) -> bool:
    if args.limit and w.n_written >= args.limit:
        return True
    if args.max_hours and w.seconds_written / 3600 >= args.max_hours:
        return True
    if getattr(args, "max_family_hours", 0):
        if w.n_written and w.n_written % 20 == 0 and getattr(w, "_refreshed_at", -1) != w.n_written:
            # other shards write to the same synth.csv: re-read the family total
            w._refreshed_at = w.n_written
            w.prior_seconds = sum(float(r["duration_s"]) for r in read_csv(w.synth)) - w.seconds_written
        if (w.prior_seconds + w.seconds_written) / 3600 >= args.max_family_hours:
            return True
    return bool(args.deadline and time.time() >= args.deadline)


def load_audio(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if target_sr and sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        a = resample_poly(a, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return a, sr


def standard_argparser(family: str, default_files: int, default_seed: int):
    p = argparse.ArgumentParser(description=f"synthesise the {family} family")
    p.add_argument("--n-files", type=int, default=default_files, help="size of the job list (all shards)")
    p.add_argument("--seed", type=int, default=default_seed)
    p.add_argument("--shard", default="0/1", help="i/n: jobs with idx %% n == i")
    p.add_argument("--limit", type=int, default=0, help="stop after this many new files (smoke)")
    p.add_argument("--max-hours", type=float, default=0.0, help="stop after this many written hours (this process)")
    p.add_argument("--max-family-hours", type=float, default=0.0,
                   help="stop when the family's synth.csv (at start) + this process's hours reach this")
    p.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; stop starting new files after")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return p


def pool_summary(utts: list[Utt]) -> str:
    from collections import Counter
    c = Counter(u.source for u in utts)
    s = Counter(u.source for u in {u.speaker: u for u in utts}.values())
    return f"pool utts={dict(c)} speakers={dict(s)}"


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["allowlist"]:
        import pandas as pd
        d = pd.read_parquet(FOLDS, columns=["slice", "speaker_ref_id"])
        z = d[d.speaker_ref_id.astype(str).str.startswith("zeroth-korean/")]
        bad = set(z[z.slice != "train_val"].speaker_ref_id)
        ok = sorted(set(z[z.slice == "train_val"].speaker_ref_id) - bad)
        IDX.mkdir(parents=True, exist_ok=True)
        ZEROTH_ALLOW.write_text("\n".join(ok) + "\n")
        print(f"{len(ok)} train_val Zeroth speakers ({len(bad)} excluded as probe/other)")
    elif sys.argv[1:] == ["pool"]:
        u = load_pool()
        print(pool_summary(u))
        j = make_jobs("x", 200, 1, u)
        print(len({x.prompt_speaker for x in j}), "prompt speakers in 200 jobs;", j[0])
