"""Shared helper for the Korean fake-voice synthesis scripts (track K, plan 07 §1 D-a).

Every family script does the same four things through this module:

1. index Zeroth (OpenSLR-40) once: utterance id, speaker dir, text, duration
   (cached at ``<out_root>/_index/zeroth_index.csv``);
2. draw a deterministic job list from a per-family seed: texts made of 1-2
   Zeroth sentences (4-15 s worth of source speech), and for cloning / VC
   families a prompt speaker + prompt utterance whose text differs from the
   synthesised text;
3. write mono PCM16 WAVs at the generator's native rate, dropping anything
   under 3.0 s or silent;
4. append one row per file to ``<family>/metadata.csv`` (header written once),
   so a job that is re-run or sharded resumes without re-synthesising.

Run each family as ``python scripts/synth/synth_<family>.py --shard i/n``.
Shards partition the job list by index modulo n; the metadata.csv rows of all
shards go to one file (append with a file lock).
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CORPUS_ROOT = Path(os.environ.get("DACON_CORPUS_ROOT", "/data/project/private/dacon-corpus"))
ZEROTH_ROOT = CORPUS_ROOT / "interim" / "zeroth-korean" / "openslr-40"
OUT_ROOT = CORPUS_ROOT / "interim" / "ko-synth"
INDEX_CSV = OUT_ROOT / "_index" / "zeroth_index.csv"

META_COLUMNS = [
    "file", "family", "model", "model_revision", "licence", "text",
    "text_source_utts", "prompt_speaker", "prompt_files", "seed",
    "duration_s", "sample_rate",
]

MIN_DURATION_S = 3.0


@dataclass
class Utt:
    utt_id: str            # e.g. 104_003_0019
    speaker: str           # "zeroth-korean/test_data_01/003/104"
    rel_path: str          # corpus-relative flac path, "zeroth-korean/openslr-40/test_data_01/003/104/104_003_0019.flac"
    text: str
    duration_s: float

    @property
    def abs_path(self) -> Path:
        return CORPUS_ROOT / "interim" / self.rel_path


@dataclass
class Job:
    idx: int
    name: str                          # output stem, e.g. mms_000123
    text: str
    text_utts: list[str]
    prompt_speaker: str = ""           # Zeroth speaker id or "<family>/<builtin>"
    prompt_utts: list[Utt] = field(default_factory=list)
    seed: int = 0

    @property
    def prompt_files(self) -> str:
        return "|".join(u.rel_path for u in self.prompt_utts)


# --------------------------------------------------------------------------- index

def build_index() -> list[Utt]:
    """Read every *.trans.txt under Zeroth and the flac headers; cache as CSV."""
    if INDEX_CSV.exists():
        with INDEX_CSV.open(newline="", encoding="utf-8") as fh:
            return [Utt(r["utt_id"], r["speaker"], r["rel_path"], r["text"], float(r["duration_s"]))
                    for r in csv.DictReader(fh)]
    import soundfile as sf
    utts: list[Utt] = []
    for trans in sorted(ZEROTH_ROOT.glob("*/*/*/*.trans.txt")):
        spk_dir = trans.parent
        rel_dir = spk_dir.relative_to(ZEROTH_ROOT)          # test_data_01/003/104
        speaker = "zeroth-korean/" + rel_dir.as_posix()
        for line in trans.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            flac = spk_dir / f"{utt_id}.flac"
            if not flac.exists():
                continue
            info = sf.info(str(flac))
            rel = "zeroth-korean/openslr-40/" + (rel_dir / flac.name).as_posix()
            utts.append(Utt(utt_id, speaker, rel, text.strip(), info.frames / info.samplerate))
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["utt_id", "speaker", "rel_path", "text", "duration_s"])
        for u in utts:
            w.writerow([u.utt_id, u.speaker, u.rel_path, u.text, f"{u.duration_s:.3f}"])
    os.replace(tmp, INDEX_CSV)
    return utts


def by_speaker(utts: list[Utt]) -> dict[str, list[Utt]]:
    d: dict[str, list[Utt]] = {}
    for u in utts:
        d.setdefault(u.speaker, []).append(u)
    return d


# --------------------------------------------------------------------------- job drawing

def _family_seed(family: str, seed: int) -> int:
    h = hashlib.sha256(f"{family}:{seed}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def draw_texts(rng: random.Random, utts: list[Utt], n: int,
               min_src_s: float = 4.0, max_src_s: float = 15.0) -> list[tuple[str, list[str]]]:
    """n texts, each 1-2 Zeroth sentences whose *source* duration sums to 4-15 s.

    Sentences are drawn uniformly over the whole corpus (with replacement across
    texts, never within a text), so the same rng state always yields the same
    list.
    """
    out = []
    pool = [u for u in utts if u.duration_s <= max_src_s]
    while len(out) < n:
        a = rng.choice(pool)
        if a.duration_s >= min_src_s and rng.random() < 0.55:
            out.append((a.text, [a.utt_id]))
            continue
        b = rng.choice(pool)
        if b.utt_id == a.utt_id or a.duration_s + b.duration_s > max_src_s:
            if a.duration_s >= min_src_s:
                out.append((a.text, [a.utt_id]))
            continue
        out.append((a.text + " " + b.text, [a.utt_id, b.utt_id]))
    return out


def make_jobs(family: str, n_files: int, seed: int, utts: list[Utt], *,
              cloning: bool, prompt_min_s: float = 3.0, prompt_max_s: float = 12.0,
              n_prompt_files: int = 1, builtin_speakers: list[str] | None = None) -> list[Job]:
    """Deterministic job list for a family.

    cloning=True: every job gets a Zeroth prompt speaker (round-robin over the
    speakers, shuffled by the seed) and n_prompt_files prompt utterances from
    that speaker; the text is drawn from other utterances (never the prompt's).
    cloning=False: prompt_speaker cycles through builtin_speakers (or is empty).
    """
    rng = random.Random(_family_seed(family, seed))
    texts = draw_texts(rng, utts, n_files)
    spk = by_speaker(utts)
    speakers = sorted(spk)
    rng.shuffle(speakers)
    jobs: list[Job] = []
    for i, (text, tutts) in enumerate(texts):
        job = Job(i, f"{family}_{i:06d}", text, tutts, seed=seed * 1_000_003 + i)
        if cloning:
            s = speakers[i % len(speakers)]
            cands = [u for u in spk[s] if prompt_min_s <= u.duration_s <= prompt_max_s
                     and u.utt_id not in tutts]
            if not cands:
                cands = [u for u in spk[s] if u.utt_id not in tutts]
            job.prompt_speaker = s
            job.prompt_utts = rng.sample(cands, min(n_prompt_files, len(cands)))
        elif builtin_speakers:
            job.prompt_speaker = f"{family}/{builtin_speakers[i % len(builtin_speakers)]}"
        jobs.append(job)
    return jobs


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    assert 0 <= i < n
    return i, n


def shard_jobs(jobs: list[Job], shard: str) -> list[Job]:
    i, n = parse_shard(shard)
    return [j for j in jobs if j.idx % n == i]


# --------------------------------------------------------------------------- output

class FamilyWriter:
    """Writes WAVs + metadata rows for one family; resumable and shard-safe."""

    def __init__(self, family: str, model: str, model_revision: str, licence: str,
                 out_root: Path = OUT_ROOT):
        self.family, self.model, self.model_revision, self.licence = family, model, model_revision, licence
        self.dir = out_root / family
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = self.dir / "metadata.csv"
        self.done = self._read_done()
        self.n_written = 0
        self.n_dropped = 0
        self.seconds_written = 0.0
        self.t0 = time.time()

    def _read_done(self) -> set[str]:
        if not self.meta.exists():
            return set()
        with self.meta.open(newline="", encoding="utf-8") as fh:
            return {r["file"] for r in csv.DictReader(fh)}

    def is_done(self, job: Job) -> bool:
        return f"{job.name}.wav" in self.done

    def write(self, job: Job, audio: np.ndarray, sr: int) -> bool:
        """audio: float32 mono in [-1, 1]. Returns False if the file was dropped."""
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
        row = {
            "file": fname, "family": self.family, "model": self.model,
            "model_revision": self.model_revision, "licence": self.licence, "text": job.text,
            "text_source_utts": "|".join(job.text_utts), "prompt_speaker": job.prompt_speaker,
            "prompt_files": job.prompt_files, "seed": job.seed,
            "duration_s": f"{dur:.3f}", "sample_rate": sr,
        }
        self._append(row)
        self.done.add(fname)
        self.n_written += 1
        self.seconds_written += dur
        return True

    def _append(self, row: dict) -> None:
        with self.meta.open("a", newline="", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0, os.SEEK_END)
                w = csv.DictWriter(fh, fieldnames=META_COLUMNS)
                if fh.tell() == 0:
                    w.writeheader()
                w.writerow(row)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def progress(self, every: int = 25) -> None:
        if self.n_written and self.n_written % every == 0:
            el = time.time() - self.t0
            print(f"[{self.family}] written={self.n_written} dropped={self.n_dropped} "
                  f"audio={self.seconds_written/3600:.2f}h wall={el/60:.1f}min "
                  f"rtf={el/max(self.seconds_written,1e-6):.3f}", flush=True)

    def summary(self) -> str:
        el = time.time() - self.t0
        return (f"[{self.family}] DONE written={self.n_written} dropped={self.n_dropped} "
                f"audio={self.seconds_written/3600:.2f}h wall={el/60:.1f}min")


def trim_silence(audio: np.ndarray, sr: int, thresh: float = 1e-3, pad_s: float = 0.15) -> np.ndarray:
    """Cut leading/trailing near-silence (frame RMS < thresh), keeping pad_s on each side."""
    if len(audio) == 0:
        return audio
    hop = max(1, sr // 100)
    n = len(audio) // hop
    if n == 0:
        return audio
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = np.nonzero(rms > thresh)[0]
    if len(loud) == 0:
        return audio[:0]
    pad = int(pad_s * sr)
    start = max(0, loud[0] * hop - pad)
    end = min(len(audio), (loud[-1] + 1) * hop + pad)
    return audio[start:end]


def load_prompt(utt: Utt, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(utt.abs_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr and sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return audio, sr


def concat_prompts(utts: list[Utt], target_sr: int, max_s: float = 15.0) -> np.ndarray:
    parts, total = [], 0.0
    for u in utts:
        a, _ = load_prompt(u, target_sr)
        a = trim_silence(a, target_sr)
        if total + len(a) / target_sr > max_s and parts:
            break
        parts.append(a)
        total += len(a) / target_sr
    return np.concatenate(parts) if parts else np.zeros(target_sr, np.float32)


def standard_argparser(family: str, default_files: int, default_seed: int):
    import argparse
    p = argparse.ArgumentParser(description=f"synthesise the {family} family")
    p.add_argument("--n-files", type=int, default=default_files, help="size of the job list (all shards)")
    p.add_argument("--seed", type=int, default=default_seed)
    p.add_argument("--shard", default="0/1", help="i/n: this process handles jobs with idx %% n == i")
    p.add_argument("--limit", type=int, default=0, help="stop after this many new files (smoke tests)")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return p


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
