"""Shared helper for the Mandarin fake-voice families (S4, docs/training/09 §1).

Same contract as round 1's ``scripts/synth/synth_common.py`` (deterministic job list per
family seed, ``--shard i/n``, resumable file-locked appends), adapted to zh:

* texts   = ``_index/texts.csv`` (FLEURS cmn_hans_cn, zh TN applied; see zh_prep.py);
* prompts = ``_index/prompts.csv`` (strategy-v3 zh pool-A train_val REAL speakers,
  4 utterances each); ``prompt_speaker`` = the manifest ``speaker_ref_id``;
* CosyVoice needs the prompt transcript: ``_index/prompt_transcripts.csv`` from
  ``zh_qc.py transcribe`` (Whisper large-v3).

Three files per family, so synthesis and QC can run concurrently without rewriting
each other's rows:

``<family>/synth.csv``     one row per written wav (round 1's columns), appended by the synth
``<family>/qc.csv``        one row per QC'd wav (file, qc_cer, rms_dbfs, ...), appended by zh_qc
``<family>/metadata.csv``  = synth ⋈ qc, rewritten atomically by zh_qc after every pass;
                           round 1's columns + qc_cer, kept, drop_reason. The ingester
                           (processing/extend.py ``_synth_families``) keeps ``kept == True``.
Only stdlib + numpy + soundfile here, so every family venv can import it.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORPUS_ROOT = Path(os.environ.get("DACON_CORPUS_ROOT", "/data/project/private/dacon-corpus"))
OUT_ROOT = Path(os.environ.get("ZH_SYNTH_ROOT", CORPUS_ROOT / "interim" / "zh-synth"))  # override for dry runs
IDX = CORPUS_ROOT / "interim" / "zh-synth" / "_index"  # shared by smoke and production roots

META_COLUMNS = [
    "file", "family", "model", "model_revision", "licence", "text",
    "text_source_utts", "prompt_speaker", "prompt_files", "seed",
    "duration_s", "sample_rate",
]
QC_COLUMNS = ["file", "qc_cer", "asr_text", "rms_dbfs", "clip_frac", "silence_frac", "kept", "drop_reason"]
FINAL_COLUMNS = META_COLUMNS + ["qc_cer", "kept", "drop_reason"]

MIN_DURATION_S = 3.0


@dataclass
class Prompt:
    prompt_id: str
    speaker: str        # manifest speaker_ref_id, e.g. aishell1_S0002
    path: str           # corpus-relative
    duration_s: float
    text: str = ""      # Whisper transcript (CosyVoice only)

    @property
    def abs_path(self) -> Path:
        return CORPUS_ROOT / self.path


@dataclass
class Job:
    idx: int
    name: str
    text: str
    text_ids: list[str]
    prompt: Prompt
    seed: int


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_prompts(with_text: bool = False) -> list[Prompt]:
    ps = [Prompt(r["prompt_id"], r["speaker_ref_id"], r["path"], float(r["duration_s"]))
          for r in read_csv(IDX / "prompts.csv")]
    if with_text:
        tr = {r["prompt_id"]: r["text"] for r in read_csv(IDX / "prompt_transcripts.csv")}
        for p in ps:
            p.text = tr.get(p.prompt_id, "")
        ps = [p for p in ps if len(p.text) >= 4]
    return ps


def _family_seed(family: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{family}:{seed}".encode()).digest()[:8], "little")


def make_jobs(family: str, n_files: int, seed: int, prompts: list[Prompt],
              max_han: int = 60) -> list[Job]:
    """Deterministic: texts of 1 sentence (>= 15 hanzi, 55 %) or 2 sentences (<= max_han
    hanzi together); prompt speakers round-robin over a seed-shuffled speaker list, one of
    the speaker's prompt utterances drawn per job."""
    rng = random.Random(_family_seed(family, seed))
    texts = [(r["text_id"], r["text"], int(r["n_han"])) for r in read_csv(IDX / "texts.csv")]
    by_spk: dict[str, list[Prompt]] = {}
    for p in prompts:
        by_spk.setdefault(p.speaker, []).append(p)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    jobs: list[Job] = []
    while len(jobs) < n_files:
        a = rng.choice(texts)
        if a[2] >= 15 and rng.random() < 0.55:
            t, ids = a[1], [a[0]]
        else:
            b = rng.choice(texts)
            if b[0] == a[0] or a[2] + b[2] > max_han:
                if a[2] < 15:
                    continue
                t, ids = a[1], [a[0]]
            else:
                t, ids = a[1] + b[1], [a[0], b[0]]
        i = len(jobs)
        spk = speakers[i % len(speakers)]
        jobs.append(Job(i, f"{family}_{i:06d}", t, ids, rng.choice(by_spk[spk]), seed * 1_000_003 + i))
    return jobs


def shard_jobs(jobs: list[Job], shard: str) -> list[Job]:
    i, n = (int(x) for x in shard.split("/"))
    assert 0 <= i < n
    return [j for j in jobs if j.idx % n == i]


def append_row(path: Path, columns: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            w = csv.DictWriter(fh, fieldnames=columns)
            if fh.tell() == 0:
                w.writeheader()
            w.writerow(row)
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


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
    def __init__(self, family: str, model: str, model_revision: str, licence: str,
                 out_root: Path = OUT_ROOT):
        self.family, self.model, self.model_revision, self.licence = family, model, model_revision, licence
        self.dir = out_root / family
        self.dir.mkdir(parents=True, exist_ok=True)
        self.synth = self.dir / "synth.csv"
        prev = read_csv(self.synth)
        self.done = {r["file"] for r in prev}
        self.seconds_prev = sum(float(r["duration_s"]) for r in prev)
        self.n_written = self.n_dropped = 0
        self.seconds_written = 0.0
        self.t0 = time.time()

    def is_done(self, job: Job) -> bool:
        return f"{job.name}.wav" in self.done

    def write(self, job: Job, audio: np.ndarray, sr: int) -> bool:
        import soundfile as sf
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.mean(axis=0)
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
            "text_source_utts": "|".join(job.text_ids), "prompt_speaker": job.prompt.speaker,
            "prompt_files": job.prompt.path, "seed": job.seed,
            "duration_s": f"{dur:.3f}", "sample_rate": sr})
        self.done.add(fname)
        self.n_written += 1
        self.seconds_written += dur
        return True

    def progress(self, every: int = 25) -> None:
        if self.n_written and self.n_written % every == 0:
            el = time.time() - self.t0
            print(f"[{self.family}] written={self.n_written} dropped={self.n_dropped} "
                  f"audio={self.seconds_written/3600:.2f}h wall={el/60:.1f}min "
                  f"rtf={el/max(self.seconds_written,1e-6):.3f}", flush=True)

    def summary(self) -> str:
        return (f"[{self.family}] DONE written={self.n_written} dropped={self.n_dropped} "
                f"audio={self.seconds_written/3600:.2f}h wall={(time.time()-self.t0)/60:.1f}min")


def standard_argparser(family: str, default_files: int, default_seed: int):
    import argparse
    p = argparse.ArgumentParser(description=f"synthesise the zh {family} family")
    p.add_argument("--n-files", type=int, default=default_files, help="size of the job list (all shards)")
    p.add_argument("--seed", type=int, default=default_seed)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=0, help="stop after this many new files (smoke)")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop once the family's synth.csv (all runs and shards) holds this much audio")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return p


def should_stop(args, w: FamilyWriter) -> bool:
    """--limit counts this process's new files; --max-hours counts the whole family's synth.csv
    (all shards), re-read every 10 files so concurrent shards stop together."""
    if args.limit and w.n_written >= args.limit:
        return True
    if not args.max_hours:
        return False
    if w.n_written % 10 == 0 and w.n_written != getattr(w, "_checked_at", -1):
        w._checked_at = w.n_written
        w._family_s = sum(float(r["duration_s"]) for r in read_csv(w.synth))
    return getattr(w, "_family_s", w.seconds_prev) / 3600 >= args.max_hours


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
