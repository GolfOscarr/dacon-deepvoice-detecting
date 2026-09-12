"""Enumerate, shard, probe, resume, consolidate.

The M tier is ~1.7 M files and `ffprobe` is ~20 ms of process spawn each, so
this module exists to make that 16 minutes instead of 8 hours and to make an
interrupted pass cost only what it did not finish.

Three things are load-bearing and all three are lessons from
`scripts/fetch_to_s3.py`, where they were learned by watching a transfer run:

* **A part's `DONE` marker is written after the part, never before.** Its
  absence means the write was interrupted and the part may be short.
* **A failure is a row.** `R2`: nothing here drops a file. A file that cannot be
  probed gets a row with `probe_ok = False` and the error, because a file that
  vanishes from a census makes the census wrong in the direction that hides
  problems.
* **Threads, not processes.** `subprocess.run` releases the GIL, so a thread
  pool gets the same parallelism with none of the pickling.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from eda.config import EdaConfig, SourceSpec
from eda.extract import run_metadata, run_signal
from eda.ids import file_id_for
from eda.planes import load_planes
from eda.sample import SAMPLE_FILE, load_sample, sampled_rows
# Importing for the side effect of registration. Critical: without these the
# registry is empty and `probe_source` writes a table of ids and nothing else --
# which would look like a successful run.
from eda.extract import identity as _identity        # noqa: F401
from eda.extract import metadata as _metadata        # noqa: F401
from eda.extract import level as _level              # noqa: F401
from eda.extract import spectral as _spectral        # noqa: F401
from eda.extract import timing as _timing            # noqa: F401

__all__ = ["BlockedSource", "FIXED_COLUMNS", "NotProbed", "SIGNAL_TABLE",
           "SignalResult", "consolidate", "enumerate_source", "load_files",
           "load_signal", "probe_source", "shards", "signal_partition"]


class NotProbed(RuntimeError):
    """A partition with no completed parts, because nothing wrote any yet.

    Its own type for the same reason `BlockedSource` has one: the CLI must tell
    an expected state during a staged fetch from a real failure, and a shared
    `RuntimeError` makes that a string comparison.
    """


class BlockedSource(RuntimeError):
    """A source deliberately marked not-runnable (`SourceSpec.blocked`).

    Its own type because the CLI must tell three states apart: **blocked** (a
    decision), **absent** (not fetched yet -- the normal state during a staged
    fetch), and **empty** (a misconfiguration). The first two are reported and
    skipped; the third is fatal, because a source that matches nothing is
    exactly the silent failure `enumerate_source` exists to refuse.
    """

#: Written for every row before any extractor runs. `file_id` is assigned here,
#: in the M tier, and never recomputed: the S tier joins on it, and a sampled
#: row that cannot be joined back to its metadata is a row that cannot enter the
#: manifest (docs/EDA/00 section 1).
FIXED_COLUMNS = ("file_id", "path", "source_name", "row_kind", "pool", "cell")


def enumerate_source(cfg: EdaConfig, source: SourceSpec) -> list[Path]:
    """Every audio file under the source root, sorted, exclusions applied.

    Sorted because the shard boundaries have to be the same on a re-run for the
    `DONE` markers to mean anything.
    """
    if source.blocked:
        raise BlockedSource(f"source {source.name!r} is blocked: {source.blocked}")
    root = cfg.source_root(source)
    if not root.exists():
        raise FileNotFoundError(
            f"source {source.name!r}: {root} does not exist. Nothing in S3 is "
            f"extracted -- `scripts/fetch_from_s3.py --extract` is the path")
    suffixes = {s.lower() for s in source.suffixes}
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in suffixes:
            continue
        # `SourceSpec` owns the matching, so this and `gates._g_eda1` cannot
        # disagree about what an exclusion means.
        if source.excludes_path(p.relative_to(root).as_posix()):
            continue
        out.append(p)
    out.sort()
    # Critical: a registered source that yields nothing is a misconfiguration --
    # wrong root, wrong suffixes, or everything excluded -- and it is silent,
    # because the directory *exists*. MLAAD is the case that proved it: it is
    # stored in S3 with its tree intact rather than as an archive, so
    # `extract_archives` unpacked nothing, left an empty
    # `interim/mlaad/v9/`, and the corpus's best generator-diversity asset
    # enumerated to zero with no error anywhere.
    if not out:
        raise RuntimeError(
            f"source {source.name!r}: {root} exists but yielded no files matching "
            f"{list(source.suffixes)}"
            + (f" after excluding {list(source.exclude)}" if source.exclude else "")
            + f". If this source is stored unarchived, it needs `stage: raw` and a "
              f"root that includes `payload/`")
    return out


def shards(paths: Sequence[Path], size: int) -> Iterator[tuple[int, list[Path]]]:
    if size < 1:
        raise ValueError(f"shard_size must be >= 1, got {size}")
    for i in range(0, len(paths), size):
        yield i // size, list(paths[i:i + size])


def _row(cfg: EdaConfig, source: SourceSpec, path: Path) -> dict:
    base = cfg.source_root(source)
    rel = path.relative_to(base)
    fixed = {
        "file_id": file_id_for(source.name, rel),
        "path": path.relative_to(cfg.root if source.stage == "interim"
                                 else cfg.raw).as_posix(),
        "source_name": source.name,
        # Exactly one of pool/cell is non-null, per `SourceSpec.__post_init__`
        # and `validate_manifest`. Both columns are written either way so the
        # table's schema does not depend on which sources happen to be in it.
        "row_kind": source.row_kind,
        "pool": source.pool,
        "cell": source.cell,
    }
    names = cfg.probe.extractors or None
    return {**fixed, **run_metadata(path, cfg.probe, names)}


@dataclass
class ProbeResult:
    source: str
    files: int
    shards_run: int
    shards_skipped: int
    table: Path


def probe_source(cfg: EdaConfig, source: SourceSpec, *, dry_run: bool = False,
                 progress=None) -> ProbeResult:
    """M tier over one source. Resumable at part granularity."""
    paths = enumerate_source(cfg, source)
    parts_dir = cfg.out / source.partition / "parts" / source.name
    table = cfg.out / source.partition / "files.parquet"
    if dry_run:
        n = len(list(shards(paths, cfg.probe.shard_size)))
        return ProbeResult(source.name, len(paths), 0, n, table)

    parts_dir.mkdir(parents=True, exist_ok=True)
    run = skipped = 0
    with ThreadPoolExecutor(max_workers=cfg.probe.workers) as pool:
        for idx, batch in shards(paths, cfg.probe.shard_size):
            part = parts_dir / f"{idx:05d}.parquet"
            done = part.with_suffix(".DONE")
            if done.exists():
                skipped += 1
                continue
            rows = list(pool.map(lambda p: _row(cfg, source, p), batch))
            pd.DataFrame(rows).to_parquet(part, index=False)
            # Critical: after the part, never before. Its absence is what tells
            # `consolidate` the write was interrupted and the part may be short.
            done.write_text(json.dumps({"rows": len(rows), "shard": idx}),
                            encoding="utf-8")
            run += 1
            if progress is not None:
                progress(source.name, idx, len(rows))
    return ProbeResult(source.name, len(paths), run, skipped, table)


def consolidate(cfg: EdaConfig, partition: str) -> Path:
    """Every completed part for one partition into its `files.parquet`.

    ⚠️ A part without its `DONE` marker is **skipped and reported**, not merged.
    The marker is written last, so its absence means the part may be short --
    and a short part is a census that silently under-counts.
    """
    part_dir = cfg.out / partition
    parts_root = part_dir / "parts"
    frames, incomplete = [], []
    for source in sorted(cfg.sources_in(partition), key=lambda s: s.name):
        d = parts_root / source.name
        if not d.exists():
            continue
        # 🔴 A source blocked *after* it was probed still has its parts on disk,
        # and merging them would put a dropped source back into the census with
        # nothing saying so. `rirs-pointsource` is the case this was written
        # for: 843 rows, every one a byte-identical MUSAN copy, sitting under
        # `out/E/parts/` from the run that found the leak. Skipped and named --
        # not deleted, because the parts are the evidence for the decision.
        if source.blocked:
            continue
        for part in sorted(d.glob("*.parquet")):
            if not part.with_suffix(".DONE").exists():
                incomplete.append(part)
                continue
            frame = pd.read_parquet(part)
            # Critical: this is what makes FIXED_COLUMNS load-bearing rather than
            # decorative. A part written by an older revision, or by a probe run
            # with a narrowed `extractors` list, is missing columns that every
            # downstream join assumes -- and a missing `file_id` would surface as
            # a KeyError three modules away instead of here, naming the file.
            missing = [c for c in FIXED_COLUMNS if c not in frame.columns]
            if missing:
                raise RuntimeError(
                    f"{part} is missing {missing}. It was written by a different "
                    f"revision of `eda.driver`; delete the part and its DONE "
                    f"marker and re-run `probe`")
            frames.append(frame)
    if incomplete:
        names = ", ".join(str(p) for p in incomplete[:5])
        raise RuntimeError(
            f"{len(incomplete)} part(s) have no DONE marker and were written by an "
            f"interrupted run: {names}. Delete them and re-run `probe`; merging a "
            f"short part is a census that under-counts without saying so")
    if not frames:
        # Critical: distinguish "never probed" from "probed and produced
        # nothing". The first is the normal state of a partition whose sources
        # are still absent -- `cell8` until SONICS is fetched -- and the CLI
        # skips it. The second would mean parts exist but none completed, which
        # the DONE check above has already raised for.
        raise NotProbed(
            f"partition {partition}: no completed parts under {parts_root}")
    df = pd.concat(frames, ignore_index=True)
    dupes = df["file_id"].duplicated()
    if dupes.any():
        raise RuntimeError(
            f"partition {partition}: {int(dupes.sum())} duplicate file_id(s), e.g. "
            f"{df.loc[dupes, 'file_id'].iloc[0]!r}. Two sources claim the same "
            f"path, or a source root overlaps another")
    part_dir.mkdir(parents=True, exist_ok=True)
    out = part_dir / "files.parquet"
    df.to_parquet(out, index=False)
    return out


def load_files(cfg: EdaConfig, partitions: Sequence[str] | None = None) -> pd.DataFrame:
    """Every partition's `files.parquet`, concatenated. The input to `eda.analyze`.

    ⚠️ A declared partition with no table is **reported on the frame**, not
    skipped silently. An audit run over half the corpus produces real-looking
    numbers for the heads it can still score, and the reader has no way to tell.
    The names land in `df.attrs["missing_partitions"]` so the CLI can print them
    and `eda.gates` can tell an empty partition from an absent one.
    """
    # 🔴 A bare string is a Sequence[str] of its own characters, so
    # `load_files(cfg, "cell8")` asked for partitions c, e, l, l, 8 and raised
    # naming them. `"E"` happened to work, which is what made it survive: every
    # pool name is one character and only the first `cellN` partition exposed
    # it. Accept the singular spelling rather than punish it.
    if isinstance(partitions, str):
        partitions = [partitions]
    partitions = list(partitions or cfg.partitions())
    frames, missing = [], []
    for partition in partitions:
        path = cfg.out / partition / "files.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(partition)
    if not frames:
        raise RuntimeError(
            f"no files.parquet under {cfg.out} for partition(s) {partitions}. Run "
            f"`probe` then `consolidate` first")
    df = pd.concat(frames, ignore_index=True)
    df.attrs["missing_partitions"] = missing
    return df


#: The S tier's table, beside `files.parquet` and joined to it on `file_id`.
SIGNAL_TABLE = "signal.parquet"
#: Written for every signal row whatever happened, so a decode failure is a row
#: rather than a gap. `signal_ok` false with `signal_error` set is the shape.
SIGNAL_FIXED = ("file_id", "source_name", "partition", "signal_ok", "signal_error")


@dataclass
class SignalResult:
    partition: str
    files: int
    shards_run: int
    shards_skipped: int
    failures: int
    table: Path


def _signal_row(cfg: EdaConfig, row: pd.Series, partition: str) -> dict:
    """One file, both planes. A decode failure is a row, not an exception.

    ⚠️ `declared_sr` comes from the M tier's `orig_sr`, and `native_rate` will
    raise if the decoder disagrees. That is deliberate and it is the reason the
    two tiers join: a file whose census row describes a different file than the
    decoder opens must not quietly contribute statistics to either.
    """
    fixed = {"file_id": row["file_id"], "source_name": row["source_name"],
             "partition": partition}
    path = (cfg.root if row.get("stage", "interim") == "interim" else cfg.raw) / row["path"]
    declared = row.get("orig_sr")
    declared = None if declared is None or pd.isna(declared) else int(declared)
    try:
        planes = load_planes(path, declared_sr=declared, file_id=str(row["file_id"]))
    except Exception as exc:                         # noqa: BLE001 -- a row, not a raise
        return {**fixed, "signal_ok": False,
                "signal_error": f"{type(exc).__name__}: {exc}"}
    return {**fixed, "signal_ok": True, "signal_error": None,
            **run_signal(planes)}


def signal_partition(cfg: EdaConfig, partition: str, *, progress=None
                     ) -> SignalResult:
    """S tier over one partition's recorded draw. Resumable at part granularity.

    The draw must already exist -- `eda.sample.draw` writes it before anything
    decodes, and this reads it. A pass that drew its own sample would measure a
    different set of files every time a source was re-probed, and the numbers it
    published would not be reproducible from `sample.json` (docs/EDA/00 §3).
    """
    files = load_files(cfg, partition)
    sample = load_sample(cfg, partition)
    if sample is None:
        raise NotProbed(
            f"partition {partition}: no {SAMPLE_FILE} -- draw the sample first. "
            f"The draw is part of the corpus definition, not a step this pass "
            f"may improvise")
    rows = sampled_rows(files, sample)

    parts_dir = cfg.out / partition / "signal_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    table = cfg.out / partition / SIGNAL_TABLE
    run = skipped = failures = 0
    with ThreadPoolExecutor(max_workers=cfg.probe.workers) as pool:
        for idx, batch in shards(list(range(len(rows))), cfg.sample.shard_size):
            part = parts_dir / f"{idx:05d}.parquet"
            done = part.with_suffix(".DONE")
            if done.exists():
                # 🔴 A part is identified by its index, so its *boundaries* are
                # only meaningful under the shard size that wrote it. Change
                # `sample.shard_size` between runs and part 00000 covers rows
                # 0..N-1 of the old size while this loop believes it covers
                # 0..M-1 -- every row in between is measured twice and both
                # copies are merged. Measured: 8 files became 14 rows with 6
                # duplicate file_ids, silently. The marker records the size so
                # the mismatch is a refusal instead.
                marker = json.loads(done.read_text(encoding="utf-8"))
                wrote = marker.get("shard_size")
                if wrote != cfg.sample.shard_size:
                    raise RuntimeError(
                        f"partition {partition}: {part.name} was written with "
                        f"shard_size={wrote} and the config now says "
                        f"{cfg.sample.shard_size}. Part boundaries are not "
                        f"comparable across sizes and resuming would measure rows "
                        f"twice. Delete {parts_dir} and re-run, or put the size back")
                skipped += 1
                continue
            out = list(pool.map(
                lambda i: _signal_row(cfg, rows.iloc[i], partition), batch))
            frame = pd.DataFrame(out)
            frame.to_parquet(part, index=False)
            failures += int((~frame["signal_ok"]).sum())
            done.write_text(
                json.dumps({"rows": len(out), "shard": idx,
                            "shard_size": cfg.sample.shard_size}),
                encoding="utf-8")
            run += 1
            if progress is not None:
                progress(partition, idx, len(out))

    frames = []
    for part in sorted(parts_dir.glob("*.parquet")):
        if part.with_suffix(".DONE").exists():
            frames.append(pd.read_parquet(part))
    if not frames:
        raise NotProbed(f"partition {partition}: no completed signal parts")
    merged = pd.concat(frames, ignore_index=True)
    # The backstop, and the same guard `consolidate` has had all along for
    # files.parquet. Its absence here was an inconsistency, not a judgement:
    # a parts directory can also carry rows from an aborted run under an older
    # draw, which no shard-size check would catch.
    dupes = merged["file_id"].duplicated()
    if dupes.any():
        raise RuntimeError(
            f"partition {partition}: {int(dupes.sum())} duplicate file_id(s) across "
            f"the signal parts, e.g. {merged.loc[dupes, 'file_id'].iloc[0]!r}. The "
            f"parts under {parts_dir} were written by runs that do not agree; delete "
            f"them and re-run")
    merged.to_parquet(table, index=False)
    return SignalResult(partition, len(rows), run, skipped,
                        int((~merged["signal_ok"]).sum()), table)


def load_signal(cfg: EdaConfig, partitions: Sequence[str] | None = None
                ) -> pd.DataFrame:
    """Every partition's `signal.parquet`, concatenated and joined to the M tier.

    The join is an inner one on `file_id` and that is the contract: an S-tier
    row with no census row cannot be labelled, and a census row outside the
    draw has no S-tier statistics. `missing_partitions` carries the partitions
    that have no signal table yet, the same way `load_files` does.
    """
    if isinstance(partitions, str):
        partitions = [partitions]
    partitions = list(partitions or cfg.partitions())
    frames, missing = [], []
    for partition in partitions:
        path = cfg.out / partition / SIGNAL_TABLE
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(partition)
    if not frames:
        raise RuntimeError(
            f"no {SIGNAL_TABLE} under {cfg.out} for partition(s) {partitions}. "
            f"Run `eda signal` first")
    signal = pd.concat(frames, ignore_index=True)
    files = load_files(cfg, [p for p in partitions if p not in missing])
    merged = signal.merge(files.drop(columns=["source_name"]), on="file_id",
                          how="inner", validate="one_to_one")
    merged.attrs["missing_partitions"] = missing
    return merged
