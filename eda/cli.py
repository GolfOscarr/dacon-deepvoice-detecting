"""`python -m eda <verb>`.

Exit status is the gate, the same way `scripts/train.py`'s is: **0 only when no
gate reports `fail`**. A `na` does not fail the run -- Phase 0 cannot answer
every gate -- but it is printed as `na` and never as a pass, so a pipeline
cannot bank a clean exit that was earned by checks that never ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from eda.analyze import duplicates as dup
from eda.analyze import grouping as grp
from eda.analyze import shortcut as sc
from eda.analyze import signal as sig
from eda import groupkeys
from eda.config import load_eda_config
from eda.groupkeys import key_report
from eda.driver import (BlockedSource, NotProbed, consolidate, load_files,
                        load_signal, probe_source, signal_partition)
from eda.sample import SampleExists, draw, load_sample
from eda.gates import FAIL, run_gates

DEFAULT_CONFIG = "configs/eda.yaml"


def _show(df: pd.DataFrame, max_rows: int = 60) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.max_colwidth", 90, "display.max_rows", max_rows):
        print(df.to_string(index=False))


def cmd_sources(cfg, args) -> int:
    rows = [{"name": s.name, "row_kind": s.row_kind, "pool": s.pool, "cell": s.cell,
             "partition": s.partition, "root": s.root, "stage": s.stage,
             # ⚠️ `cfg.source_root`, not `cfg.root / root`. A `stage: raw`
             # source lives in the sync tree, and hardcoding the interim one
             # reported `mlaad` as absent while 16,006 of its files were in the
             # census and 2,000 had just been decoded.
             "exists": cfg.source_root(s).exists(),
             "excludes": len(s.exclude), "allowlist": bool(s.allowlist)}
            for s in cfg.sources_in(args.partition)]
    _show(pd.DataFrame(rows))
    return 0


def cmd_probe(cfg, args) -> int:
    selected = ([cfg.source(n) for n in args.source] if args.source
                else cfg.sources_in(args.partition))
    if not selected:
        print("no sources selected", file=sys.stderr)
        return 1
    def progress(name, idx, rows):
        print(f"  [{name}] shard {idx:05d}: {rows} rows", flush=True)

    probed, absent, blocked = 0, [], []
    for source in selected:
        # 🔴 Three states, and only one of them is fatal. A staged fetch means
        # most sources are legitimately **absent** for most of the run, and
        # dying on the first one makes wave-by-wave probing impossible -- which
        # is the whole shape of docs/EDA/08. **Blocked** is a recorded decision.
        # **Empty** stays fatal: a source that exists and matches nothing is the
        # silent failure `enumerate_source` exists to refuse.
        try:
            res = probe_source(cfg, source, dry_run=args.dry_run, progress=progress)
        except FileNotFoundError:
            absent.append(source.name)
            continue
        except BlockedSource as exc:
            blocked.append(f"{source.name}: {str(exc).split(': ', 1)[-1][:80]}")
            continue
        probed += 1
        verb = "would probe" if args.dry_run else "probed"
        print(f"{verb} {res.source}: {res.files} files, "
              f"{res.shards_run} shard(s) run, {res.shards_skipped} already done")

    # Critical: counted and printed, never just omitted. `fetch_from_s3.py` ends
    # the same way, and for the same reason -- a skipped source that leaves no
    # trace is indistinguishable from one that was never registered.
    print(f"\n{probed} probed, {len(absent)} absent, {len(blocked)} blocked")
    if absent:
        print(f"  absent (not fetched yet): {', '.join(sorted(absent))}")
    for b in blocked:
        print(f"  blocked {b}")
    if not probed:
        print("nothing was probed", file=sys.stderr)
        return 1
    return 0


def cmd_consolidate(cfg, args) -> int:
    partitions = [args.partition] if args.partition else cfg.partitions()
    done, skipped = 0, []
    for partition in partitions:
        try:
            out = consolidate(cfg, partition)
        except NotProbed:
            # Expected while the fetch is staged: `cell8` has no parts until
            # SONICS lands. Counted and named, never silently omitted.
            skipped.append(partition)
            continue
        except RuntimeError as exc:
            print(f"partition {partition}: {exc}", file=sys.stderr)
            return 1
        done += 1
        # Critical: a blocked source whose parts are still on disk is skipped by
        # `consolidate`, and the count would otherwise drop with nothing saying
        # why. Naming it here is the difference between "the census shrank" and
        # "we dropped `rirs-pointsource` and here is the row count without it".
        dropped = [s.name for s in cfg.sources_in(partition) if s.blocked]
        note = f"; {len(dropped)} blocked and skipped: {', '.join(sorted(dropped))}" if dropped else ""
        print(f"partition {partition}: {out} ({len(pd.read_parquet(out))} rows){note}")
    print(f"\n{done} consolidated, {len(skipped)} not probed yet"
          + (f": {', '.join(skipped)}" if skipped else ""))
    return 0 if done else 1


#: Written by `analyze`, read by `gates`. Named once so the two verbs cannot
#: drift apart over a filename.
SHORTCUT_TABLE = "shortcut_audit.parquet"
GROUPING_TABLE = "grouping_report.parquet"
DEPTH_TABLE = "depth_profile.parquet"
DUPLICATE_TABLE = "duplicates.parquet"
DUPLICATE_SUMMARY = "duplicates_summary.json"


def _analyses(cfg, files: pd.DataFrame) -> dict:
    try:
        duplicates = dup.cross_source_report(files)
    except dup.NoHashes as exc:
        # `None` is what `eda.gates` reads as "the sweep did not run". Letting
        # this crash would lose the other three analyses; swallowing it into an
        # empty result would report `pass` for a check nobody performed.
        print(f"warning: duplicate sweep skipped -- {exc}", file=sys.stderr)
        duplicates = None
    return {
        "shortcut": sc.shortcut_audit(files, cfg.analysis),
        "grouping": grp.grouping_report(files, cfg.analysis, cfg.gates),
        "duplicates": duplicates,
        "depth": grp.depth_profile(files, cfg.analysis),
    }


def cmd_sample(cfg, args) -> int:
    """Draw the S-tier sample, or print the draw that already exists.

    Separate from `signal` on purpose. The draw decides every S-tier number
    anyone will quote, so it is its own step with its own output, and a reader
    can see what was drawn before a day of decoding starts.
    """
    partitions = [args.partition] if args.partition else cfg.partitions()
    drawn = 0
    for partition in partitions:
        try:
            files = load_files(cfg, partition)
        except RuntimeError as exc:
            print(f"partition {partition}: {exc}", file=sys.stderr)
            continue
        try:
            sample = draw(cfg, files, partition, force=args.redraw)
        except SampleExists:
            existing = load_sample(cfg, partition)
            print(f"partition {partition}: {existing.n_drawn} of "
                  f"{existing.n_population} already drawn (seed {existing.seed}"
                  f"{', full' if existing.full else ''}) -- --redraw to replace")
            continue
        drawn += 1
        print(f"partition {partition}: drew {sample.n_drawn} of "
              f"{sample.n_population}"
              + (" (measured in full)" if sample.full
                 else f" at {sample.per_stratum}/stratum by "
                      f"{list(sample.stratify_by)}"
                      # The spread decides *which* files, so it belongs in the
                      # line a reader copies into a note, not only in the JSON.
                      + (f" spread evenly over {list(sample.spread_by)}"
                         if sample.spread_by else "")
                      + f", seed {sample.seed}"))
    print(f"\n{drawn} partition(s) drawn")
    return 0


def cmd_signal(cfg, args) -> int:
    """S tier: decode the draw on both planes."""
    partitions = [args.partition] if args.partition else cfg.partitions()
    done, skipped = 0, []
    for partition in partitions:
        try:
            res = signal_partition(
                cfg, partition,
                progress=lambda part, idx, rows: print(
                    f"  [{part}] shard {idx:05d}: {rows} rows", flush=True))
        except (NotProbed, RuntimeError) as exc:
            skipped.append(partition)
            print(f"partition {partition}: skipped -- {exc}", file=sys.stderr)
            continue
        done += 1
        # ⚠️ Failures are printed even when zero. A decode-failure count that
        # only appears when it is non-zero trains the reader to skim past the
        # line, and F-S2 says corruption is a finding.
        print(f"partition {partition}: {res.table} ({res.files} rows, "
              f"{res.shards_run} shard(s) run, {res.shards_skipped} already done, "
              f"{res.failures} decode failure(s))")
    print(f"\n{done} partition(s) measured, {len(skipped)} skipped"
          + (f": {', '.join(skipped)}" if skipped else ""))
    return 0 if done else 1


def cmd_keys(cfg, args) -> int:
    """Per source: the grouping key, how many there are, and the evidence.

    Its own verb rather than more output on `analyze` because it is the thing a
    reader checks when a fold table looks wrong, and `analyze` is a three-minute
    run over the whole census. This reads `files.parquet` and the publishers'
    metadata; nothing here decodes.
    """
    files = load_files(cfg, args.partition)
    report = key_report(cfg, files)
    with pd.option_context("display.width", 200, "display.max_colwidth", 88,
                           "display.max_rows", None):
        print(report.to_string(index=False))
    short = report[report["group_key_kind"] != groupkeys.PATH]
    print(f"\n{len(short)} of {len(report)} source(s) carry a key of the "
          f"publisher's own; the rest fall back to path depth "
          f"(`eda analyze` prints which depth)")
    return 0


def cmd_report(cfg, args) -> int:
    """The S tier, read: docs/EDA/09 step 6.

    Its own verb rather than more output on `analyze`, for the reason `keys` is:
    `analyze` is a multi-minute pass over 382,068 census rows, and this reads a
    58,885-row table that is already on disk. Nothing here decodes.
    """
    try:
        signal = load_signal(cfg, args.partition)
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    out = cfg.out / "_shared"
    out.mkdir(parents=True, exist_ok=True)
    # 🔴 A partition-scoped run writes partition-scoped files. Sharing one name
    # means `report --partition D` silently replaces the corpus-wide table with
    # a one-row version, and every number quoted from it afterwards is a pool-D
    # number wearing a corpus label. Measured: it happened, during the smoke
    # test for this verb.
    scope = f"_{args.partition}" if args.partition else ""

    ok = signal["signal_ok"].fillna(False)
    print(f"=== S tier: {len(signal)} rows, {int(ok.sum())} decoded, "
          f"{len(signal) - int(ok.sum())} failure(s); "
          f"partitions {sorted(signal['partition'].unique())}")

    tables = {
        "duration": (sig.duration_report(signal),
                     "duration and silence, against the test set's 4-60 s window"),
        "bandwidth": (sig.bandwidth_report(signal),
                      "what the 16 kHz chain removes, paired per file"),
        "level": (sig.level_report(signal),
                  "loudness, headroom, DC and clipping on the chain plane"),
    }
    for name, (table, caption) in tables.items():
        table.to_parquet(out / f"signal_{name}{scope}.parquet", index=False)
        print(f"\n=== {name}: {caption}")
        _show(table)
    print(f"\nwrote {len(tables)} table(s) to {out}")
    return 0


def cmd_analyze(cfg, args) -> int:
    files = load_files(cfg)
    out = cfg.out / "_shared"
    out.mkdir(parents=True, exist_ok=True)
    res = _analyses(cfg, files)

    kinds = files["row_kind"].value_counts().to_dict()
    print(f"\n=== files: {len(files)} rows, {files['source_name'].nunique()} sources; "
          f"{kinds}")
    print(f"    pools {sorted(files['pool'].dropna().unique())} · "
          f"cells {sorted(int(c) for c in files['cell'].dropna().unique())}")
    ok = files["probe_ok"].fillna(False)
    print(f"    probe_ok {int(ok.sum())}/{len(files)}; "
          f"{int((~ok).sum())} unreadable (recorded, not dropped)")
    absent = files.attrs.get("missing_partitions") or []
    if absent:
        print(f"    ⚠️  partition(s) {absent} are declared but have no "
              f"files.parquet. Every number below is over a partial corpus")

    print("\n=== X1 shortcut audit (metadata only)")
    _show(res["shortcut"])
    res["shortcut"].to_parquet(out / SHORTCUT_TABLE, index=False)

    print("\n=== grouping atoms (path-derived; a suggestion, not an assignment)")
    _show(res["grouping"])
    res["grouping"].to_parquet(out / GROUPING_TABLE, index=False)
    res["depth"].to_parquet(out / DEPTH_TABLE, index=False)

    d = res["duplicates"]
    if d is None:
        print("\n=== E1 duplicates: NOT RUN (no sha256 column)")
    else:
        print(f"\n=== E1 duplicates: {d['groups']} group(s) over {d['rows']} rows; "
              f"{d['cross_source_groups']} cross-source, "
              f"{d['cross_pool_groups']} cross-pool")
        if len(d["pairs"]):
            _show(d["pairs"].head(20))
        # Critical: the summary is written unconditionally. A clean sweep that
        # left no artifact was indistinguishable from a sweep that never ran,
        # and `eda gates` then reported `na` for what was a pass.
        (out / DUPLICATE_SUMMARY).write_text(
            json.dumps(dup.summary_of(d), indent=2), encoding="utf-8")
        if len(d["detail"]):
            d["detail"].to_parquet(out / DUPLICATE_TABLE, index=False)

    gates = run_gates(cfg, files, audit=res["shortcut"], groups=res["grouping"],
                      dupes=None if d is None else dup.summary_of(d))
    print("\n=== gates")
    _show(gates)
    gates.to_parquet(out / "gates.parquet", index=False)
    agg = gates.attrs["aggregate"]
    print(f"\naggregate: {agg}")
    return 1 if (gates["verdict"] == FAIL).any() else 0


def cmd_gates(cfg, args) -> int:
    """Re-run the gates over what `analyze` already wrote.

    Reads only; it must not silently recompute an analysis, because a gate
    reading a freshly computed input would report `pass` for a check the
    recorded run never performed.
    """
    files = load_files(cfg)
    shared = cfg.out / "_shared"

    def table(name):
        p = shared / name
        return pd.read_parquet(p) if p.exists() else None

    summary_path = shared / DUPLICATE_SUMMARY
    dupes = (json.loads(summary_path.read_text(encoding="utf-8"))
             if summary_path.exists() else None)
    gates = run_gates(cfg, files, audit=table(SHORTCUT_TABLE),
                      groups=table(GROUPING_TABLE), dupes=dupes)
    _show(gates)
    print(f"\naggregate: {gates.attrs['aggregate']}")
    return 1 if (gates["verdict"] == FAIL).any() else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eda", description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    sub = p.add_subparsers(dest="verb", required=True)

    #: Partition, not pool: a whole_file source has no pool to partition by and
    #: writes to `cell<N>/` instead. `eda sources` prints both columns.
    part_help = "output partition: a pool (A-E, mixed) or a cell (cell5, cell8)"

    s = sub.add_parser("sources", help="what the config declares, and whether it is on disk")
    s.add_argument("--partition", help=part_help)
    s.set_defaults(fn=cmd_sources)

    s = sub.add_parser("probe", help="M tier: ffprobe + sha256 over 100%%, resumable")
    s.add_argument("--partition", help=part_help)
    s.add_argument("--source", nargs="*", default=[])
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("consolidate", help="parts -> files.parquet")
    s.add_argument("--partition", help=part_help)
    s.set_defaults(fn=cmd_consolidate)

    s = sub.add_parser("sample", help="draw the S-tier sample and record it")
    s.add_argument("--partition", help=part_help)
    s.add_argument("--redraw", action="store_true",
                   help="replace an existing draw. Every S-tier number published "
                        "from the old one becomes unreproducible")
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("signal", help="S tier: decode the draw on both planes")
    s.add_argument("--partition", help=part_help)
    s.set_defaults(fn=cmd_signal)

    s = sub.add_parser("keys", help="the publisher's own grouping key, per source")
    s.add_argument("--partition", help=part_help)
    s.set_defaults(fn=cmd_keys)

    s = sub.add_parser("report", help="read the S tier: duration, bandwidth, level")
    s.add_argument("--partition", help=part_help)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("analyze", help="X1 + E1 + grouping, then the gates")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("gates", help="re-read the saved analyses and re-run the gates")
    s.set_defaults(fn=cmd_gates)

    args = p.parse_args(argv)
    return args.fn(load_eda_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
