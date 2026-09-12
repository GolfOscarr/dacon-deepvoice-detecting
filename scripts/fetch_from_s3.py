#!/usr/bin/env python3
"""Pull the raw corpus out of S3 onto a training machine, and verify it.

    python3 scripts/fetch_from_s3.py --list
    python3 scripts/fetch_from_s3.py --all --dest /data/raw
    python3 scripts/fetch_from_s3.py mlaad musdb18-hq --dest /data/raw
    python3 scripts/fetch_from_s3.py --pool B --dest /data/raw --extract /data/interim
    python3 scripts/fetch_from_s3.py --all --dest /data/raw --verify-only

The inverse of `fetch_to_s3.py`, and deliberately not its mirror image.

*It is driven by S3, not by the registry.* `scripts/sources.yaml` says what we
*intend* to hold; the bucket says what we *actually* hold. A training machine
should act on the second, and this way the only thing it needs is AWS
credentials -- not a matching checkout of this repo.

*Verification is the default, not a flag.* Every source carries
`_meta/checksums.sha256`, written from the bytes the publisher served. This
script re-hashes what it downloaded and compares. That is not ceremony: a
two-writer collision during acquisition once produced a file 125% of its true
size, and on a different day would have produced one of exactly the right size
with the wrong contents. Size is not evidence; the hash is.

*A source without `_meta/DONE` is skipped.* The marker is written last, after
the payload and the checksums, so its absence means the upload was interrupted
and the payload may be short. `--include-incomplete` overrides that for
inspection, and says so loudly.

Costs: reading out of the bucket is billed as egress, and training reads the
corpus repeatedly. Pull once to local disk and keep it -- `--verify-only`
re-checks what is already there without transferring anything. Where the bucket
and the training machine are in different regions, co-locating them is worth
more than any tuning in here.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Sequence

DEFAULT_BUCKET = "hyeonseop-s3"
DEFAULT_PREFIX = "dacon-deepfake-detection/data/raw"
CHUNK = 1 << 22


# --------------------------------------------------------------------------
# discovering what the bucket holds
# --------------------------------------------------------------------------

def s3_ls(uri: str) -> list[tuple[str, int]]:
    """`[(key, size)]` under a prefix, recursively."""
    proc = subprocess.run(["aws", "s3", "ls", "--recursive", uri],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"aws s3 ls failed: {proc.stderr.strip()[:300]}")
    out = []
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            out.append((parts[3], int(parts[2])))
    return out


def discover(bucket: str, prefix: str) -> dict[str, dict]:
    """`name -> {version, bytes, files, complete}` for everything in the store.

    Critical: `complete` is the presence of `_meta/DONE`, which fetch_to_s3
    writes only after the payload and checksums are both up. Treating a
    payload-shaped prefix as complete would hand training a truncated archive.
    """
    root = f"s3://{bucket}/{prefix.rstrip('/')}/"
    entries = s3_ls(root)
    base = prefix.rstrip("/") + "/"
    sources: dict[str, dict] = {}
    for key, size in entries:
        rel = key[len(base):] if key.startswith(base) else key
        bits = rel.split("/")
        if len(bits) < 3:
            continue
        name, version, rest = bits[0], bits[1], "/".join(bits[2:])
        s = sources.setdefault(name, {"version": version, "bytes": 0,
                                      "files": [], "complete": False})
        if rest == "_meta/DONE":
            s["complete"] = True
        elif rest.startswith("payload/"):
            s["bytes"] += size
            s["files"].append(rest[len("payload/"):])
    return sources


def read_meta(bucket: str, prefix: str, name: str, version: str,
              fname: str) -> str | None:
    """One `_meta/` file as text, or None when it is absent."""
    uri = f"s3://{bucket}/{prefix.rstrip('/')}/{name}/{version}/_meta/{fname}"
    proc = subprocess.run(["aws", "s3", "cp", uri, "-"],
                          capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def parse_checksums(text: str) -> dict[str, str]:
    """`sha256sum` format -> `{filename: digest}`.

    Caveat: the filename may contain spaces, so split only on the first run of
    whitespace and keep the rest verbatim.
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, fname = line.partition("  ")
        if not fname:
            digest, _, fname = line.partition(" ")
        if digest and fname:
            out[fname.strip()] = digest.strip()
    return out


# --------------------------------------------------------------------------
# local side
# --------------------------------------------------------------------------

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float | None) -> str:
    if not n:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return "?"


def selected(names, only: Sequence[str]) -> list[str]:
    """The payload-relative names matching any `--only` glob; all of them when
    no glob is given.

    Critical: the same function filters the sync and the verification. If the
    two ever disagree, a partial fetch reports every unfetched file as
    "missing" -- a red run for a corpus that is exactly what was asked for --
    and the reader learns to ignore the one check that matters.
    """
    if not only:
        return list(names)
    return [n for n in names if any(fnmatch.fnmatch(n, g) for g in only)]


def verify_dir(payload: pathlib.Path, want: dict[str, str]) -> tuple[list, list]:
    """`(ok, bad)` for the files a checksum manifest names.

    A file the manifest does not name is not checked and not reported -- the
    manifest is the authority on what should be there, and `aws s3 sync` may
    legitimately leave unrelated files alone.
    """
    ok, bad = [], []
    for fname, digest in sorted(want.items()):
        p = payload / fname
        if not p.exists():
            bad.append((fname, "missing"))
        elif sha256_file(p) != digest:
            bad.append((fname, "sha256 mismatch"))
        else:
            ok.append(fname)
    return ok, bad


def sync_down(bucket: str, prefix: str, name: str, version: str,
              dest: pathlib.Path, dry_run: bool,
              only: Sequence[str] = ()) -> None:
    uri = f"s3://{bucket}/{prefix.rstrip('/')}/{name}/{version}/"
    cmd = ["aws", "s3", "sync", uri, str(dest), "--only-show-errors"]
    # `--only` is a *partial* fetch: exclude everything, then re-include the
    # payload paths asked for. CompSpoof is the case it exists for -- 111.8 GB
    # in the store, of which pool E needs the two `*_source.tar.gz` that carry
    # `env_sources/`. Egress is billed and the rest is speech and mixtures we
    # already have better sources for.
    # Critical: ONE `--exclude "*"` first, then every include. aws applies
    # filters in the order given, so a second `--exclude "*"` after an include
    # cancels it and the sync transfers nothing at all.
    if only:
        cmd += ["--exclude", "*"]
        for glob in only:
            cmd += ["--include", f"payload/{glob}"]
    if dry_run:
        cmd.append("--dryrun")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"sync of {name} failed: {proc.stderr.strip()[:300]}")
    if dry_run and proc.stdout.strip():
        print("\n".join("      " + l for l in proc.stdout.strip().splitlines()[:6]))


# --------------------------------------------------------------------------
# split archives
# --------------------------------------------------------------------------

#: A `split -b` piece: a two-letter tail after something archive-shaped.
_SPLIT_TAIL = re.compile(r"^(?P<base>.+\.(?:tar|tgz|tar\.gz|tar\.bz2|tar\.xz|zip))"
                         r"\.(?P<piece>[a-z]{2})$", re.IGNORECASE)
#: A spanned zip piece: `name.z01` .. `name.zNN`, whose last part is `name.zip`.
_SPANNED_ZIP = re.compile(r"^(?P<base>.+)\.z(?P<piece>\d{2})$", re.IGNORECASE)


def split_groups(files: Sequence[pathlib.Path]) -> dict[pathlib.Path, list[pathlib.Path]]:
    """`{whole archive -> its pieces, in order}` for the split sets in `files`.

    Two conventions, because publishers use both and neither is guessable from
    one filename:

    * **`split -b`** -- `database_eval.tar.gz.aa/.ab/.ac`. Concatenating the
      pieces *is* the archive (PartialSpoof, CompSpoof).
    * **spanned zip** -- `CFAD.z01/.z02/.z03` plus `CFAD.zip`, where the `.zip`
      is the **last** piece and holds the central directory. Concatenation is
      NOT the archive: the parts must be joined by `zip -s 0`, and Python's
      `zipfile` cannot read either the pieces or the set (CFAD, Codecfake).

    ⚠️ A group is only returned when it is **contiguous from the first piece**.
    A missing `.z02` would otherwise be joined into a corrupt archive that
    extracts partially and reports success, which is worse than not extracting.
    """
    by_base: dict[pathlib.Path, dict[str, pathlib.Path]] = {}
    kinds: dict[pathlib.Path, str] = {}
    for f in files:
        m = _SPLIT_TAIL.match(f.name)
        if m:
            base = f.with_name(m.group("base"))
            by_base.setdefault(base, {})[m.group("piece").lower()] = f
            kinds[base] = "cat"
            continue
        m = _SPANNED_ZIP.match(f.name)
        if m:
            base = f.with_name(m.group("base") + ".zip")
            by_base.setdefault(base, {})[m.group("piece")] = f
            kinds[base] = "zip"
    out: dict[pathlib.Path, list[pathlib.Path]] = {}
    for base, pieces in by_base.items():
        if kinds[base] == "cat":
            want = [f"{a}{b}" for a in "abcdefghijklmnopqrstuvwxyz"
                    for b in "abcdefghijklmnopqrstuvwxyz"]
            ordered, i = [], 0
            while i < len(want) and want[i] in pieces:
                ordered.append(pieces[want[i]]); i += 1
            if len(ordered) != len(pieces):
                continue                       # a hole: leave it to the warning
            out[base] = ordered
        else:
            # The `.zip` last part must exist, or there is no central directory
            # and nothing can be joined.
            if not base.exists():
                continue
            ordered, i = [], 1
            while f"{i:02d}" in pieces:
                ordered.append(pieces[f"{i:02d}"]); i += 1
            if len(ordered) != len(pieces):
                continue
            out[base] = ordered + [base]
    return out


def join_split(base: pathlib.Path, pieces: Sequence[pathlib.Path],
               work: pathlib.Path) -> pathlib.Path:
    """Reassemble one split archive under `work` and return the joined path.

    Costs a second copy of the archive on disk, which is why it goes to `work`
    rather than beside the payload -- the caller can delete it after extraction
    and keep the (verified) pieces.
    """
    work.mkdir(parents=True, exist_ok=True)
    if base.suffix.lower() == ".zip" and pieces and pieces[-1] == base:
        # `zip -s 0 in.zip --out joined.zip` is the documented way to turn a
        # spanned set back into one file. Python's zipfile cannot: the stdlib
        # has no multi-disk support, so this shells out or it does not happen.
        joined = work / f"{base.stem}.joined.zip"
        if joined.exists():
            return joined
        proc = subprocess.run(["zip", "-q", "-s", "0", str(base), "--out", str(joined)],
                              capture_output=True, text=True)
        if proc.returncode or not joined.exists():
            raise RuntimeError(
                f"joining spanned zip {base.name} failed: "
                f"{(proc.stderr or proc.stdout).strip()[:300]}. `zip` must be "
                f"installed; Python's zipfile has no multi-disk support")
        return joined
    joined = work / f"{base.name}.joined"
    if joined.exists():
        return joined
    with joined.open("wb") as fh:
        for piece in pieces:
            with piece.open("rb") as src:
                for block in iter(lambda: src.read(CHUNK), b""):
                    fh.write(block)
    return joined


def extract_archives(payload: pathlib.Path, out: pathlib.Path,
                     work: pathlib.Path | None = None, keep_joined: bool = False
                     ) -> int:
    """Unpack tar/zip archives into `out`. Returns how many were unpacked.

    Split sets are **joined first** (`split_groups`, `join_split`) and the
    joined copy is deleted afterwards unless `keep_joined`. Four of the sources
    in the store ship this way and none of them could be extracted before.

    Caveat: members whose path escapes `out` are skipped rather than written.
    These archives come from third parties, and a `../` member would otherwise
    write outside the extraction root.
    """
    out.mkdir(parents=True, exist_ok=True)
    work = work or out.parent / "_joining"
    n = 0
    unpacked: list[pathlib.Path] = []
    files = [q for q in sorted(payload.rglob("*")) if q.is_file()]
    groups = split_groups(files)
    consumed = {q for pieces in groups.values() for q in pieces}
    joined_paths: list[pathlib.Path] = []
    for base, pieces in sorted(groups.items()):
        try:
            joined = join_split(base, pieces, work)
        except Exception as exc:                       # noqa: BLE001
            print(f"      [warn] {base.name}: {exc}")
            continue
        print(f"      [join] {base.name}: {len(pieces)} piece(s) -> {joined.name}")
        joined_paths.append(joined)
    # Critical: **recursive**. `iterdir()` misses any publisher that nests its
    # archives, and SONICS does exactly that -- its ten zips live under
    # `payload/fake_songs/`, so a non-recursive walk yielded the *directory*,
    # `is_file()` was False, and 30 GiB extracted to nothing while the run
    # reported success.
    for p in files + joined_paths:
        # A piece of a split set is not an archive on its own; the joined file
        # stands in for the whole group.
        if p in consumed:
            continue
        try:
            if tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    safe = [m for m in tf.getmembers() if _inside(out, out / m.name)]
                    # filter="data" is the 3.14 default and refuses absolute
                    # paths, links out of the tree and device nodes. Kept
                    # alongside the _inside() check rather than instead of it:
                    # one is stdlib policy, the other is ours, and these are
                    # third-party archives.
                    tf.extractall(out, members=safe, filter="data")
                    n += 1
            elif zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    safe = [m for m in zf.namelist() if _inside(out, out / m)]
                    zf.extractall(out, members=safe)
                    n += 1
            else:
                unpacked.append(p)
                continue
        except Exception as exc:                       # noqa: BLE001
            print(f"      [warn] {p.name}: {exc}")
    # Critical: a file that looks like an archive and was not unpacked is
    # reported. Split archives are the live case -- CompSpoof ships
    # `development.tar.gz.part_aa..ae`, and neither `is_tarfile` nor
    # `is_zipfile` recognises a part, so each is skipped. Silently returning a
    # smaller `n` is the "validates and does nothing" failure: the caller sees a
    # successful extraction of a corpus that is not there.
    if not keep_joined:
        for q in joined_paths:
            q.unlink(missing_ok=True)
        if work.exists() and not any(work.iterdir()):
            work.rmdir()
    missed = [p for p in unpacked if _looks_like_archive(p)]
    if missed:
        print(f"      [warn] {len(missed)} archive-looking file(s) not unpacked "
              f"(split archive?): {', '.join(p.name for p in missed[:4])}")
    return n


#: Suffix patterns that mean "this was meant to be unpacked". Deliberately not
#: a general guess: payloads legitimately contain README.md, LICENSE and CSVs,
#: and warning about those would train the reader to ignore the warning.
_ARCHIVE_HINTS = (".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar")


def _looks_like_archive(p: pathlib.Path) -> bool:
    name = p.name.lower()
    if ".part_" in name or re.search(r"\.z\d{2}$|\.\d{3}$", name):
        return True
    if any(name.endswith(s) for s in _ARCHIVE_HINTS):
        return True
    # 🔴 `split -b` names its pieces `.aa`, `.ab`, ... with no digits and no
    # hint of their own, so the checks above miss them entirely. PartialSpoof
    # ships `database_eval.tar.gz.aa/.ab/.ac` -- 5.4 GB, the whole eval set --
    # and it was skipped silently, which is the SONICS defect wearing different
    # letters. Only treat a two-letter tail as a split piece when what precedes
    # it is itself archive-shaped, so an `.srt.en` or a `.model.pt` is not
    # mistaken for one.
    stem, _, tail = name.rpartition(".")
    return (len(tail) == 2 and tail.isalpha()
            and any(stem.endswith(s) for s in _ARCHIVE_HINTS))


def _inside(root: pathlib.Path, target: pathlib.Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def print_table(sources: dict[str, dict], meta: dict[str, dict]) -> None:
    print(f"{'SOURCE':22s} {'VERSION':16s} {'POOL':5s} {'FILES':>7s} {'SIZE':>10s}  STATE")
    print("-" * 82)
    total = 0
    for name in sorted(sources):
        s = sources[name]
        m = meta.get(name) or {}
        state = "complete" if s["complete"] else "INCOMPLETE (no DONE)"
        total += s["bytes"] if s["complete"] else 0
        print(f"{name:22s} {s['version']:16s} {str(m.get('pool','?')):5s} "
              f"{len(s['files']):7d} {human(s['bytes']):>10s}  {state}")
    print("-" * 82)
    done = sum(1 for s in sources.values() if s["complete"])
    print(f"{done}/{len(sources)} complete, {human(total)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("names", nargs="*", help="source names; default is none")
    p.add_argument("--all", action="store_true", help="every complete source")
    p.add_argument("--list", action="store_true", help="show the store and exit")
    p.add_argument("--pool", help="only sources in this pool (A-E, or 'mixed')")
    p.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("data/raw"))
    p.add_argument("--extract", type=pathlib.Path,
                   help="also unpack archives under this directory")
    p.add_argument("--only", action="append", default=[], metavar="GLOB",
                   help="payload paths matching this glob only; repeatable. A "
                        "partial fetch: DONE still means the upload was "
                        "complete, but your copy deliberately is not")
    p.add_argument("--verify-only", action="store_true",
                   help="re-hash what is already local; transfer nothing")
    p.add_argument("--dry-run", action="store_true", help="show what would transfer")
    p.add_argument("--include-incomplete", action="store_true",
                   help="also take sources with no DONE marker (may be truncated)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = p.parse_args()

    try:
        sources = discover(args.bucket, args.prefix)
    except Exception as exc:                           # noqa: BLE001
        print(f"cannot read s3://{args.bucket}/{args.prefix}: {exc}", file=sys.stderr)
        return 2
    if not sources:
        print(f"nothing under s3://{args.bucket}/{args.prefix}")
        return 0

    meta: dict[str, dict] = {}
    for name, s in sources.items():
        raw = read_meta(args.bucket, args.prefix, name, s["version"], "acquisition.json")
        if raw:
            try:
                meta[name] = json.loads(raw)
            except json.JSONDecodeError:
                pass

    if args.list or not (args.names or args.all or args.pool):
        print_table(sources, meta)
        if not args.list:
            print("\nNothing selected. Use --all, --pool X, or name sources.")
        return 0

    if args.names:
        unknown = [n for n in args.names if n not in sources]
        if unknown:
            print(f"not in the store: {unknown}", file=sys.stderr)
            return 2
        wanted = list(args.names)
    else:
        wanted = sorted(sources)
    if args.pool:
        wanted = [n for n in wanted
                  if str((meta.get(n) or {}).get("pool", "")) == args.pool]
        if not wanted:
            print(f"no sources in pool {args.pool!r}")
            return 0

    print(f"source  s3://{args.bucket}/{args.prefix}/")
    print(f"dest    {args.dest}\n")

    failures, skipped = [], []
    for name in wanted:
        s = sources[name]
        if not s["complete"] and not args.include_incomplete:
            skipped.append(name)
            print(f"  [skip] {name}: no DONE marker -- upload was interrupted, "
                  f"payload may be short (--include-incomplete to take it anyway)")
            continue
        if not s["complete"]:
            print(f"  [warn] {name}: taking an INCOMPLETE source on request")

        local = args.dest / name / s["version"]
        if not args.verify_only:
            take = selected(s["files"], args.only)
            if args.only and not take:
                print(f"  [skip] {name}: no payload file matches {args.only}")
                skipped.append(name)
                continue
            size = s["bytes"] if not args.only else None
            print(f"  [get ] {name}/{s['version']}  "
                  f"{human(size) if size else '(partial)'}, {len(take)} file(s)"
                  + (f" of {len(s['files'])}" if args.only else ""))
            try:
                sync_down(args.bucket, args.prefix, name, s["version"], local,
                          args.dry_run, args.only)
            except Exception as exc:                   # noqa: BLE001
                print(f"  [FAIL] {name}: {exc}", file=sys.stderr)
                failures.append(name)
                continue
        if args.dry_run:
            continue

        text = read_meta(args.bucket, args.prefix, name, s["version"], "checksums.sha256")
        if not text:
            print(f"  [warn] {name}: no checksums.sha256 in the store -- cannot verify")
        else:
            # Critical: verify only what was asked for. Handing the full
            # manifest to a partial fetch reports every file it deliberately
            # did not take as "missing", which is a FAIL for a correct run.
            want = parse_checksums(text)
            want = {k: v for k, v in want.items() if k in set(selected(want, args.only))}
            ok, bad = verify_dir(local / "payload", want)
            if bad:
                for fname, why in bad[:5]:
                    print(f"  [BAD ] {name}/{fname}: {why}", file=sys.stderr)
                print(f"  [FAIL] {name}: {len(bad)} of {len(ok) + len(bad)} "
                      f"file(s) failed verification", file=sys.stderr)
                failures.append(name)
                continue
            print(f"  [ok  ] {name}: {len(ok)} file(s) verified against sha256")

        if args.extract:
            n = extract_archives(local / "payload", args.extract / name / s["version"])
            print(f"  [xtr ] {name}: {n} archive(s) unpacked")

    print(f"\n{len(wanted) - len(failures) - len(skipped)} ready, "
          f"{len(failures)} failed, {len(skipped)} skipped")
    if failures:
        print("Failed verification is not a warning: do not train on these.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
