#!/usr/bin/env python3
"""Acquire a dataset into the S3 raw store, with its provenance.

    python3 scripts/fetch_to_s3.py --list
    python3 scripts/fetch_to_s3.py --plan                    # what would run
    python3 scripts/fetch_to_s3.py --all                     # every ready source
    python3 scripts/fetch_to_s3.py musan rirs-noises         # named sources

The layout it writes, per source, is deliberately boring:

    <prefix>/<name>/<version>/payload/<file>          bytes exactly as served
    <prefix>/<name>/<version>/_meta/acquisition.json  provenance
    <prefix>/<name>/<version>/_meta/checksums.sha256
    <prefix>/<name>/<version>/_meta/DONE              idempotence marker

Three properties are load-bearing and each one is here for a reason:

*Archives are not unpacked.* The sha256 we record is the sha256 of what the
publisher served, so the file we hand DACON at the 2nd-stage review is provably
the file we were given. Unpacking happens downstream, into `interim/`.

*The licence gate is data, not a comment.* `scripts/sources.yaml` carries a
`redistribution` verdict per source and this script refuses anything that is not
`ok`. DACON requires submitting the actual training files (#417280) and answered
that data whose licence bars third-party provision cannot be used at all -- so a
download that outruns its licence verdict is not a shortcut, it is rework.
`--i-have-verified` exists for the moment somebody actually does the review, and
it names the source explicitly so it cannot be set once and forgotten.

*Resumable, idempotent, and it does not need disk for the whole corpus.* Each
artifact is staged to a scratch directory, hashed, uploaded, then deleted, so
peak local disk is one artifact rather than 400 GB. A re-run skips anything
whose DONE marker is already in S3.

Caveat: this writes the *source*-level ledger. The per-file ledger that
`training/manifest.py` consumes needs track/artist/speaker granularity, which is
built when archives are unpacked -- and which docs/data/08 records as the one
Phase B decision with no second chance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.parse

REGISTRY = pathlib.Path(__file__).resolve().parent / "sources.yaml"
DEFAULT_BUCKET = "hyeonseop-s3"
DEFAULT_PREFIX = "dacon-deepfake-detection/data/raw"

FETCHABLE_STATUS = "ready"
REDIST_OK = "ok"
VALID_STATUS = ("ready", "needs_terms", "needs_request", "needs_review", "blocked")
VALID_REDIST = ("ok", "review", "no")
VALID_KINDS = ("http", "mdc", "hf", "kaggle")

MDC_API = "https://mozilladatacollective.com/api"
CHUNK = 1 << 22          # 4 MiB: large enough that hashing is not the bottleneck


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Source:
    name: str
    version: str
    pool: str
    status: str
    redistribution: str
    licence: str
    licence_url: str
    verified_at_origin: bool
    fetch: dict
    why: str = ""
    caveat: str = ""
    caveat_compliance: str = ""
    requirement: str = ""
    blocker: str = ""
    verified_date: str = ""
    verified_note: str = ""
    defer: bool = False

    @property
    def slug(self) -> str:
        return f"{self.name}/{self.version}"

    def blocked_reason(self) -> str | None:
        """Why this source must not be fetched, or None if it may be.

        Critical: the two conditions are checked separately and both are
        reported. A source can be `ready` with a `review` verdict (we know how
        to get it, we have not cleared the licence) and the opposite is also
        possible, so collapsing them into one boolean loses the actionable half.
        """
        if self.redistribution != REDIST_OK:
            return (f"redistribution={self.redistribution!r} -- "
                    f"{self.blocker or self.requirement or 'no verdict recorded'}")
        if self.status != FETCHABLE_STATUS:
            return (f"status={self.status!r} -- "
                    f"{self.requirement or self.blocker or 'not ready'}")
        return None


def load_registry(path: pathlib.Path = REGISTRY) -> list[Source]:
    """Parse and validate the registry.

    Validation is not decoration: a typo in `redistribution` that silently read
    as "not ok" would quietly stop acquiring a source, and a typo that read as
    "ok" would quietly acquire one we have no right to ship.
    """
    import yaml

    doc = yaml.safe_load(path.read_text())
    if doc.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version {doc.get('schema_version')!r}")

    out: list[Source] = []
    seen: set[str] = set()
    for i, raw in enumerate(doc.get("sources") or []):
        where = f"{path}: sources[{i}]"
        missing = [k for k in ("name", "version", "pool", "status", "redistribution",
                               "licence", "licence_url", "fetch") if k not in raw]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        if raw["status"] not in VALID_STATUS:
            raise ValueError(f"{where}: status {raw['status']!r} not in {VALID_STATUS}")
        if str(raw["redistribution"]) not in VALID_REDIST:
            raise ValueError(f"{where}: redistribution {raw['redistribution']!r} "
                             f"not in {VALID_REDIST}")
        kind = (raw["fetch"] or {}).get("kind")
        if kind not in VALID_KINDS:
            raise ValueError(f"{where}: fetch.kind {kind!r} not in {VALID_KINDS}")

        fields = {f.name for f in dataclasses.fields(Source)}
        kwargs = {k: v for k, v in raw.items() if k in fields}
        kwargs["redistribution"] = str(raw["redistribution"])   # YAML reads `no` as False
        kwargs["verified_at_origin"] = bool(raw.get("verified_at_origin", False))
        src = Source(**kwargs)
        if src.slug in seen:
            raise ValueError(f"{where}: duplicate source {src.slug!r}")
        seen.add(src.slug)

        # A source we are allowed to ship must have had its licence read at
        # origin. Reported-from-a-secondary-source is how docs/data/11 describes
        # every row it lists, and it is explicitly not verification.
        if src.redistribution == REDIST_OK and not src.verified_at_origin:
            raise ValueError(f"{where}: {src.name} is redistribution=ok but "
                             f"verified_at_origin is false -- read the licence "
                             f"at {src.licence_url} and record it first")
        out.append(src)
    return out


# --------------------------------------------------------------------------
# S3 keys and paths -- pure, so the tests can pin them
# --------------------------------------------------------------------------

def s3_base(prefix: str, src: Source) -> str:
    return f"{prefix.rstrip('/')}/{src.name}/{src.version}"


def s3_uri(bucket: str, prefix: str, src: Source, *parts: str) -> str:
    """Root of the source's prefix, or a path under it. No trailing slash.

    Caveat: with no parts this must not end in `/` -- callers append their own
    separator, and a doubled one makes an S3 key with an empty path segment,
    which is legal, silently different, and impossible to spot in a listing.
    """
    return "/".join((f"s3://{bucket}/{s3_base(prefix, src)}", *parts))


def artifact_name(url: str) -> str:
    """Filename for a URL, refusing anything that would escape the prefix.

    Caveat: a publisher controls this string. `..` or a leading slash in a
    path segment would write outside the source's own prefix, so both are
    rejected rather than sanitised -- a silently renamed artifact would break
    the checksum-to-filename correspondence the ledger depends on.
    """
    name = pathlib.PurePosixPath(urllib.parse.urlparse(url).path).name
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"cannot derive a safe filename from {url!r}")
    return name


# --------------------------------------------------------------------------
# shell helpers
# --------------------------------------------------------------------------

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


def s3_exists(uri: str) -> bool:
    return subprocess.run(["aws", "s3", "ls", uri],
                          capture_output=True, text=True).returncode == 0


def s3_put_bytes(data: str, uri: str) -> None:
    proc = subprocess.run(["aws", "s3", "cp", "-", uri],
                          input=data, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"upload to {uri} failed: {proc.stderr.strip()}")


def s3_put_file(local: pathlib.Path, uri: str) -> None:
    proc = subprocess.run(["aws", "s3", "cp", str(local), uri, "--only-show-errors"],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"upload of {local.name} failed: {proc.stderr.strip()}")


def hf_list_files(repo: str, repo_type: str, token: str | None) -> list[str]:
    """Every file in a HF repo.

    Critical: do NOT use the `siblings` field of `/api/datasets/{repo}`. For
    MLAAD it reports 99,411 files and the repo actually holds **534,539** -- a
    5x undercount that made a 174 GB download look like a 30 GB one. The tree
    listing below is the count that matches reality.
    """
    from huggingface_hub import HfApi
    return [f for f in HfApi(token=token).list_repo_files(repo, repo_type=repo_type)]


def _hf_capped(src: Source, stage: pathlib.Path, env: dict, cap: int) -> list[dict]:
    """Take at most `cap` files from each leaf directory, deterministically.

    Why cap at all: MLAAD is 1002.9 h across 534,539 files (~174 GB) and
    docs/data/04 budgets **30 h** from it. Taking the rest is not free -- DOSS
    measured a domain-balanced 0.2k h beating a naive 6.4k h (2.77% vs 3.29%
    EER), so an uncapped pull is the failure mode the plan already warns about,
    and it costs ~32 h of wall clock out of a 20-day runway.

    Why per *directory*: MLAAD is laid out `fake/<language>/<generator>/<file>`,
    so the leaf directory is the generator -- exactly the `source x generator`
    domain key DOSS says to cap on. Capping per directory caps per domain, and
    keeps all 175 generators rather than taking all of the first few.

    Critical: the selection is seeded and the chosen list is written to
    `_meta/selection.json`. A corpus built from an unrecorded random subset is
    not reproducible, and #417333 A6 makes reproducibility the thing the
    2nd-stage submission rests on.
    """
    import collections
    import random

    from huggingface_hub import hf_hub_download

    repo = src.fetch["repo_id"]
    repo_type = src.fetch.get("repo_type", "dataset")
    seed = int(src.fetch.get("seed", 0))
    token = env.get("HF_TOKEN")

    print(f"  [list] {repo} ...")
    all_files = hf_list_files(repo, repo_type, token)
    by_dir: dict[str, list[str]] = collections.defaultdict(list)
    for f in all_files:
        by_dir[str(pathlib.PurePosixPath(f).parent)].append(f)

    chosen: list[str] = []
    for d in sorted(by_dir):
        files = sorted(by_dir[d])                      # sort first: order must
        rng = random.Random(f"{seed}:{d}")             # not depend on the API
        chosen.extend(files if len(files) <= cap else rng.sample(files, cap))

    print(f"  [cap ] {len(all_files)} files in {len(by_dir)} dirs "
          f"-> {len(chosen)} selected (cap={cap}/dir, seed={seed})")

    # Serially, 16k small files is dominated by per-request latency rather than
    # bandwidth -- snapshot_download uses 8 workers for exactly this reason, and
    # the first version of this function dropped that by looping.
    def _one(rel: str) -> dict:
        p = pathlib.Path(hf_hub_download(repo_id=repo, filename=rel,
                                         repo_type=repo_type, token=token,
                                         local_dir=str(stage)))
        return {"filename": rel, "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "url": f"https://huggingface.co/datasets/{repo}"}

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for i, rec in enumerate(pool.map(_one, chosen), 1):
            records.append(rec)
            if i % 1000 == 0 or i == len(chosen):
                print(f"  [get ] {i}/{len(chosen)}")
    (stage / "_selection.json").write_text(json.dumps(
        {"repo": repo, "seed": seed, "cap_per_dir": cap,
         "files_in_repo": len(all_files), "dirs": len(by_dir),
         "selected": chosen}, indent=2) + "\n")
    return records


def s3_sync_dir(local: pathlib.Path, uri: str) -> None:
    """Upload a directory tree, preserving relative paths.

    `sync` rather than `cp --recursive` because it skips what is already there,
    so an interrupted 99k-file upload resumes instead of restarting.
    """
    proc = subprocess.run(["aws", "s3", "sync", str(local), uri,
                           "--only-show-errors", "--exclude", ".cache/*"],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"sync to {uri} failed: {proc.stderr.strip()[:400]}")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return "?"


# --------------------------------------------------------------------------
# resolving a source to concrete download URLs
# --------------------------------------------------------------------------

def resolve_urls(src: Source, env: dict) -> list[tuple[str, str]]:
    """`[(url, filename)]` for a source, hitting an API first where needed."""
    kind = src.fetch["kind"]
    if kind == "http":
        urls = src.fetch.get("urls") or []
        if not urls:
            raise RuntimeError(f"{src.name}: no URLs in the registry yet")
        return [(u, artifact_name(u)) for u in urls]
    if kind == "mdc":
        return [_resolve_mdc(src, env)]
    if kind == "hf":
        raise RuntimeError(f"{src.name}: hf repos go through hf_snapshot(), "
                           f"not the one-URL-at-a-time path")
    if kind == "kaggle":
        raise RuntimeError(f"{src.name}: kaggle sources are fetched via the CLI, "
                           f"not this path")
    raise RuntimeError(f"{src.name}: unknown fetch kind {kind!r}")


def _resolve_mdc(src: Source, env: dict) -> tuple[str, str]:
    """POST /datasets/{id}/download for a presigned URL.

    Caveat: this call counts against MDC's 30-downloads-per-organisation-per-day
    limit, and it fails with a terms message rather than a 401 when the account
    has not accepted the dataset's terms -- there is no consent endpoint in the
    API, so that acceptance is a browser click.
    """
    import requests

    key = env.get("MDC_API_KEY")
    if not key:
        raise RuntimeError("MDC_API_KEY is not set")

    ds_id = src.fetch.get("dataset_id")
    if not ds_id:
        slug = src.fetch.get("slug")
        if not slug:
            raise RuntimeError(f"{src.name}: neither dataset_id nor slug in the registry")
        r = requests.get(f"{MDC_API}/datasets/{slug}", timeout=60)
        r.raise_for_status()
        ds_id = r.json()["id"]

    r = requests.post(f"{MDC_API}/datasets/{ds_id}/download",
                      headers={"Authorization": f"Bearer {key}"}, timeout=120)
    if r.status_code == 429:
        raise RuntimeError(f"{src.name}: MDC rate limit hit; retry after "
                           f"{r.headers.get('Retry-After', '?')}s "
                           f"(30 downloads per org per day, resets midnight UTC)")
    body = r.json()
    if "downloadUrl" not in body:
        raise RuntimeError(f"{src.name}: MDC refused -- {body.get('error', body)}")
    return body["downloadUrl"], src.fetch.get("filename") or f"{src.name}.tar.gz"


def hf_snapshot(src: Source, stage: pathlib.Path, env: dict) -> list[dict]:
    """Mirror a whole HF dataset repo, then hash it. Returns ledger records.

    Critical: HF datasets are not one archive. MLAAD is **99,411 individual
    files**, so the one-URL-at-a-time path that works for a tarball becomes
    99,411 download+upload round trips -- and an earlier version of this script
    did exactly that, with the flattened name `fake__am__Edge-TTS__x.wav`.

    Two things were wrong with it. The obvious one is throughput. The one that
    would have hurt later is that **the directory IS the metadata**: MLAAD lays
    out `fake/<language>/<generator>/<file>.wav`, and the generator is our
    generator-disjoint split axis (docs/data/08) -- the single most important
    fact about a Pool B file. Flattening it into the basename made the split
    axis a substring to be re-parsed rather than a path.

    So the repo is mirrored with its structure intact and uploaded with one
    `aws s3 sync`. snapshot_download is resumable and parallel, and sync only
    transfers what is missing, so an interrupted run resumes cheaply.
    """
    from huggingface_hub import snapshot_download

    repo = src.fetch["repo_id"]
    cap = src.fetch.get("cap_per_dir")
    if cap:
        return _hf_capped(src, stage, env, cap)

    print(f"  [sync] {repo} -> {stage} (resumable, whole repo)")
    snapshot_download(repo_id=repo, repo_type=src.fetch.get("repo_type", "dataset"),
                      local_dir=str(stage), token=env.get("HF_TOKEN"),
                      max_workers=8)

    records = []
    for f in sorted(stage.rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            records.append({"filename": str(f.relative_to(stage)),
                            "sha256": sha256_file(f), "bytes": f.stat().st_size,
                            "url": f"https://huggingface.co/datasets/{repo}"})
    return records


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def download(url: str, dest: pathlib.Path, env: dict) -> None:
    """Fetch to `dest`, resuming a partial file if one is there.

    curl rather than requests: it does Range resume, retry with backoff and a
    progress meter without us reimplementing any of it, and these are multi-GB
    artifacts over links that do drop.
    """
    headers = []
    if "huggingface.co" in url and env.get("HF_TOKEN"):
        headers = ["-H", f"Authorization: Bearer {env['HF_TOKEN']}"]
    # The progress meter is worth having on a terminal and is thousands of
    # useless lines in a log file, so it follows the tty rather than a flag.
    quiet = [] if sys.stdout.isatty() else ["--no-progress-meter"]
    # Critical: NO `--retry` here. `-C -` computes its resume offset once, when
    # curl starts. An internal retry re-requests from that original offset
    # rather than from the file's current size, so every internal retry throws
    # away everything downloaded since the process began -- observed on Common
    # Voice English going *backwards* from 74 GB to 45 GB, which meant an 88 GB
    # transfer could never converge. Retries belong in download_with_resume,
    # which starts a fresh curl and so recomputes the offset correctly.
    cmd = ["curl", "-fL", "-C", "-", *quiet, *headers, "-o", str(dest), url]
    proc = subprocess.run(cmd)
    if proc.returncode:
        raise RuntimeError(f"download failed ({proc.returncode}): {url}")


DOWNLOAD_ATTEMPTS = 12   # cheap now: each attempt resumes, none restarts


def download_with_resume(src: Source, url: str, name: str,
                         dest: pathlib.Path, env: dict) -> None:
    """Download, resuming across whole-transfer failures.

    curl's own `--retry` covers a connection that fails while it is running. It
    does not cover curl *exiting* with a partial file, and that is the failure
    we actually hit: Common Voice English is an 88 GB object behind a presigned
    Cloudflare R2 URL and it died with `curl 18` (partial transfer) three times,
    at 66%, 66% and 48%. Each attempt here resumes from what is already on disk
    via `-C -`, so a long transfer converges instead of restarting.

    Critical: an MDC presigned URL is re-minted between attempts. The signature
    is time-limited, so reusing a stale one turns a resumable transfer into a
    403 that looks like a source problem. Caveat: each re-mint spends one of the
    30 downloads-per-organisation-per-day, which is why the attempt count is
    small rather than generous.
    """
    expected = src.fetch.get("bytes") if len(src.fetch.get("urls", [1])) == 1 else None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        # Critical: a partial larger than the object cannot be resumed -- `-C -`
        # asks for a range past EOF and the server can only answer 416, forever.
        # It also means the file is *corrupt*, not merely incomplete: the only
        # way to overshoot is two writers appending at different offsets, which
        # is what happened when a killed run left an orphaned curl (PPID=1)
        # downloading into the same path as its replacement. Observed at 118.3 GB
        # against an expected 94.6 GB -- 125% -- with the excess interleaved
        # through the file, so size alone was the only visible symptom.
        if expected and dest.exists() and dest.stat().st_size > expected:
            print(f"  [bad ] {name}: {human(dest.stat().st_size)} exceeds the "
                  f"expected {human(expected)} -- corrupt, restarting from zero")
            dest.unlink()
        try:
            download(url, dest, env)
            if expected and dest.stat().st_size != expected:
                raise RuntimeError(f"size mismatch: got {dest.stat().st_size}, "
                                   f"expected {expected}")
            return
        except RuntimeError as exc:
            have = dest.stat().st_size if dest.exists() else 0
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"{exc} (after {attempt} attempts, "
                                   f"{human(have)} on disk)") from exc
            print(f"  [retr] {name}: attempt {attempt} failed ({exc}); "
                  f"{human(have)} kept, resuming")
            if src.fetch["kind"] == "mdc":
                url, _ = _resolve_mdc(src, env)


# --------------------------------------------------------------------------
# the per-source flow
# --------------------------------------------------------------------------

def acquire(src: Source, bucket: str, prefix: str, work: pathlib.Path,
            env: dict, dry_run: bool = False, keep: bool = False) -> dict:
    base = s3_uri(bucket, prefix, src)
    done_uri = f"{base}/_meta/DONE"
    if s3_exists(done_uri):
        print(f"  [skip] {src.slug} -- DONE marker present")
        return {"source": src.name, "status": "skipped"}

    is_hf = src.fetch["kind"] == "hf"
    if dry_run:
        if is_hf:
            print(f"  [plan] {src.slug}: whole HF repo {src.fetch['repo_id']} "
                  f"-> {base}/payload/ (mirror + sync)")
            return {"source": src.name, "status": "planned", "artifacts": -1}
        urls = resolve_urls(src, env)
        print(f"  [plan] {src.slug}: {len(urls)} artifact(s) -> {base}/payload/")
        for u, n in urls[:8]:
            print(f"         {n}  <- {u.split('?')[0][:96]}")
        if len(urls) > 8:
            print(f"         ... and {len(urls) - 8} more")
        return {"source": src.name, "status": "planned", "artifacts": len(urls)}

    stage = work / src.name / src.version
    stage.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        if is_hf:
            # One mirror, one sync -- see hf_snapshot for why per-file is wrong.
            records = hf_snapshot(src, stage, env)
            print(f"  [put ] {len(records)} files, "
                  f"{human(sum(r['bytes'] for r in records))} (aws s3 sync)")
            s3_sync_dir(stage, f"{base}/payload/")
        else:
            urls = resolve_urls(src, env)
            print(f"  [plan] {src.slug}: {len(urls)} artifact(s) -> {base}/payload/")
            for url, name in urls:
                local = stage / name
                print(f"  [get ] {name}")
                download_with_resume(src, url, name, local, env)
                digest = sha256_file(local)
                size = local.stat().st_size
                # A publisher-supplied checksum is the only thing that proves
                # the bytes are the bytes. Size alone does not: an interleaved
                # two-writer file can land on the right length. Mismatch is
                # fatal rather than a warning -- a silently corrupt archive in
                # the corpus is worse than a failed fetch, and worse still at
                # the 2nd-stage review where we assert this is what we trained on.
                want = src.fetch.get("sha256")
                if want and digest != want:
                    raise RuntimeError(
                        f"{name}: sha256 {digest} does not match the publisher's "
                        f"{want} -- the download is corrupt, not merely finished")
                verified = "  [verified]" if want else ""
                print(f"  [put ] {name}  {human(size)}  "
                      f"sha256={digest[:16]}...{verified}")
                s3_put_file(local, f"{base}/payload/{name}")
                records.append({"filename": name, "sha256": digest, "bytes": size,
                                "url": url.split("?")[0]})
                if not keep:
                    local.unlink()
    except Exception:
        # Critical: keep the partial. curl resumes with `-C -`, so a 60 GB
        # transfer that dies at 66% costs one retry rather than a restart --
        # and Common Voice English died at exactly that point twice (curl 18,
        # partial transfer) before this branch existed, discarding ~58 GB each
        # time because the `finally` below deleted the staging directory on the
        # way out.
        print(f"  [keep] {stage} preserved for resume")
        raise
    else:
        if not keep:
            shutil.rmtree(stage, ignore_errors=True)

    checksums = "".join(f"{r['sha256']}  {r['filename']}\n" for r in records)
    s3_put_bytes(checksums, f"{base}/_meta/checksums.sha256")
    s3_put_bytes(json.dumps(acquisition_record(src, records), indent=2) + "\n",
                 f"{base}/_meta/acquisition.json")
    s3_put_bytes(dt.datetime.now(dt.timezone.utc).isoformat() + "\n", done_uri)
    print(f"  [done] {src.slug}  {len(records)} artifact(s), "
          f"{human(sum(r['bytes'] for r in records))}")
    return {"source": src.name, "status": "acquired", "artifacts": len(records)}


def acquisition_record(src: Source, artifacts: list[dict]) -> dict:
    """The provenance row, in the shape docs/data/08 specifies.

    Written at acquisition time on purpose. The build plan notes that
    retrofitting provenance is impossible for generated audio, and doing it by
    hand for downloaded audio is merely miserable -- so it is emitted here,
    where the facts are still in scope, rather than reconstructed later.
    """
    return {
        "source_name": src.name,
        "source_version": src.version,
        "pool": src.pool,
        "source_url": src.licence_url,
        "licence": src.licence,
        "licence_url": src.licence_url,
        "redistribution_verdict": src.redistribution,
        "verified_at_origin": src.verified_at_origin,
        "verified_date": src.verified_date,
        "verified_note": src.verified_note,
        "origin_type": "dataset",
        "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "acquired_by": os.environ.get("USER", "unknown"),
        "fetch_kind": src.fetch["kind"],
        "artifacts": artifacts,
        "total_bytes": sum(a["bytes"] for a in artifacts),
        "notes": src.why,
        "caveat": src.caveat,
    }


# --------------------------------------------------------------------------
# env / cli
# --------------------------------------------------------------------------

def unbuffer_stdout() -> None:
    """Line-buffer stdout and stderr.

    Python block-buffers stdout whenever it is not a tty, so
    `fetch_to_s3.py --all > fetch.log &` leaves the log empty until the process
    exits -- which, on a multi-hour fetch, is precisely when progress stops
    being worth reporting. The only way to tell where such a run had got to was
    to stat the staging directory, and that only works from the same machine.

    Caveat: swallowing the error is deliberate. `sys.stdout` is not always a
    TextIOWrapper -- pytest's capture replaces it, and some harnesses hand over
    an object with no `reconfigure` at all. Failing to line-buffer costs a
    little log latency; raising here would abort the fetch.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass


def load_env(path: pathlib.Path) -> dict:
    """Read .env without exporting it -- keys stay in this process."""
    env = dict(os.environ)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def print_table(sources: list[Source]) -> None:
    order = {s: i for i, s in enumerate(VALID_STATUS)}
    print(f"{'SOURCE':22s} {'VER':14s} {'POOL':5s} {'STATUS':14s} {'REDIST':7s} SIZE")
    print("-" * 84)
    for s in sorted(sources, key=lambda x: (order.get(x.status, 9), x.name)):
        size = human((s.fetch or {}).get("bytes"))
        status = "ready (defer)" if (s.defer and s.status == "ready") else s.status
        print(f"{s.name:22s} {s.version:14s} {s.pool:5s} {status:14s} "
              f"{s.redistribution:7s} {size}")
    # Deferred sources are cleared but not scheduled, so counting them under
    # "fetchable now" would overstate both the count and the transfer.
    ready = [s for s in sources if s.blocked_reason() is None and not s.defer]
    deferred = [s for s in sources if s.blocked_reason() is None and s.defer]
    total = sum((s.fetch or {}).get("bytes") or 0 for s in ready)
    print("-" * 84)
    print(f"{len(ready)}/{len(sources)} scheduled now, {human(total)}")
    if deferred:
        d_total = sum((s.fetch or {}).get("bytes") or 0 for s in deferred)
        print(f"{len(deferred)} cleared but deferred, {human(d_total)} "
              f"({', '.join(s.name for s in deferred)})")


def main() -> int:
    unbuffer_stdout()
    root = pathlib.Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("names", nargs="*", help="source names; default is none")
    p.add_argument("--all", action="store_true", help="every fetchable source")
    p.add_argument("--list", action="store_true", help="show the registry and exit")
    p.add_argument("--plan", action="store_true", help="resolve URLs, upload nothing")
    p.add_argument("--bucket", default=os.environ.get("DACON_S3_BUCKET", DEFAULT_BUCKET))
    p.add_argument("--prefix", default=os.environ.get("DACON_S3_PREFIX", DEFAULT_PREFIX))
    p.add_argument("--work-dir", default="/tmp/dacon-fetch",
                   help="scratch for staging; needs room for one artifact")
    p.add_argument("--keep", action="store_true", help="do not delete staged files")
    p.add_argument("--i-have-verified", nargs="*", default=[], metavar="NAME",
                   help="override the licence gate for these named sources -- "
                        "only after an actual G2 review")
    args = p.parse_args()

    try:
        sources = load_registry()
    except Exception as exc:                       # noqa: BLE001 -- surface it plainly
        print(f"registry error: {exc}", file=sys.stderr)
        return 2

    if args.list or not (args.names or args.all):
        print_table(sources)
        if not (args.list or args.names or args.all):
            print("\nNothing selected. Use --all, or name sources, or --list.")
        return 0

    by_name = {s.name: s for s in sources}
    unknown = [n for n in args.names if n not in by_name]
    if unknown:
        print(f"unknown source(s): {unknown}", file=sys.stderr)
        return 2

    # `defer` separates two things that are not the same question: whether we
    # are *allowed* to take a source, and whether we *want* it now. ASVspoof 5
    # is cleanly licensed and 142 GB -- taking it whole is exactly what the DOSS
    # result argues against (0.2k h domain-balanced beat 6.4k h naive), so it is
    # cleared and left out of --all until somebody asks for it by name.
    wanted = ([s for s in sources if not s.defer] if args.all
              else [by_name[n] for n in args.names])
    if args.all:
        for s in sources:
            if s.defer:
                print(f"  [defer] {s.name} -- cleared but not scheduled; "
                      f"fetch by name to take it ({s.caveat.strip().splitlines()[0][:60]})")
    env = load_env(root / ".env")
    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    print(f"target  s3://{args.bucket}/{args.prefix}/")
    print(f"staging {work}\n")

    results, skipped = [], []
    for src in wanted:
        reason = src.blocked_reason()
        if reason and src.name not in args.i_have_verified:
            skipped.append((src, reason))
            continue
        if reason:
            print(f"  [warn] {src.name}: gate overridden by --i-have-verified "
                  f"({reason.splitlines()[0]})")
        try:
            results.append(acquire(src, args.bucket, args.prefix, work, env,
                                   dry_run=args.plan, keep=args.keep))
        except Exception as exc:                   # noqa: BLE001
            print(f"  [FAIL] {src.name}: {exc}", file=sys.stderr)
            results.append({"source": src.name, "status": "failed", "error": str(exc)})

    if skipped:
        print("\nNot fetched -- see docs/data/12-acquisition-status.md:")
        for src, reason in skipped:
            print(f"  {src.name:22s} {reason.strip().splitlines()[0][:96]}")

    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n{len([r for r in results if r['status'] == 'acquired'])} acquired, "
          f"{len([r for r in results if r['status'] == 'skipped'])} already present, "
          f"{len(failed)} failed, {len(skipped)} gated")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
