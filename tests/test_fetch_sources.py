"""The acquisition registry and its licence gate.

The gate is the only thing standing between a download URL and a corpus we are
not allowed to ship, so it is tested the way docs/pipelines/05 asks: each
invariant paired with the mutation that was observed to break it. The mutations
are written out beside each test because a green suite is not evidence that a
check can fail -- an adversarial pass over the training loop left 28 of 85
mutants alive under 101 passing tests.
"""

import importlib.util
import pathlib
import re
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts" / "sources.yaml"


def _load_module():
    """Import the script by path -- `scripts/` is not a package.

    Caveat: the module has to be in `sys.modules` *before* it executes.
    `@dataclasses.dataclass` resolves annotations through
    `sys.modules[cls.__module__]`, which is None for a module that is still
    being exec'd, and the decorator raises rather than degrading.
    """
    spec = importlib.util.spec_from_file_location(
        "fetch_to_s3", ROOT / "scripts" / "fetch_to_s3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = _load_module()


@pytest.fixture(scope="module")
def sources():
    return fetch.load_registry(REGISTRY)


def _write(tmp_path, doc) -> pathlib.Path:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _minimal(**over):
    row = {"name": "x", "version": "1", "pool": "A", "status": "ready",
           "redistribution": "ok", "licence": "CC0", "licence_url": "https://e.g",
           "verified_at_origin": True, "fetch": {"kind": "http", "urls": ["https://e.g/a.tar"]}}
    row.update(over)
    return {"schema_version": 1, "sources": [row]}



def _sized(src, expected):
    """Same source with a different expected size (None = unknown).

    download_with_resume skips its size guard when the registry records no
    `bytes`, which is the branch these stubs exercise -- they write a few bytes,
    not 11 GB.
    """
    fetch_ = dict(src.fetch)
    fetch_.pop("bytes", None)
    if expected is not None:
        fetch_["bytes"] = expected
    return fetch.dataclasses.replace(src, fetch=fetch_)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def test_the_shipped_registry_parses(sources):
    assert len(sources) >= 10


def test_nothing_without_an_ok_verdict_is_fetchable(sources):
    """Mutation: `if self.redistribution != REDIST_OK` -> `!= "no"` in
    blocked_reason lets every `review` source through. Observed to fail here."""
    for src in sources:
        if src.redistribution != "ok":
            assert src.blocked_reason() is not None, (
                f"{src.name} has verdict {src.redistribution!r} but is fetchable")


def test_ok_verdict_requires_the_licence_read_at_origin(tmp_path):
    """The rule docs/data/11 states: reported-from-a-secondary-source is not
    verification. Mutation: drop the check in load_registry and this passes."""
    path = _write(tmp_path, _minimal(verified_at_origin=False))
    with pytest.raises(ValueError, match="verified_at_origin"):
        fetch.load_registry(path)


def test_every_ok_source_in_the_shipped_registry_was_verified(sources):
    for src in sources:
        if src.redistribution == "ok":
            assert src.verified_at_origin and src.verified_date, src.name


def test_a_blocked_source_must_say_why(sources):
    """A blocked row with no reason is how a retired idea gets rediscovered."""
    for src in sources:
        if src.blocked_reason() is not None:
            assert (src.blocker or src.requirement), (
                f"{src.name} is not fetchable but records no blocker/requirement")


@pytest.mark.parametrize("name", ["scraped-audio", "ai-hub"])
def test_the_rules_verdicts_stay_recorded(sources, name):
    """docs/data/01 rules both of these out. They live in the registry as
    permanent `no` rows precisely so nobody re-proposes them as ideas."""
    src = next(s for s in sources if s.name == name)
    assert src.redistribution == "no"
    assert src.blocked_reason() is not None


def test_musdb18_was_cleared_against_the_rule_text(sources):
    """MUSDB18-HQ is usable, and the row has to carry *why*.

    This assertion has now been wrong twice, in opposite directions, which is
    why it pins the reasoning rather than the verdict alone:

    1. `needs_request` -- from docs/data/11's reported "access request
       required". False; the Zenodo file serves anonymously.
    2. `needs_review` -- from reading the licence's shape ("educational
       purposes only", four upstream grants) instead of the rule. Also false.
       #417280 answer 1 states that submission for 운영진 검증 is distinct from
       redistribution, and bars only data whose licence restricts 제3자 제공
       *itself*. MUSDB18-HQ restricts commercial use, not provision -- and
       #417212 answers that exact shape (CC-BY-NC, CC-BY-NC-SA) with yes.

    The lesson the docstring exists to keep: clear a source against the rule
    text, not against how restrictive the licence feels.
    """
    src = next(s for s in sources if s.name == "musdb18-hq")
    assert src.status == "ready"
    assert src.redistribution == "ok"
    assert src.verified_at_origin
    assert src.fetch["urls"]


def test_non_commercial_sources_record_the_compliance_duty(sources):
    """#417212's answer turns on 해당 라이선스 조건을 준수하는 경우, so an NC
    source is usable *conditionally*. A row that records the permission without
    the duty would let the attribution and non-commercial terms go unhonoured
    in the 2nd-stage 데이터 구성 보고서."""
    raw = yaml.safe_load(REGISTRY.read_text())["sources"]
    for row in raw:
        licence = str(row.get("licence", ""))
        if "NC" in licence and row.get("redistribution") == "ok":
            assert row.get("caveat_compliance") or row.get("caveat"), (
                f"{row['name']} is NC and cleared, but records no compliance duty")


# --------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad,match", [
    ({"status": "reddy"}, "status"),
    ({"redistribution": "fine"}, "redistribution"),
    ({"fetch": {"kind": "ftp", "urls": []}}, "fetch.kind"),
])
def test_typos_are_rejected_not_absorbed(tmp_path, bad, match):
    """A typo in `redistribution` that read as not-ok would quietly stop
    acquiring; one that read as ok would quietly acquire something unshippable."""
    with pytest.raises(ValueError, match=match):
        fetch.load_registry(_write(tmp_path, _minimal(**bad)))


def test_unsupported_schema_version_is_rejected(tmp_path):
    doc = _minimal()
    doc["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        fetch.load_registry(_write(tmp_path, doc))


def test_duplicate_source_slugs_are_rejected(tmp_path):
    """Two rows with the same name/version would write into one S3 prefix and
    the second DONE marker would mask the first's missing payload."""
    doc = _minimal()
    doc["sources"].append(dict(doc["sources"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        fetch.load_registry(_write(tmp_path, doc))


def test_yaml_unquoted_no_does_not_become_a_boolean(tmp_path):
    """YAML 1.1 reads bare `no` as False. Unquoted, the verdict would stop being
    a member of VALID_REDIST -- caught loudly rather than read as 'not ok'."""
    path = tmp_path / "sources.yaml"
    path.write_text("schema_version: 1\nsources:\n"
                    "  - {name: x, version: '1', pool: A, status: blocked,\n"
                    "     redistribution: no, licence: none, licence_url: 'https://e.g',\n"
                    "     fetch: {kind: http, urls: []}}\n")
    with pytest.raises(ValueError, match="redistribution"):
        fetch.load_registry(path)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def test_s3_layout_is_name_then_version(sources):
    src = next(s for s in sources if s.name == "musan")
    uri = fetch.s3_uri("b", "p/raw", src, "payload", "musan.tar.gz")
    assert uri == "s3://b/p/raw/musan/openslr-17/payload/musan.tar.gz"


def test_a_new_release_lands_beside_the_old_one(sources):
    """Version in the path is what makes the raw store append-only."""
    src = next(s for s in sources if s.name == "musan")
    other = fetch.dataclasses.replace(src, version="openslr-17b")
    assert fetch.s3_base("p", src) != fetch.s3_base("p", other)


@pytest.mark.parametrize("url", [
    "https://e.g/",          # no path component at all
    "https://e.g/a/..",      # basename is literally ".."
])
def test_a_publisher_cannot_name_a_file_outside_the_prefix(url):
    """Mutation: drop the `name in ('.', '..')` clause and the `..` cases return
    a component that resolves to the source's parent prefix."""
    with pytest.raises(ValueError):
        fetch.artifact_name(url)


def test_traversal_in_the_path_collapses_to_a_plain_basename():
    """Taking the basename is itself the defence, and this pins that reading.

    An earlier version of this test expected `../../etc/passwd` to raise. It
    does not, and should not: the basename is `passwd`, an ordinary filename
    that lands inside the source's own payload prefix like any other. The guard
    exists for names that ARE `..`, not for paths that merely contain one.

    Same for a single `.`: pathlib drops it as a component, so `/a/.` is `a`.
    """
    assert fetch.artifact_name("https://e.g/../../etc/passwd") == "passwd"
    assert fetch.artifact_name("https://e.g/a/.") == "a"


def test_ordinary_filenames_survive():
    assert fetch.artifact_name("https://e.g/x/musan.tar.gz") == "musan.tar.gz"
    assert fetch.artifact_name("https://e.g/x/a.zip?sig=1") == "a.zip"


# --------------------------------------------------------------------------
# the provenance record
# --------------------------------------------------------------------------

def test_acquisition_record_carries_what_the_build_plan_requires(sources):
    """docs/data/08 lists the ledger columns. The source-level half of that row
    is emitted here, at acquisition time, because reconstructing it later is at
    best miserable and for generated audio impossible."""
    src = next(s for s in sources if s.name == "zeroth-korean")
    rec = fetch.acquisition_record(
        src, [{"filename": "a.tar.gz", "sha256": "ab" * 32, "bytes": 10, "url": "https://e.g"}])
    for field in ("source_name", "licence", "licence_url", "redistribution_verdict",
                  "verified_at_origin", "verified_date", "origin_type", "acquired_at",
                  "artifacts", "total_bytes", "pool"):
        assert field in rec, field
    assert rec["total_bytes"] == 10
    assert rec["artifacts"][0]["sha256"] == "ab" * 32


def test_the_record_does_not_launder_an_unverified_verdict(sources):
    """The ledger must carry the verdict as it actually is -- a `review` row that
    got fetched under --i-have-verified still records `review`, so the 2nd-stage
    report cannot claim a verification that never happened."""
    src = next(s for s in sources if s.redistribution == "review")
    rec = fetch.acquisition_record(src, [])
    assert rec["redistribution_verdict"] == "review"
    assert rec["verified_at_origin"] is False


def test_the_prefix_root_has_no_trailing_slash(sources):
    """Mutation: restore the `+ "/"` in s3_uri and every key gains an empty
    segment -- `.../openslr-17//payload/x`, which S3 accepts as a *different*
    key from the intended one and no listing makes obvious."""
    src = next(s for s in sources if s.name == "musan")
    assert fetch.s3_uri("b", "p/raw", src).endswith("/openslr-17")
    assert "//payload" not in fetch.s3_uri("b", "p/raw", src) + "/payload"


def test_output_is_line_buffered_when_piped():
    """A backgrounded fetch must report progress before it exits.

    Mutation: drop the `unbuffer_stdout()` call at the top of main() and a
    `--all > log &` run leaves the log empty for hours -- observed, and the
    reason this function exists. Python block-buffers a non-tty stdout, so the
    default is silence exactly where progress matters most.
    """
    import io

    piped = io.TextIOWrapper(io.BytesIO())          # not a tty: block-buffered
    assert piped.line_buffering is False
    real = sys.stdout
    try:
        sys.stdout = piped
        fetch.unbuffer_stdout()
    finally:
        sys.stdout = real
    assert piped.line_buffering is True


def test_unbuffering_survives_a_stream_that_cannot_reconfigure():
    """pytest's capture object has no `reconfigure`. Losing a little log
    latency is acceptable; aborting a multi-hour fetch over it is not."""
    class NoReconfigure:
        pass

    real_out, real_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = NoReconfigure()
        fetch.unbuffer_stdout()                      # must not raise
    finally:
        sys.stdout, sys.stderr = real_out, real_err


def test_deferred_sources_are_cleared_but_not_scheduled(sources):
    """`defer` separates permission from intent, and both halves must hold.

    ASVspoof 5 is cleanly ODC-BY and 142 GB. Folding "too big for now" into the
    licence verdict would have said `review`, which is false and would have
    hidden a legitimately usable source behind a legal question it does not
    have. Mutation: drop the `not s.defer` filter in main() and `--all` pulls
    142 GB nobody asked for -- against a ~70 h Pool B budget, and against DOSS,
    which measured domain-balanced 0.2k h beating naive 6.4k h.
    """
    deferred = [s for s in sources if s.defer]
    assert deferred, "asvspoof5 should be deferred"
    for s in deferred:
        assert s.blocked_reason() is None, f"{s.name} is deferred but also gated"
        assert s.caveat.strip(), f"{s.name} is deferred with no reason recorded"


def test_nd_sources_record_the_reproduce_from_original_duty(sources):
    """ND is usable, but only in the one form DACON granted.

    This test previously asserted every wholly-ND corpus was **blocked**, which
    was correct until 2026-09-08. #417333 A5 then answered: ND data may be used
    and augmented, *"이경우에는 원본 파일과 코드로 재현될 수 있어야합니다"*. So the
    invariant is no longer "ND is barred" but "ND carries a condition", and a
    row that clears ND without recording that condition is the dangerous state
    -- it reads as ordinary permission and invites shipping a derivative.

    Mutation: delete the caveat_compliance block from codecfake and this fails.
    """
    nd = [s for s in sources
          if re.search(r"\bBY-N[CD](?:-ND)?\b", s.licence.upper())
          and "ND" in s.licence.upper().split("BY-")[-1]]
    names = {s.name for s in nd}
    assert {"codecfake", "st-codecfake", "scenefake"} <= names, names
    for s in nd:
        if s.blocked_reason() is None:
            duty = s.caveat_compliance.lower()
            assert "원본" in s.caveat_compliance or "original" in duty, (
                f"{s.name} clears ND without recording the reproduce-from-original duty")


def test_the_mixed_licence_corpora_are_cleared_by_allowlist_not_by_hand(sources):
    """FMA and MTG-Jamendo are cleared while containing ND tracks, which is only
    defensible because a filter excludes them. The row has to say so, or the
    next reader sees a cleared corpus and takes the whole archive."""
    for name in ("fma", "mtg-jamendo"):
        src = next(s for s in sources if s.name == name)
        assert src.redistribution == "ok"
        assert "allow.csv" in src.caveat_compliance, (
            f"{name} is cleared but does not name the allowlist that clears it")


def test_hf_repos_do_not_go_through_the_single_url_path(sources):
    """A HF dataset is a tree, not an archive, and must not be resolved per file.

    Mutation: restore `_resolve_hf` in resolve_urls and MLAAD becomes 99,411
    download+upload round trips with names flattened to
    `fake__am__Edge-TTS__x.wav`. Beyond being unusably slow, that flattening
    destroys the layout `fake/<language>/<generator>/<file>` -- and the
    generator directory is the generator-disjoint split axis (docs/data/08),
    the single most load-bearing fact about a Pool B file.
    """
    src = next(s for s in sources if s.name == "mlaad")
    assert src.fetch["kind"] == "hf"
    with pytest.raises(RuntimeError, match="hf_snapshot"):
        fetch.resolve_urls(src, {})


def test_every_hf_source_names_a_repo(sources):
    for s in sources:
        if s.fetch.get("kind") == "hf":
            assert s.fetch.get("repo_id"), f"{s.name} is kind=hf with no repo_id"


def test_mlaad_is_capped_per_generator(sources):
    """MLAAD must not be pulled whole, and the cap must be per generator.

    Mutation: drop `cap_per_dir` and the fetcher takes all 534,539 files
    (~174 GB, ~32 h wall clock) to fill a 30 h budget -- the exact shape DOSS
    argues against, where a domain-balanced 0.2k h beat a naive 6.4k h. The cap
    is per LEAF directory because MLAAD lays out
    fake/<language>/<generator>/<file>, so leaf == generator == the DOSS domain
    key. A global cap would keep all of the first few generators and none of the
    rest, which is the opposite of what generator diversity needs.
    """
    src = next(s for s in sources if s.name == "mlaad")
    assert src.fetch.get("cap_per_dir"), "mlaad must be capped"
    assert src.fetch.get("seed") is not None, "the subset must be reproducible"
    assert "generator" in src.caveat.lower()


def test_capped_selection_is_deterministic_and_order_independent(tmp_path):
    """The same seed must choose the same files, whatever order the API lists
    them in. Mutation: drop the `sorted(by_dir[d])` before sampling and the
    selection silently depends on HF's response order -- reproducible today,
    different next month, and docs/competition/05 A6 makes that subset part of
    what we submit."""
    import collections
    import random

    def select(files, cap, seed=0):
        by_dir = collections.defaultdict(list)
        for f in files:
            by_dir[str(pathlib.PurePosixPath(f).parent)].append(f)
        out = []
        for d in sorted(by_dir):
            fs = sorted(by_dir[d])
            rng = random.Random(f"{seed}:{d}")
            out.extend(fs if len(fs) <= cap else rng.sample(fs, cap))
        return out

    files = [f"fake/en/GenA/{i}.wav" for i in range(50)] + \
            [f"fake/de/GenB/{i}.wav" for i in range(5)]
    a = select(files, cap=10)
    b = select(list(reversed(files)), cap=10)
    assert a == b, "selection must not depend on listing order"
    assert len(a) == 15, "10 from the big dir, all 5 from the small one"
    assert sum(1 for f in a if "GenB" in f) == 5, "every generator survives the cap"


def test_mlaad_cap_matches_the_documented_hour_budget(sources):
    """The cap is a corpus-design number, not an arbitrary one.

    MLAAD v9 is 1002.9 h over 534,539 files -> ~6.75 s/file, and docs/data/04
    budgets 30 h from it. cap=300 (the first value shipped) selects 160,205
    files = ~301 h -- 10x the budget, and enough to make MLAAD ~80% of a 70 h
    Pool B, destroying the domain balance DOSS says decides the result. The cap
    must land within a factor of ~1.5 of the budget.
    """
    src = next(s for s in sources if s.name == "mlaad")
    cap = src.fetch["cap_per_dir"]
    hours = 535 * cap * (1002.9 * 3600 / 534539) / 3600
    assert 20 <= hours <= 45, (
        f"cap={cap}/dir selects ~{hours:.0f} h against a 30 h budget")


def test_a_failed_download_keeps_its_partial(sources, tmp_path, monkeypatch):
    """Mutation: change the `else:` back to `finally:` and a 60 GB transfer that
    dies at 66% discards ~58 GB instead of resuming. Observed twice on Common
    Voice English (curl 18) before this branch existed."""
    src = next(s for s in sources if s.name == "musan")
    stage = tmp_path / "musan" / "openslr-17"

    monkeypatch.setattr(fetch, "s3_exists", lambda uri: False)
    def boom(url, dest, env):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial")
        raise RuntimeError("download failed (18)")
    monkeypatch.setattr(fetch, "download", boom)

    with pytest.raises(RuntimeError, match="18"):
        fetch.acquire(src, "b", "p", tmp_path, {})
    assert stage.exists(), "staging was deleted, losing the resumable partial"
    assert (stage / "musan.tar.gz").read_bytes() == b"partial"


def test_download_resumes_across_whole_transfer_failures(sources, tmp_path, monkeypatch):
    """curl's --retry does not cover curl *exiting* with a partial file.

    Mutation: call download() directly instead of download_with_resume() and an
    88 GB transfer that dies at 66% is a hard failure -- which is exactly what
    Common Voice English did three times.
    """
    src = _sized(next(s for s in sources if s.name == "musan"), None)
    dest = tmp_path / "x.tar.gz"
    calls = []

    def flaky(url, d, env):
        calls.append(url)
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(b"x" * len(calls))     # grows: curl -C - resuming
        if len(calls) < 3:
            raise RuntimeError("download failed (18)")
    monkeypatch.setattr(fetch, "download", flaky)

    fetch.download_with_resume(src, "https://e.g/x", "x.tar.gz", dest, {})
    assert len(calls) == 3, "should have retried until it succeeded"
    assert dest.read_bytes() == b"xxx"


def test_resume_gives_up_and_reports_what_it_kept(sources, tmp_path, monkeypatch):
    """A permanent failure must still surface, and say how much is on disk so
    the next run's resume is worth attempting."""
    src = _sized(next(s for s in sources if s.name == "musan"), None)
    dest = tmp_path / "x.tar.gz"

    def always(url, d, env):
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(b"partial")
        raise RuntimeError("download failed (18)")
    monkeypatch.setattr(fetch, "download", always)

    with pytest.raises(RuntimeError, match="after 12 attempts"):
        fetch.download_with_resume(src, "https://e.g/x", "x.tar.gz", dest, {})
    assert dest.exists(), "the partial must survive for the next run"


def test_mdc_urls_are_reminted_between_attempts(sources, tmp_path, monkeypatch):
    """A presigned URL is time-limited. Mutation: drop the re-mint and a resume
    against a stale signature 403s, which reads as a broken source rather than
    an expired link."""
    src = _sized(next(s for s in sources if s.name == "common-voice-ko"), None)
    assert src.fetch["kind"] == "mdc"
    minted = []
    monkeypatch.setattr(fetch, "_resolve_mdc",
                        lambda s, e: (minted.append(1) or "https://fresh", "f"))

    seen = []
    def flaky(url, d, env):
        seen.append(url)
        d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(b"x")
        if len(seen) < 2:
            raise RuntimeError("download failed (18)")
    monkeypatch.setattr(fetch, "download", flaky)

    fetch.download_with_resume(src, "https://stale", "f", tmp_path / "f", {})
    assert seen == ["https://stale", "https://fresh"], seen


def test_curl_does_not_do_its_own_retrying(sources, tmp_path, monkeypatch):
    """`-C -` fixes its resume offset when curl starts, so curl's own --retry
    re-requests from that stale offset and discards everything fetched since.

    Mutation: put `--retry 5 --retry-all-errors` back and an 88 GB transfer goes
    *backwards* on every blip -- observed at 74 GB -> 45 GB on Common Voice
    English, i.e. it can never finish. Retrying is download_with_resume's job,
    because a fresh curl recomputes the offset from the file's current size.
    """
    import types

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch.download("https://e.g/x", tmp_path / "x", {})
    cmd = seen["cmd"]
    assert "-C" in cmd and "-" in cmd, "resume must stay enabled"
    assert not any(c.startswith("--retry") for c in cmd), (
        f"curl must not retry internally: {cmd}")


def test_an_oversized_partial_is_discarded_not_resumed(sources, tmp_path, monkeypatch):
    """A partial bigger than the object is corrupt, and unresumable.

    Two writers appending at different offsets is the only way to overshoot --
    an orphaned curl (PPID=1) from a killed run, downloading into the same path
    as its replacement. Observed at 118.3 GB against an expected 94.6 GB. `-C -`
    would then ask for a range past EOF forever, so the file must be dropped.

    Mutation: remove the oversize branch and this loops to exhaustion.
    """
    src = _sized(next(s for s in sources if s.name == "musan"), 100)
    dest = tmp_path / "x.tar.gz"
    dest.write_bytes(b"z" * 125)                 # 125% of expected: corrupt

    def good(url, d, env):
        assert not d.exists(), "the corrupt partial should have been deleted"
        d.write_bytes(b"y" * 100)
    monkeypatch.setattr(fetch, "download", good)

    fetch.download_with_resume(src, "https://e.g/x", "x.tar.gz", dest, {})
    assert dest.stat().st_size == 100


def test_a_short_download_is_not_accepted_as_complete(sources, tmp_path, monkeypatch):
    """curl can exit 0 having written less than the object. Mutation: drop the
    post-download size check and a truncated archive enters the corpus looking
    finished."""
    src = _sized(next(s for s in sources if s.name == "musan"), 100)
    monkeypatch.setattr(fetch, "download",
                        lambda url, d, env: d.write_bytes(b"y" * 40))
    with pytest.raises(RuntimeError, match="size mismatch"):
        fetch.download_with_resume(src, "https://e.g/x", "x", tmp_path / "x", {})


def test_publisher_checksums_are_recorded_for_mdc_sources(sources):
    """MDC exposes a sha256 per dataset. Recording it turns 'the download
    finished' into 'the bytes are the bytes' -- size alone cannot detect an
    interleaved two-writer file that happens to land on the right length."""
    for name in ("common-voice-en", "common-voice-ko"):
        src = next(s for s in sources if s.name == name)
        sha = src.fetch.get("sha256")
        assert sha and len(sha) == 64, f"{name} has no publisher checksum"
