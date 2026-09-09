"""Pulling the raw store back down, and proving it arrived intact.

The consumer side has one job the producer side does not: deciding whether what
landed on the training machine is trustworthy. Acquisition already produced a
file that was 125% of its true size from two writers colliding, so "the transfer
finished" is not the same claim as "the bytes are right" -- and on a different
day that collision would have produced a file of exactly the right length.
Every test here is paired with the mutation that breaks it.
"""

import hashlib
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    """Import by path; `scripts/` is not a package. The module must be in
    sys.modules before it executes -- see tests/test_fetch_sources.py."""
    spec = importlib.util.spec_from_file_location(
        "fetch_from_s3", ROOT / "scripts" / "fetch_from_s3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pull = _load_module()


def _write(path: pathlib.Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def test_matching_files_verify(tmp_path):
    d = tmp_path / "payload"
    want = {"a.tar.gz": _write(d / "a.tar.gz", b"hello"),
            "b.zip": _write(d / "b.zip", b"world")}
    ok, bad = pull.verify_dir(d, want)
    assert sorted(ok) == ["a.tar.gz", "b.zip"] and bad == []


def test_a_single_flipped_byte_is_caught(tmp_path):
    """The failure mode that size cannot see.

    Mutation: compare `st_size` instead of the digest and this passes, which is
    exactly how a silently corrupt archive reaches training.
    """
    d = tmp_path / "payload"
    want = {"a.tar.gz": _write(d / "a.tar.gz", b"hello world")}
    (d / "a.tar.gz").write_bytes(b"hello_world")        # same length, one byte off
    ok, bad = pull.verify_dir(d, want)
    assert ok == [] and bad == [("a.tar.gz", "sha256 mismatch")]


def test_a_missing_file_is_reported_separately_from_a_corrupt_one(tmp_path):
    """'not there' and 'there but wrong' need different fixes -- re-sync versus
    re-acquire at source -- so they must not collapse into one message."""
    d = tmp_path / "payload"
    d.mkdir(parents=True)
    ok, bad = pull.verify_dir(d, {"gone.tar.gz": "0" * 64})
    assert bad == [("gone.tar.gz", "missing")]


def test_files_the_manifest_does_not_name_are_left_alone(tmp_path):
    """The manifest is the authority on what should be present. An unrelated
    file is not a verification failure -- flagging it would make every run of
    `aws s3 sync` against a shared directory look broken."""
    d = tmp_path / "payload"
    want = {"a.tar.gz": _write(d / "a.tar.gz", b"x")}
    _write(d / "notes.txt", b"unrelated")
    ok, bad = pull.verify_dir(d, want)
    assert ok == ["a.tar.gz"] and bad == []


# --------------------------------------------------------------------------
# the checksum manifest
# --------------------------------------------------------------------------

def test_checksums_parse_in_sha256sum_format():
    text = ("aa" * 32 + "  first.tar.gz\n") + ("bb" * 32 + "  second.zip\n")
    got = pull.parse_checksums(text)
    assert got == {"first.tar.gz": "aa" * 32, "second.zip": "bb" * 32}


def test_a_filename_with_spaces_survives_parsing():
    """Mutation: `line.split()` and `my file.tar.gz` becomes `my`, so the file
    reads as missing and a good corpus is rejected."""
    text = "cc" * 32 + "  my file name.tar.gz\n"
    assert pull.parse_checksums(text) == {"my file name.tar.gz": "cc" * 32}


def test_blank_lines_are_ignored():
    assert pull.parse_checksums("\n\n" + "dd" * 32 + "  a\n\n") == {"a": "dd" * 32}


# --------------------------------------------------------------------------
# discovering the store
# --------------------------------------------------------------------------

def _fake_ls(monkeypatch, rows):
    monkeypatch.setattr(pull, "s3_ls", lambda uri: rows)


def test_discover_groups_by_source_and_version(monkeypatch):
    base = "p/raw/"
    _fake_ls(monkeypatch, [
        (base + "musan/openslr-17/payload/musan.tar.gz", 100),
        (base + "musan/openslr-17/_meta/DONE", 33),
        (base + "musan/openslr-17/_meta/checksums.sha256", 82),
    ])
    got = pull.discover("b", "p/raw")
    assert got["musan"]["version"] == "openslr-17"
    assert got["musan"]["complete"] is True
    assert got["musan"]["files"] == ["musan.tar.gz"]
    assert got["musan"]["bytes"] == 100, "_meta must not count toward payload size"


def test_a_source_without_a_done_marker_is_incomplete(monkeypatch):
    """DONE is written last, after payload and checksums. Its absence means the
    upload was interrupted and the payload may be short.

    Mutation: default `complete` to True and an interrupted upload is handed to
    training as a finished corpus. This is not hypothetical -- compspoof-v2 sits
    in the real store right now with 3 files and no DONE.
    """
    base = "p/raw/"
    _fake_ls(monkeypatch, [(base + "half/v1/payload/a.tar.gz", 10)])
    assert pull.discover("b", "p/raw")["half"]["complete"] is False


def test_incomplete_sources_are_skipped_by_default(monkeypatch, tmp_path, capsys):
    """Mutation: drop the `complete` check in main() and a truncated archive is
    downloaded and trained on without a word."""
    _fake_ls(monkeypatch, [("p/raw/half/v1/payload/a.tar.gz", 10)])
    monkeypatch.setattr(pull, "read_meta", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv",
                        ["x", "half", "--dest", str(tmp_path), "--prefix", "p/raw"])
    assert pull.main() == 0
    out = capsys.readouterr().out
    assert "[skip] half" in out and "no DONE marker" in out


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def test_extraction_refuses_members_that_escape_the_root(tmp_path):
    """These archives are third-party. A `../` member would write outside the
    extraction root. Mutation: call extractall() unfiltered and it does."""
    import tarfile

    payload = tmp_path / "payload"
    payload.mkdir()
    inside = tmp_path / "seed.txt"
    inside.write_bytes(b"data")
    with tarfile.open(payload / "evil.tar", "w") as tf:
        tf.add(inside, arcname="good.txt")
        tf.add(inside, arcname="../escaped.txt")

    out = tmp_path / "out"
    pull.extract_archives(payload, out)
    assert (out / "good.txt").exists()
    assert not (tmp_path / "escaped.txt").exists(), "member escaped the root"


def test_inside_accepts_normal_paths_and_rejects_traversal(tmp_path):
    assert pull._inside(tmp_path, tmp_path / "a" / "b.txt")
    assert not pull._inside(tmp_path, tmp_path / ".." / "b.txt")
