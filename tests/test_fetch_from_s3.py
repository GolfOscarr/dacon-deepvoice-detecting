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
import shutil
import subprocess
import pathlib
import zipfile
import sys

import numpy as np
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


def test_extraction_is_recursive_over_the_payload(tmp_path, capsys):
    """🔴 SONICS nests its ten zips under `payload/fake_songs/`. A
    non-recursive `iterdir()` yielded the *directory*, `is_file()` was False,
    and 30 GiB extracted to nothing while the run reported success."""
    import zipfile

    payload = tmp_path / "payload" / "fake_songs"
    payload.mkdir(parents=True)
    inner = tmp_path / "src"
    inner.mkdir()
    (inner / "a.wav").write_bytes(b"RIFF....")
    with zipfile.ZipFile(payload / "part_01.zip", "w") as zf:
        zf.write(inner / "a.wav", "songs/a.wav")

    out = tmp_path / "out"
    assert pull.extract_archives(tmp_path / "payload", out) == 1
    assert (out / "songs" / "a.wav").exists()


def test_an_unpacked_archive_looking_file_is_reported(tmp_path, capsys):
    """CompSpoof ships `development.tar.gz.part_aa..ae`; neither `is_tarfile`
    nor `is_zipfile` recognises a part. Returning a smaller count silently is
    the "validates and does nothing" failure -- the caller sees a successful
    extraction of a corpus that is not there."""
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "development.tar.gz.part_aa").write_bytes(b"\x1f\x8b not really")
    (payload / "README.md").write_text("not an archive, and not a warning")

    assert pull.extract_archives(payload, tmp_path / "out") == 0
    warned = capsys.readouterr().out
    assert "not unpacked" in warned and "part_aa" in warned
    assert "README" not in warned, "warning about docs trains the reader to ignore it"


# --------------------------------------------------------------------------
# partial fetch (`--only`)
# --------------------------------------------------------------------------

def test_only_selects_by_glob_and_no_glob_takes_everything():
    names = ["eval.tar.gz", "test_source.tar.gz", "eval_source.tar.gz",
             "development.tar.gz.part_aa"]
    assert pull.selected(names, ["*_source.tar.gz"]) == ["test_source.tar.gz",
                                                         "eval_source.tar.gz"]
    assert pull.selected(names, []) == names


def test_the_sync_excludes_once_and_then_includes(monkeypatch):
    """🔴 aws applies filters in the order given. A second `--exclude *` after
    an include cancels it, and the sync transfers nothing while reporting
    success. Mutation: emit the exclude inside the loop and this catches it."""
    seen = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(pull.subprocess, "run",
                        lambda cmd, **k: (seen.update(cmd=cmd), _Done())[1])
    pull.sync_down("b", "p", "n", "v1", pathlib.Path("/tmp/x"), False,
                   ["a.tar.gz", "b.tar.gz"])
    cmd = seen["cmd"]
    assert cmd.count("--exclude") == 1
    assert cmd.index("--exclude") < cmd.index("--include")
    assert "payload/a.tar.gz" in cmd and "payload/b.tar.gz" in cmd


def test_a_partial_fetch_verifies_only_what_it_asked_for(tmp_path):
    """🔴 The trap this pairs with: handing the *full* manifest to a partial
    fetch reports every file it deliberately did not take as `missing`, which
    is a red run for a correct one. CompSpoof is 18 payload files and pool E
    needs two of them."""
    d = tmp_path / "payload"
    want = {"eval_source.tar.gz": _write(d / "eval_source.tar.gz", b"x"),
            "development.tar.gz.part_aa": "0" * 64}
    only = ["*_source.tar.gz"]
    trimmed = {k: v for k, v in want.items() if k in set(pull.selected(want, only))}
    ok, bad = pull.verify_dir(d, trimmed)
    assert ok == ["eval_source.tar.gz"] and bad == []

    # and without the trim it is a false failure
    ok, bad = pull.verify_dir(d, want)
    assert bad == [("development.tar.gz.part_aa", "missing")]


def test_split_pieces_named_by_letters_are_reported_too():
    """🔴 `split -b` names its pieces `.aa`, `.ab`, ... -- no digits, no
    extension of their own -- so the digit-based checks miss them completely.
    PartialSpoof ships `database_eval.tar.gz.aa/.ab/.ac`, 5.4 GB and the whole
    eval set, and it was skipped with nothing said. Same defect as SONICS,
    different letters.

    Mutation: drop the two-letter branch and the first three assertions fail."""
    for name in ("database_eval.tar.gz.aa", "database_eval.tar.gz.ab",
                 "CFAD.z01", "train_split.zip", "database_dev.tar.gz"):
        assert pull._looks_like_archive(pathlib.Path(name)), name
    # and things that merely end in two letters are not archives
    for name in ("notes.md", "readme.txt", "model.pt", "subtitles.srt.en"):
        assert not pull._looks_like_archive(pathlib.Path(name)), name


# --------------------------------------------------------------------------
# split archives
# --------------------------------------------------------------------------

def _tar_of(path, names):
    import io, tarfile
    with tarfile.open(path, "w:gz") as tf:
        for n in names:
            data = (n * 100).encode()
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_a_split_tar_is_joined_and_extracted(tmp_path):
    """🔴 Four sources in the store ship this way -- PartialSpoof, CompSpoof,
    CFAD, Codecfake -- and none of them could be extracted at all before. The
    pieces are not archives individually, so every one was skipped."""
    payload = tmp_path / "payload"
    payload.mkdir()
    whole = _tar_of(tmp_path / "whole.tar.gz", ["a.wav", "b.wav"])
    blob = whole.read_bytes()
    half = len(blob) // 2
    (payload / "whole.tar.gz.aa").write_bytes(blob[:half])
    (payload / "whole.tar.gz.ab").write_bytes(blob[half:])
    whole.unlink()

    out = tmp_path / "out"
    assert pull.extract_archives(payload, out) == 1
    assert {q.name for q in out.iterdir()} == {"a.wav", "b.wav"}
    # the joined copy is not left behind to double the corpus on disk
    assert not (tmp_path / "_joining").exists()


def test_a_split_set_with_a_hole_is_not_joined(tmp_path):
    """🔴 Joining `.aa` and `.ac` produces a corrupt archive that extracts
    partially and reports success -- worse than not extracting. The group must
    be contiguous from the first piece or it is left to the warning.

    Mutation: drop the contiguity check and this extracts a truncated tar."""
    payload = tmp_path / "payload"
    payload.mkdir()
    whole = _tar_of(tmp_path / "whole.tar.gz", ["a.wav", "b.wav", "c.wav"])
    blob = whole.read_bytes()
    third = len(blob) // 3
    (payload / "whole.tar.gz.aa").write_bytes(blob[:third])
    (payload / "whole.tar.gz.ac").write_bytes(blob[2 * third:])   # .ab missing
    whole.unlink()

    assert pull.split_groups(sorted(payload.iterdir())) == {}
    out = tmp_path / "out"
    assert pull.extract_archives(payload, out) == 0


def test_a_spanned_zip_is_recognised_as_one_group(tmp_path):
    """CFAD ships `CFAD.z01..z03` plus `CFAD.zip`, where the `.zip` is the LAST
    piece and carries the central directory -- so concatenation is wrong and
    `zip -s 0` is the only correct join. Python's zipfile has no multi-disk
    support at all."""
    payload = tmp_path / "payload"
    payload.mkdir()
    for name in ("CFAD.z01", "CFAD.z02", "CFAD.z03", "CFAD.zip"):
        (payload / name).write_bytes(b"x" * 10)
    groups = pull.split_groups(sorted(payload.iterdir()))
    assert list(groups) == [payload / "CFAD.zip"]
    pieces = [q.name for q in groups[payload / "CFAD.zip"]]
    assert pieces == ["CFAD.z01", "CFAD.z02", "CFAD.z03", "CFAD.zip"]


def test_a_spanned_zip_missing_its_last_part_is_not_joined(tmp_path):
    """Without the `.zip` there is no central directory, so there is nothing to
    join and the pieces are just bytes."""
    payload = tmp_path / "payload"
    payload.mkdir()
    for name in ("CFAD.z01", "CFAD.z02"):
        (payload / name).write_bytes(b"x" * 10)
    assert pull.split_groups(sorted(payload.iterdir())) == {}


def test_pieces_are_not_mistaken_for_archives_of_their_own(tmp_path):
    """A `.aa` piece must be consumed by its group, not reported as an
    unpacked archive as well -- the warning would then fire on a set that was
    successfully extracted."""
    payload = tmp_path / "payload"
    payload.mkdir()
    whole = _tar_of(tmp_path / "w.tar.gz", ["a.wav"])
    blob = whole.read_bytes()
    (payload / "w.tar.gz.aa").write_bytes(blob[: len(blob) // 2])
    (payload / "w.tar.gz.ab").write_bytes(blob[len(blob) // 2:])
    whole.unlink()
    out = tmp_path / "out"
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pull.extract_archives(payload, out)
    assert "not unpacked" not in buf.getvalue()


def test_a_short_join_is_refused_rather_than_extracted(tmp_path):
    """🔴 The first version of `join_split` shelled out to `zip -s 0 --out`,
    the documented tool for spanned zips. On Codecfake it produced **10.7 GB
    from 32.1 GB of pieces**, silently, exit code 0. The only reason it was
    caught is that Python then refused the truncated archive as a possible zip
    bomb -- had the entry been smaller, it would have extracted part of a
    corpus and reported success.

    Concatenation makes the size a checkable claim. Mutation: drop the size
    comparison and this passes with a half-length archive."""
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "w.tar.gz.aa").write_bytes(b"a" * 100)
    (payload / "w.tar.gz.ab").write_bytes(b"b" * 100)
    pieces = [payload / "w.tar.gz.aa", payload / "w.tar.gz.ab"]
    base = payload / "w.tar.gz"
    work = tmp_path / "work"

    joined = pull.join_split(base, pieces, work)
    assert joined.stat().st_size == 200

    # a stale short join from an interrupted run is redone, not trusted
    joined.write_bytes(b"a" * 100)
    again = pull.join_split(base, pieces, work)
    assert again.stat().st_size == 200


def test_a_nested_zip_is_unpacked_in_place(tmp_path):
    """🔴 Codecfake's spanned zip holds exactly one entry: a 91 GB `train.zip`.
    One extraction pass leaves a corpus that is still a zip file."""
    inner = tmp_path / "train.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("a.wav", b"x" * 10)
        zf.writestr("b.wav", b"y" * 10)
    payload = tmp_path / "payload"
    payload.mkdir()
    outer = payload / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "deep/train.zip")
    inner.unlink()

    out = tmp_path / "out"
    assert pull.extract_archives(payload, out) == 2      # outer + nested
    assert {q.name for q in out.rglob("*.wav")} == {"a.wav", "b.wav"}
    assert not list(out.rglob("*.zip")), "the intermediate is not left behind"


@pytest.mark.skipif(not shutil.which("zip"), reason="needs Info-ZIP to build the fixture")
def test_a_real_spanned_zip_round_trips(tmp_path):
    """🔴 Built with the actual tool rather than mocked, because the thing that
    went wrong was a *behavioural* assumption: Python reads a small
    concatenated spanned zip happily and refuses a zip64 one with "zipfiles
    that span multiple disks are not supported". A first version of this
    dispatched on that message, so which code path ran depended on the
    publisher's archive size. Codecfake's is zip64 -- 91 GB in one entry -- and
    it fell through to a warning.

    Mutation: route spanned joins through `zipfile` and the extraction is
    silently empty for a zip64 set."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        (src / f"f{i}.bin").write_bytes(rng.bytes(120_000))

    payload = tmp_path / "payload"
    payload.mkdir()
    subprocess.run(["zip", "-q", "-r", "-s", "1m", str(payload / "whole.zip"), "src"],
                   cwd=tmp_path, check=True)
    pieces = sorted(q.name for q in payload.iterdir())
    assert len(pieces) >= 2, f"expected a split set, got {pieces}"

    groups = pull.split_groups(sorted(payload.iterdir()))
    assert list(groups) == [payload / "whole.zip"]
    assert pull.is_spanned_zip(payload / "whole.zip", groups[payload / "whole.zip"])

    out = tmp_path / "out"
    assert pull.extract_archives(payload, out) == 1
    got = sorted(q.name for q in out.rglob("*.bin"))
    assert got == sorted(q.name for q in src.iterdir())
    for q in out.rglob("*.bin"):
        assert q.read_bytes() == (src / q.name).read_bytes(), q.name


@pytest.mark.skipif(not shutil.which("zip"), reason="needs Info-ZIP to build the fixture")
def test_concatenating_a_spanned_zip_does_not_produce_a_readable_archive(tmp_path):
    """🔴 The measurement that settled the join, kept because the wrong answer
    is so plausible. Concatenating a spanned zip's pieces -- with or without
    stripping the 4-byte `PK\\x07\\x08` spanning signature -- gives a file whose
    **central directory reads** and whose **entries do not**. So a join checked
    by "does it list" passes and the corpus is still unreadable.

    Only `zip -s 0 --out` rewrites the offsets."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(1)
    for i in range(8):
        (src / f"f{i}.bin").write_bytes(rng.bytes(200_000))
    payload = tmp_path / "payload"
    payload.mkdir()
    subprocess.run(["zip", "-q", "-r", "-s", "1m", str(payload / "w.zip"), "src"],
                   cwd=tmp_path, check=True)
    pieces = sorted(payload.iterdir())
    assert len(pieces) >= 2

    cat = tmp_path / "cat.zip"
    with cat.open("wb") as fh:
        for q in pieces:
            fh.write(q.read_bytes())
    with zipfile.ZipFile(cat) as zf:              # the directory reads...
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        assert names
        with pytest.raises(zipfile.BadZipFile):   # ...and the entries do not
            zf.read(names[0])

    # the real join, and it round-trips
    joined = pull.join_split(payload / "w.zip", pieces, tmp_path / "work")
    with zipfile.ZipFile(joined) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            assert zf.read(info.filename) == (tmp_path / info.filename).read_bytes()


@pytest.mark.skipif(not shutil.which("zip"), reason="needs Info-ZIP to build the fixture")
def test_a_truncated_spanned_join_is_refused(tmp_path):
    """🔴 Info-ZIP 3.0 exits 0 and truncates spanned sets whose segments exceed
    4 GB. On Codecfake it wrote **10,737,418,467 bytes from 32,060,882,354** of
    pieces, silently. The size check is the only thing between that and a
    corpus that extracts part-way and reports success.

    Mutation: drop the comparison and this returns a short archive."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(np.random.default_rng(2).bytes(400_000))
    payload = tmp_path / "payload"
    payload.mkdir()
    subprocess.run(["zip", "-q", "-r", "-s", "1m", str(payload / "w.zip"), "src"],
                   cwd=tmp_path, check=True)
    pieces = sorted(payload.iterdir())

    work = tmp_path / "work"
    work.mkdir()
    # a short join left by a previous run, exactly the Codecfake shape
    (work / "w.joined.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    with pytest.raises(RuntimeError, match="bytes from pieces totalling"):
        pull.join_split(payload / "w.zip", pieces, work)
