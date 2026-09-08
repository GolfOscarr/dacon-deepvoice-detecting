"""`training.synthetic` -- the reproducibility claim its docstrings make.

`write_synthetic_corpus` says "two runs produce byte-identical corpora", and
that is what makes a reproducibility test elsewhere in this suite a test of the
pipeline rather than of the fixture. It was true of the audio and false of the
manifest: the ``sha256`` column was built from Python's `hash()`, which is
salted per process, so the column differed between runs and nothing noticed --
nothing reads it.

Critical: the assertions here run **two interpreters**, at two explicit
``PYTHONHASHSEED`` values. An in-process check cannot see this class of defect
at all: within one process `hash()` is perfectly stable, which is exactly how it
survives review.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from training.synthetic import synthetic_manifest

ROOT = Path(__file__).resolve().parent.parent

_MANIFEST_PROBE = f"""
import hashlib, sys
sys.path.insert(0, {str(ROOT)!r})
from training.synthetic import synthetic_manifest
m = synthetic_manifest(n_per_pool=20, n_whole_file=20, seed=0)
print(hashlib.blake2b(m.to_csv(index=False).encode(), digest_size=16).hexdigest())
print(hashlib.blake2b("".join(m["sha256"]).encode(), digest_size=16).hexdigest())
"""

_CORPUS_PROBE = f"""
import hashlib, sys, tempfile
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from training.synthetic import synthetic_manifest, write_synthetic_corpus
m = synthetic_manifest(n_per_pool=2, n_whole_file=2, seed=0, duration_range=(0.4, 0.5))
root = Path(tempfile.mkdtemp())
write_synthetic_corpus(m, root, seed=0)
h = hashlib.blake2b(digest_size=16)
for p in sorted(root.rglob("*")):
    if p.is_file():
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
print(h.hexdigest())
"""


def _in_subprocess(source: str, hash_seed: str) -> list[str]:
    """Run `source` in a fresh interpreter at a chosen `PYTHONHASHSEED`.

    Chosen rather than left to chance: hash randomisation is on by default, so
    two unseeded runs *usually* disagree, but "usually" is a flaky test. Two
    fixed, different seeds reproduce the cross-process condition every time.
    """
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    done = subprocess.run([sys.executable, "-c", source], capture_output=True,
                          text=True, env=env, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def test_the_manifest_is_identical_across_processes():
    """Two interpreters, two hash seeds, one manifest.

    Critical: this is the whole of D4. `sha256` came from
    `abs(hash(file_id)) & 0xFFFFFFFFFFFF`, and `hash()` on a `str` is salted
    from `PYTHONHASHSEED`, so the column was a different 12 hex digits in every
    process. Nothing reads it today, which is why a green suite said nothing;
    the moment anything joins two manifests on it, or diffs one run's manifest
    against another's, the mismatch is silent and total.
    """
    a = _in_subprocess(_MANIFEST_PROBE, "0")
    b = _in_subprocess(_MANIFEST_PROBE, "1")
    assert a[1] == b[1], f"the sha256 column differs across processes: {a[1]} vs {b[1]}"
    assert a[0] == b[0], "the manifest differs across processes"


def test_the_written_corpus_is_byte_identical_across_processes():
    """The other half of `write_synthetic_corpus`'s claim, on the bytes.

    This half was already right -- the audio keys blake2b on `file_id` -- and it
    is asserted anyway: it is the claim the manifest half was quietly failing,
    and an audio path that acquired the same defect would be far worse.
    """
    a = _in_subprocess(_CORPUS_PROBE, "0")
    b = _in_subprocess(_CORPUS_PROBE, "1")
    assert a == b, "two processes wrote different corpora"


def test_the_digest_column_is_a_function_of_the_file_id_alone():
    """The contract in-process, so the subprocess test has something to mean.

    Equal `file_id`s give equal digests and different ones give different
    digests -- otherwise a stable column could be stable because it is constant.
    """
    m = synthetic_manifest(n_per_pool=20, n_whole_file=20, seed=0)
    by_id = dict(zip(m["file_id"], m["sha256"]))
    assert len(by_id) == len(m), "file_id is not unique; the mapping below is not one"
    assert len(set(by_id.values())) == len(by_id), "digests collide across file_ids"
    for file_id, digest in by_id.items():
        assert digest == hashlib.blake2b(file_id.encode(), digest_size=6).hexdigest()

    # And it moves with the id rather than with the row's position: a manifest
    # of a different size gives the rows it shares the same digests.
    wider = synthetic_manifest(n_per_pool=40, n_whole_file=40, seed=0)
    shared = dict(zip(wider["file_id"], wider["sha256"]))
    common = set(by_id) & set(shared)
    assert common
    assert all(by_id[k] == shared[k] for k in common)


def test_the_hash_keyed_digest_that_shipped_does_differ_across_processes():
    """MUTATION for the two-process test above: the expression that shipped.

    `abs(hash(file_id)) & 0xFFFFFFFFFFFF` run in two interpreters at two hash
    seeds, and nothing else. If this came out equal, the test above would be
    passing because `_in_subprocess` cannot see a per-process salt rather than
    because the manifest is reproducible -- which is the only way a check on
    "two runs agree" can be vacuous.
    """
    old = '''print(f'{abs(hash("B00000")) & 0xFFFFFFFFFFFF:012x}')'''
    a = _in_subprocess(old, "0")
    b = _in_subprocess(old, "1")
    assert a != b, (
        "the salted-hash expression agreed across two hash seeds, so the "
        "reproducibility tests above are not measuring what they claim")
