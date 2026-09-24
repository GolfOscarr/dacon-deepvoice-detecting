"""`file_id` construction and decomposition, and path-component matching.

One implementation each, because both were written twice and one of them was
wrong both times. `eda.driver` built a `file_id` and `eda.analyze.grouping` and
`eda.gates` each took it apart again with their own `str.split(":")`; the
exclusion matcher existed in `driver.enumerate_source` and in `gates._g_eda1`
as two copies of the same substring test.

Critical: matching is on **path components**, not on substrings. The obvious
`rel.startswith(x) or f"/{x}" in f"/{rel}"` reads correctly and is wrong: with
`exclude: ["real"]` it also excludes `real_half/` and `a/really/`. On a licence
gate that errs toward excluding too much, which is the safe direction and still
makes the gate's "0 rows under [...]" report describe something other than what
was asked for.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable

__all__ = ["file_id_for", "relpath_of", "source_of", "under_any"]

#: Separates the source name from the path within it. A colon cannot appear in
#: a source name (`eda.config` builds them from the S3 directory) and is legal
#: but vanishingly rare in a path, so it partitions cleanly and stays readable
#: in an error message.
SEP = ":"


def file_id_for(source_name: str, rel: Path | PurePosixPath | str) -> str:
    """`<source>:<path relative to the source root>`.

    Critical: derived from the path rather than from an enumeration index, so a
    re-run that finds one more file does not renumber the corpus.
    """
    # Critical: a guard, not decoration. This used to take a `SourceSpec`, and
    # an f-string will happily interpolate one -- producing a `file_id` built
    # from a dataclass repr that then joins to nothing, silently. Anything with
    # a `__str__` is a valid argument to an f-string and almost none of them are
    # valid source names.
    if not isinstance(source_name, str):
        raise TypeError(
            f"source_name must be a str, got {type(source_name).__name__}. "
            f"Pass `source.name`, not the SourceSpec")
    if SEP in source_name:
        raise ValueError(
            f"source name {source_name!r} contains {SEP!r}, which separates the "
            f"two halves of a file_id")
    return f"{source_name}{SEP}{PurePosixPath(rel).as_posix()}"


def relpath_of(file_id: str) -> str:
    """The path half of a `file_id`. Raises on a malformed one rather than
    returning the whole string, which is how a bad join silently matches
    nothing."""
    source, sep, rel = file_id.partition(SEP)
    if not sep:
        raise ValueError(
            f"malformed file_id {file_id!r}: expected '<source>{SEP}<relpath>'")
    return rel


def source_of(file_id: str) -> str:
    source, sep, _ = file_id.partition(SEP)
    if not sep:
        raise ValueError(
            f"malformed file_id {file_id!r}: expected '<source>{SEP}<relpath>'")
    return source


def under_any(rel: str, prefixes: Iterable[str]) -> bool:
    """Is `rel` at or below any of `prefixes`, matching whole path components?

    `under_any("real_half/x.wav", ["real"])` is **False**; `under_any(
    "real/x.wav", ["real"])` and `under_any("a/real/x.wav", ["real"])` are True.
    A prefix may itself be multi-component (`env_sources/EnvSDD`).
    """
    parts = PurePosixPath(rel).parts
    for prefix in prefixes:
        want = PurePosixPath(prefix).parts
        if not want:
            continue
        n = len(want)
        if any(parts[i:i + n] == want for i in range(len(parts) - n + 1)):
            return True
    return False
