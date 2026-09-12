"""M tier: what a file *is*, as bytes.

Byte identity is pass 1 of the duplicate sweep (docs/EDA/05 E1). It is what
caught the RIRS/MUSAN overlap -- 88 files under the same basenames, different
`file_id`, different `source_name`, so `folds.grouping_atoms`' union-find could
not link them and the same recording sat in TRAIN on the fold where its twin was
in VAL.

⚠️ It is only pass 1. Byte identity cannot see the same recording re-encoded at
a different bitrate, which is what redistribution across music corpora actually
looks like -- and the specific open prediction in docs/EDA/05 is that MUSAN's
`music/fma/` and `music/jamendo/` subtrees are FMA and Jamendo material, which
we are separately acquiring. Pass 2 is the content fingerprint, and it needs a
decode, so it is S tier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from eda.extract import register_metadata

__all__ = ["IDENTITY_COLUMNS", "identity_row"]

IDENTITY_COLUMNS = ("identity_ok", "sha256", "file_bytes")


@register_metadata("identity", IDENTITY_COLUMNS, "identity_ok")
def identity_row(path: Path, probe: Any) -> dict[str, Any]:
    """Streaming sha256 and the byte count.

    Critical: the hash is over the **whole** file, not a prefix. A prefix hash
    would collide on two recordings sharing a container header, which is exactly
    the population this check runs on.
    """
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(probe.hash_chunk_bytes):
            h.update(chunk)
            size += len(chunk)
    return {"identity_ok": True, "sha256": h.hexdigest(), "file_bytes": size}
