#!/usr/bin/env python3
"""Partition a per-track-licensed corpus into what we may ship and what we may not.

    python3 scripts/filter_track_licences.py fma  --meta <dir> --out <dir>
    python3 scripts/filter_track_licences.py jamendo --meta <dir> --out <dir>

FMA and MTG-Jamendo are single downloads whose audio is **not** under a single
licence. The dataset-level terms cover the code and the metadata; each track
carries its own grant, and in FMA's case there are 114 distinct ones. Waving the
archive through as "per-track CC" would put tracks we cannot ship into the
corpus, and finding that out at the 2nd-stage review is the expensive way.

The partition follows the rule text, not the vibe of the licence string:

*ALLOW* — the licence permits provision to a third party. Per DACON #417280 a
submission for 운영진 검증 is distinct from redistribution, and the only bar is a
licence that restricts 제3자 제공 *itself*. #417212 answers the NC and NC-SA
cases with yes. So CC0, public domain, BY, BY-SA, BY-NC and BY-NC-SA all pass.

*DERIVATIVES_BARRED* — the **ND** family. Provision of the unmodified file is
fine, but our pipeline's whole design is augmentation and composition
(docs/data/06), and ND plausibly bars the derivative we would actually train on.
Held separately rather than merged into either bucket: the call is a legal read
(the same one CtrSVDD is waiting on), not something this script should decide.

*DENY* — the licence restricts provision. FMA's "FMA-Limited: Download Only" is
the clear case, and anything unrecognised lands here too. Critical: unknown
defaults to deny. A licence string this script has never seen is not evidence of
permission, and the failure it would cause is silent.

Caveat: `license_title` is FMA's own metadata about somebody else's grant. It is
good enough to *exclude* on and not good enough to be the only record for what
we keep -- the ledger stores the string and the URL so a human can re-check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# Order matters: ND is checked before the permissive families, because
# "Attribution-NonCommercial-NoDerivatives" matches both.
DENY_PATTERNS = (
    r"fma-limited",
    r"download only",
    r"all rights reserved",
)
ND_PATTERNS = (
    r"noderiv",
    r"no deriv",
    r"nd\b",
)
ALLOW_PATTERNS = (
    r"^cc0",
    r"public domain",
    r"^attribution",
    r"creative commons attribution",
)


def classify(title: str | None) -> str:
    """One licence string -> allow | derivatives_barred | deny."""
    if title is None or not str(title).strip():
        return "deny"                      # nothing stated is not permission
    t = str(title).strip().lower()
    if any(re.search(p, t) for p in DENY_PATTERNS):
        return "deny"
    if any(re.search(p, t) for p in ND_PATTERNS):
        return "derivatives_barred"
    if any(re.search(p, t) for p in ALLOW_PATTERNS):
        return "allow"
    return "deny"                          # unrecognised defaults to deny


def load_fma(meta: pathlib.Path):
    import pandas as pd
    df = pd.read_csv(meta / "fma_metadata" / "raw_tracks.csv",
                     usecols=["track_id", "license_title", "license_url"],
                     low_memory=False)
    return df.rename(columns={"track_id": "id", "license_title": "licence",
                              "license_url": "licence_url"})


def load_jamendo(meta: pathlib.Path):
    """MTG-Jamendo's `audio_licenses.txt`.

    Critical: the licence is **not** in the dataset's TSVs. `raw.tsv` carries
    TRACK_ID / ARTIST_ID / ALBUM_ID / PATH / DURATION / TAGS and nothing about
    rights, so a filter built from the TSV alone would silently pass everything.
    The per-track grant lives in a separate `audio_licenses.txt` at the repo
    root -- not under `data/`, where the TSVs are -- as repeating blocks:

        14/214.mp3
        Intro chiante by David TMX from Jamendo: http://...track/214
        Available under a Creative Commons Attribution-... license: http://...
        <blank>

    Parsed positionally off the `.mp3` path line rather than by block size,
    because a missing blank line would otherwise shift every later track's
    licence by one -- a failure that produces a plausible-looking file.
    """
    import pandas as pd

    path = meta / "audio_licenses.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found -- fetch it from the repo root:\n"
                         "  https://raw.githubusercontent.com/MTG/"
                         "mtg-jamendo-dataset/master/audio_licenses.txt")

    rows, current = [], None
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.endswith(".mp3"):
            if current is not None:
                rows.append(current)
            current = {"id": line, "licence": None, "licence_url": ""}
        elif current is not None and "available under" in line.lower():
            lic, _, url = line.partition(": http")
            current["licence"] = lic.replace("Available under a", "").strip()
            current["licence_url"] = ("http" + url) if url else ""
    if current is not None:
        rows.append(current)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("corpus", choices=("fma", "jamendo"))
    p.add_argument("--meta", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    args = p.parse_args()

    df = {"fma": load_fma, "jamendo": load_jamendo}[args.corpus](args.meta)
    df["verdict"] = df["licence"].map(classify)

    args.out.mkdir(parents=True, exist_ok=True)
    counts = df["verdict"].value_counts().to_dict()
    total = len(df)

    print(f"{args.corpus}: {total} tracks")
    for verdict in ("allow", "derivatives_barred", "deny"):
        n = counts.get(verdict, 0)
        print(f"  {verdict:20s} {n:7d}  {100 * n / total:5.1f}%")

    for verdict in ("allow", "derivatives_barred", "deny"):
        sub = df[df["verdict"] == verdict]
        path = args.out / f"{args.corpus}_{verdict}.csv"
        sub.to_csv(path, index=False)
        print(f"  -> {path.name}  ({len(sub)} rows)")

    # What each bucket was actually made of, so a human can audit the mapping
    # rather than trusting the regexes.
    breakdown = {
        v: df[df["verdict"] == v]["licence"].fillna("(none stated)")
          .value_counts().head(30).to_dict()
        for v in ("allow", "derivatives_barred", "deny")
    }
    (args.out / f"{args.corpus}_breakdown.json").write_text(
        json.dumps(breakdown, indent=2, ensure_ascii=False) + "\n")
    print(f"  -> {args.corpus}_breakdown.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
