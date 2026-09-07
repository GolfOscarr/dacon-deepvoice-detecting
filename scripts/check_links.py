#!/usr/bin/env python3
"""Check every relative markdown link and heading anchor in the repo.

Two failure modes this catches, both of which shipped silently before it existed:

* a link to a file that does not exist;
* a link to an anchor that does not exist -- usually because the target is a
  *table row* rather than a heading. GitHub only generates anchors for headings.

Anchors are computed with GitHub's algorithm: render inline links to their label,
drop everything that is not alphanumeric / space / hyphen (which removes emoji and
punctuation but leaves the spaces around them), lowercase, then spaces to hyphens.
A heading ending in an emoji therefore keeps a trailing hyphen -- that is correct,
not a typo.

Note the `{#custom-id}` attribute syntax is NOT supported by GitHub Flavored
Markdown and is deliberately not honoured here: it renders as literal text.

    python3 scripts/check_links.py     # exit 1 if anything is broken
"""

import pathlib
import re
import sys

SKIP = (".venv", ".git", "node_modules", ".pytest_cache")


def slug(heading: str) -> str:
    """GitHub's heading -> anchor transformation."""
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)   # [label](url) -> label
    h = re.sub(r"[`*_]", "", h)                            # code / emphasis marks
    s = "".join(c for c in h.strip().lower() if c.isalnum() or c in " -_")
    return s.replace(" ", "-")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    files = [m for m in root.rglob("*.md") if not any(s in str(m) for s in SKIP)]

    anchors = {
        md.resolve(): {slug(h) for h in re.findall(r"^#{1,6}\s+(.*)$", md.read_text(), re.M)}
        for md in files
    }

    broken = []
    for md in sorted(files):
        rel = md.relative_to(root)
        for _, link in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", md.read_text()):
            if link.startswith(("http", "mailto:", "#!")):
                continue
            path, _, frag = link.partition("#")
            target = (md.parent / path).resolve() if path else md.resolve()
            if not target.exists():
                broken.append(f"{rel}: missing file   -> {link}")
            elif frag and target.suffix == ".md" and frag not in anchors.get(target, set()):
                broken.append(f"{rel}: missing anchor -> {link}")

    if broken:
        print("\n".join(broken))
        print(f"\n{len(broken)} broken link(s) in {len(files)} files")
        return 1
    print(f"all links and anchors OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
