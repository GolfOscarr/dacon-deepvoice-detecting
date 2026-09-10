"""BEATs encoder, vendored from microsoft/unilm.

Vendored rather than pip-installed for two reasons that both come from
docs/architecture/01-design-envelope.md: the submission installs offline
inside a 10-minute budget, and `beats` is not on PyPI at all -- the upstream
README's own instructions are "copy these files".

Provenance is in PROVENANCE.md; the licence is in LICENSE (MIT, read at
origin). `backbone.py` and `modules.py` are byte-for-byte upstream except for
one line: `from modules import (` -> `from .modules import (`, because a
package-relative import is what makes this a package rather than a pair of
top-level modules that would shadow anything else called `modules`.

⚠️ Do not edit these two files to fix our problems. Everything we need to
change lives in `models/frontends.py`, so that a future re-vendor is a copy
rather than a merge.
"""

from .backbone import TransformerEncoder

__all__ = ["TransformerEncoder"]
