"""Cross-file analyses. **These read parquet and never touch audio.**

That separation is what makes the standing rule affordable: docs/EDA/06 requires
re-running the shortcut audit and adversarial validation after *every* corpus
change, which is seconds against a table and hours against a corpus. It also
means every analysis is unit-testable on a synthetic DataFrame, with no audio
fixtures and no decode path to keep in sync.
"""
