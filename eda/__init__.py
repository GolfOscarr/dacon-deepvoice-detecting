"""EDA: what is actually in the corpus, measured rather than assumed.

The plan this implements is `docs/EDA/`. Three rules carry over from it and are
enforced here rather than documented:

* **R1 -- two measurement planes.** Every signal statistic is computed on the
  file as published (`native`) and again after the 16 kHz chain the model sees
  (`chain`). Extractors are never told which plane they are running on, so none
  of them can branch on it.
* **R2 -- a filter is a distribution-shift decision.** Nothing in this package
  drops a row. Extraction emits every file it is pointed at, failures included.
* **Emit every column you compute.** `scripts/build_test_corpus.py` computed a
  sha256 per file and hardcoded `dup_group = None`; the 88-file split leak was a
  value that was calculated and thrown away.
"""
