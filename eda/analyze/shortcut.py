"""X1 / E-S2: can the label be predicted from metadata alone?

This is the single most important number in the EDA, and the gate is
**AUC < 0.60** per head (docs/data/07 E-S2). It detects the failure mode that
kills this kind of project: a confound separating real from fake with no
acoustic content in it at all. Our corpus is five speech corpora, two
fake-speech corpora, two music corpora and one TTM corpus, each produced by a
different recording chain at a different sample rate in a different container --
so `FILE_FAKE` is nearly aligned with *which archive a file came from*, and a
classifier that learns the archive scores beautifully in CV and nothing on the
leaderboard.

⚠️ **What this can and cannot audit at the M tier.** Labels come from whichever
of the two sources the row kind says: a **component** row's labels are fixed by
its pool (`POOL_LABELS`), a **whole_file** row's by its cell (`CELL_TABLE`).
That is `validate_manifest`'s own rule, and it is why SONICS matters here --
49,074 AI songs enter as cell 8 and are the only rows in the corpus that are
positive on *both* fake heads at once.

`FILE_FAKE` is still not auditable at this tier for the composed majority: a
composed file's fake status does not exist until the sampler has drawn one. The
file head is audited by X5 over the spec stream, not here.

⚠️ **Two AUCs, and they answer different questions.**

* **ungrouped** is the gate. "Is the label predictable from metadata?" -- which
  is the question, and stratifying it by source would destroy exactly the effect
  being measured.
* **grouped by source** asks whether the shortcut *survives a source holdout*.
  A confound that persists when whole sources are held out is a property of the
  pools rather than of one archive, and is the worse of the two findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda.config import AnalysisConfig
# Critical: the private name is the point. `POOL_LABELS`' values are tuples, and
# their order is `manifest._LABEL_COLUMNS`' -- so writing the positions out here
# would mean that reordering those columns silently swaps this audit's heads,
# which is the same defect the repo already hit when the head-to-metric-weight
# mapping was untested and a File/Voice swap passed 749 of 749 tests.
# `training/config.py` reaches for `models.config._build` for the same reason.
from training.manifest import POOL_LABELS, _LABEL_COLUMNS
from training.spec import CELL_TABLE

__all__ = ["HEADS", "METADATA_FEATURES", "MISSING_BOOL", "build_design",
           "head_labels", "shortcut_audit", "univariate_auc"]

#: `head -> index into a POOL_LABELS tuple`, derived from the manifest's own
#: column order. A pool or a label column added there cannot drift out of here.
HEADS = {c.removeprefix("label_"): i for i, c in enumerate(_LABEL_COLUMNS)}

#: Booleans are tri-state here -- true, false, and "the container has no such
#: header". Folding absent into false would make a wav indistinguishable from an
#: mp3 with no Xing frame, and those are different facts.
MISSING_BOOL = -1.0

#: Physical, per-file, legal under rule 2.4. Critical: `source_name` and `pool`
#: are NOT features -- including the corpus id would make the audit trivially
#: score 1.0 and tell us nothing we did not already know.
METADATA_FEATURES = {
    "numeric": ("orig_sr", "orig_channels", "bit_rate", "duration_s",
                "file_bytes", "bits_per_raw_sample", "n_streams"),
    "boolean": ("xing_tag", "lame_tag", "id3_tag"),
    "categorical": ("container", "codec_name", "sample_fmt", "encoder", "lame_version"),
}


def head_labels(files: pd.DataFrame, head: str) -> pd.Series:
    """Label for one head; null where the metric ignores it.

    🔴 Two label sources, and `row_kind` picks between them -- exactly as
    `validate_manifest` does. A component's labels are fixed by its **pool**; a
    whole_file row has no pool at all (*"a whole file is used as-is, not drawn
    from a component pool"*) and its labels come from its **cell**.

    A fake label is `None` on the axis the row does not carry -- a pool-A row
    has no `music_fake`, and `POOL_LABELS` says so with `None`, the same way
    `CELL_TABLE` does. Those rows are dropped from that head's audit rather than
    coerced to 0, because the masked EER ignores them too.
    """
    idx = HEADS[head]
    out = pd.Series(pd.NA, index=files.index, dtype="object")

    if "pool" in files.columns:
        by_pool = {pool: labels[idx] for pool, labels in POOL_LABELS.items()}
        out = out.mask(files["pool"].notna(), files["pool"].map(by_pool))
    if "cell" in files.columns:
        by_cell = {cell: labels[idx] for cell, labels in CELL_TABLE.items()}
        cells = pd.to_numeric(files["cell"], errors="coerce")
        out = out.mask(cells.notna(), cells.map(by_cell))
    # `None` from either table means "the metric ignores this row for this
    # head", and so does a row neither table could label. Both become NA, which
    # `shortcut_audit` drops.
    return pd.to_numeric(out, errors="coerce")


def build_design(files: pd.DataFrame, *, top_k: int = 20,
                 features: dict[str, tuple[str, ...]] | None = None
                 ) -> tuple[np.ndarray, list[str]]:
    """Named columns to a dense float matrix. Metadata by default.

    ⚠️ `features` is a parameter because the same audit has to run over two
    different question sets: the M tier's metadata, and the S tier's per-plane
    acoustics (`eda.analyze.signal.plane_features`). Copying the function to
    change one dict is how the two would drift -- and the drift that matters is
    the missingness handling below, which is the subtle half.

    🔴 **Missingness is a feature, and deliberately so.** `bit_rate` is null for
    every wav and present for every mp3, so an imputation that hid that would
    hide the confound rather than measure it. Every numeric column contributes a
    median-imputed value *and* an `_isna` indicator.

    Categoricals are one-hot over the top `top_k` values with everything else
    folded into `__other__`, which keeps `encoder` -- effectively a corpus
    fingerprint -- usable without exploding the design.
    """
    features = METADATA_FEATURES if features is None else features
    unknown = sorted(set(features) - set(METADATA_FEATURES))
    if unknown:
        raise KeyError(
            f"feature spec has unknown group(s) {unknown}; it takes exactly "
            f"{sorted(METADATA_FEATURES)}, and a typo'd key would silently "
            f"contribute no columns at all")
    blocks, names = [], []
    for col in features.get("numeric", ()):
        raw = pd.to_numeric(files.get(col), errors="coerce") if col in files else pd.Series(
            np.nan, index=files.index)
        isna = raw.isna().to_numpy(dtype=float)
        med = raw.median()
        filled = raw.fillna(0.0 if pd.isna(med) else med).to_numpy(dtype=float)
        blocks += [filled[:, None], isna[:, None]]
        names += [col, f"{col}_isna"]
    for col in features.get("boolean", ()):
        raw = files.get(col)
        if raw is None:
            raw = pd.Series(np.nan, index=files.index)
        blocks.append(raw.map({True: 1.0, False: 0.0}).fillna(MISSING_BOOL)
                      .to_numpy(dtype=float)[:, None])
        names.append(col)
    for col in features.get("categorical", ()):
        raw = (files[col] if col in files else pd.Series(np.nan, index=files.index))
        raw = raw.astype("object").where(raw.notna(), "__missing__").astype(str)
        keep = list(raw.value_counts().head(top_k).index)
        folded = raw.where(raw.isin(keep), "__other__")
        for value in sorted(set(folded)):
            blocks.append((folded == value).to_numpy(dtype=float)[:, None])
            names.append(f"{col}={value}")
    return np.hstack(blocks) if blocks else np.zeros((len(files), 0)), names


def univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC of a single column. Ties averaged; direction-free (>= 0.5).

    No sklearn: this runs over every column of the design and the Mann-Whitney
    form is exact and cheap. Reported so the audit says *which* feature carries
    the shortcut -- neutralizing the wrong one is worse than neutralizing none.
    """
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(x).rank(method="average").to_numpy()
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def shortcut_audit(files: pd.DataFrame, cfg: AnalysisConfig | None = None, *,
                   features: dict[str, tuple[str, ...]] | None = None,
                   group_column: str = "source_name") -> pd.DataFrame:
    """One row per head: ungrouped AUC, grouped AUC, top features.

    🔴 `group_column` is the holdout axis, and the two available ones answer
    **different questions** -- neither is the stricter one, and reading them as
    a single "grouped AUC" is a mistake this docstring exists to prevent:

    * ``source_name`` -- the **archive** holdout. "Does this survive an unseen
      publisher?" It is what [06 X1c](../../docs/EDA/06-cross-pool.md) used and
      is the default for comparability. Harsh, and often unmeasurable: with four
      music sources, holding one out leaves a single-class fold.
    * ``group_key`` -- the **content** holdout. "Does this survive an unseen
      clip or speaker *from a publisher already in training*?" A held-out
      FakeMusicCaps clip leaves 5,520 others behind, so an archive-level
      confound is still fully available. Measured: it scores **higher**, not
      lower, exactly because it removes less.

    `group_key` is what makes `music_fake` measurable at all -- 5,764 groups
    against 4 sources -- and that number must never be quoted as though it were
    an archive holdout.

    Caveat: a head with fewer than two classes present, or with a source on only
    one side, reports `nan` rather than a number. That is the honest answer and
    it must not read as a pass -- `eda.gates` treats it as `na`, and `na` beats
    `pass` in the same ordering the VG tri-state uses.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = cfg or AnalysisConfig()
    rows = []
    for head in HEADS:
        y_all = head_labels(files, head)
        keep = y_all.notna()
        sub, y = files[keep], y_all[keep].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            rows.append({"head": head, "n": int(keep.sum()), "n_splits": 0,
                         "auc": float("nan"), "auc_source_grouped": float("nan"),
                         "top_features": "", "note": "fewer than two classes present"})
            continue
        x, names = build_design(sub, top_k=cfg.top_k, features=features)
        if group_column not in sub.columns:
            raise KeyError(
                f"cannot group the audit by {group_column!r}: not in the frame. "
                f"`group_key` is written by `consolidate` and reaches the signal "
                f"table through `load_signal`\'s join")
        groups = sub[group_column].to_numpy()

        def cv_auc(splitter, use_groups: bool) -> float:
            scores = []
            args = (x, y, groups) if use_groups else (x, y)
            for tr, te in splitter.split(*args):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, random_state=cfg.seed))
                model.fit(x[tr], y[tr])
                scores.append(roc_auc_score(y[te], model.predict_proba(x[te])[:, 1]))
            return float(np.mean(scores)) if scores else float("nan")

        # A class with fewer members than folds makes sklearn warn and the last
        # folds degenerate. Clamp, and report the clamp: a 2-fold audit and a
        # 5-fold audit are different measurements, and one silently substituted
        # for the other is how an ablation ends up comparing two things.
        minority = int(np.bincount(y).min())
        n_splits = max(2, min(cfg.n_splits, minority))
        auc = cv_auc(StratifiedKFold(n_splits, shuffle=True,
                                     random_state=cfg.seed), False)
        n_groups = len(np.unique(groups))
        grouped = (cv_auc(StratifiedGroupKFold(min(n_splits, n_groups)), True)
                   if n_groups >= 2 else float("nan"))
        # Caveat worth naming rather than leaving as a bare nan: when each source
        # sits entirely on one side of the label -- which is the normal case here,
        # since a corpus is either fake speech or real speech -- holding a source
        # out can leave a fold with one class in its test half, and every such
        # fold is skipped. All of them skipped is not "no shortcut"; it is "not
        # measurable this way".
        note = ("" if grouped == grouped else
                f"grouped AUC unmeasurable: {n_groups} {group_column} group(s), "
                f"and holding one out leaves a single-class fold")
        if n_splits != cfg.n_splits:
            note = "; ".join(filter(None, [
                note,
                f"cv narrowed to {n_splits} folds: the minority class has "
                f"{minority} rows"]))
        uni = sorted(((univariate_auc(x[:, i], y), names[i]) for i in range(x.shape[1])),
                     key=lambda t: (-(t[0] if t[0] == t[0] else 0), t[1]))
        rows.append({
            "head": head, "n": int(keep.sum()), "n_splits": n_splits, "auc": auc,
            "auc_source_grouped": grouped,
            "top_features": "; ".join(f"{n}={a:.3f}" for a, n in uni[:5]),
            "note": note,
        })
    return pd.DataFrame(rows)
