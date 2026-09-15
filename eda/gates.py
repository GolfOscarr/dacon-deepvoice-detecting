"""The EDA gates, as a tri-state.

Critical: **`fail` beats `na` beats `pass`**, and the ordering is the point. It
is `training.validate`'s `quotable` rule, restated here because this repo has
paid for the alternative twice -- VG1 A10 reported SKIP on every run because
`folds.scheme_version` was null, so "the run's scheme_version matches
folds.parquet's" compared nothing and read as fine; and `quotable` was earnable
entirely by SKIPs until the tri-state landed.

A gate whose input was never computed reports **`na`**, never `pass`. A gate
that cannot fail is not a gate.

Phase scoping is explicit: G-EDA4, G-EDA6 and G-EDA7 need the S tier, a built
fold table, or a filter that does not exist yet, so they report `na` with the
reason. That is not a stub -- it is the correct verdict for a check whose input
is absent, and it is the difference between "we looked and it was fine" and "we
have not looked".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eda.analyze.duplicates import SUMMARY_KEYS
from eda.config import EdaConfig
from eda.ids import relpath_of

__all__ = ["FAIL", "GateResult", "NA", "PASS", "run_gates", "worst"]

PASS, NA, FAIL = "pass", "na", "fail"
#: Lower is worse. `min` over this is the aggregate verdict.
_ORDER = {FAIL: 0, NA: 1, PASS: 2}


def worst(verdicts) -> str:
    verdicts = list(verdicts)
    return min(verdicts, key=lambda v: _ORDER[v]) if verdicts else NA


@dataclass
class GateResult:
    gate: str
    verdict: str
    detail: str

    def as_row(self) -> dict:
        return {"gate": self.gate, "verdict": self.verdict, "detail": self.detail}


def _as_int(value: str) -> str:
    """`000002` -> `2`; anything non-numeric unchanged."""
    v = value.strip()
    return str(int(v)) if v.lstrip("-").isdigit() else v


def _allowlist_ids(values: pd.Series, key: str, *, is_id: bool = False) -> pd.Series:
    """One implementation for both sides of the comparison.

    Critical: the same transform must run on the file paths and on the CSV
    column, or the two sets are in different spaces and nothing matches. Running
    it twice from two call sites is how that drifts.
    """
    if key == "relpath":
        return values.astype(str).str.strip()
    stems = values.astype(str).map(
        lambda v: v.strip() if is_id else Path(v).stem)
    if key == "int_stem":
        return stems.map(lambda v: _as_int(Path(v).stem if is_id else v))
    return stems


def _g_eda1(cfg: EdaConfig, files: pd.DataFrame) -> list[GateResult]:
    """Licence. Exclusions honoured; every allowlisted source's ids permitted."""
    out = []
    for source in cfg.sources:
        sub = files[files["source_name"] == source.name]
        if source.exclude:
            # `SourceSpec` owns the matching; this used to be a second copy of
            # the substring test in `driver.enumerate_source`, and both were
            # wrong the same way.
            rel = sub["file_id"].map(relpath_of)
            bad = rel[rel.map(source.excludes_path)]
            out.append(GateResult(
                f"G-EDA1/exclude/{source.name}",
                NA if sub.empty else (FAIL if len(bad) else PASS),
                f"no rows: {source.name} is not probed yet"
                if sub.empty else
                f"{len(bad)} row(s) under an excluded subtree ({source.exclude_reason})"
                if len(bad) else f"0 rows under {list(source.exclude)}"))
        if source.pool == "D" and not source.exclude:
            out.append(GateResult(
                f"G-EDA1/exclude/{source.name}", NA,
                "pool D declares no exclusion. docs/EDA/04 D1 is a blocking licence "
                "gate: FakeMusicCaps is MusicCaps re-generated, and MusicCaps itself "
                "is YouTube-derived and ruled out by docs/data/01. Enumerate the "
                "archive and record the decision -- `na`, not `pass`"))
        if not source.allowlist:
            continue
        path = cfg.root / source.allowlist
        if not path.exists():
            out.append(GateResult(
                f"G-EDA1/allowlist/{source.name}", NA,
                f"allowlist {path} not present -- the licence partition lives in "
                f"s3://<bucket>/dacon-deepfake-detection/data/licences/ and has "
                f"not been pulled. NOT a pass: docs/data/12 section 4"))
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if source.allowlist_column not in (reader.fieldnames or []):
                out.append(GateResult(
                    f"G-EDA1/allowlist/{source.name}", NA,
                    f"{source.allowlist} has no column "
                    f"{source.allowlist_column!r}; it has "
                    f"{reader.fieldnames}. Not a FAIL -- a mis-named column "
                    f"would reject every row for a reason that is not a licence"))
                continue
            allowed = {row[source.allowlist_column].strip() for row in reader}
        # Critical: ids come from the path *below the source root*, which is
        # what an allowlist is written against. `path` is relative to
        # EdaConfig.root and carries the source directory with it.
        rel = sub["file_id"].map(relpath_of)
        ids = _allowlist_ids(rel, source.allowlist_key)
        allowed = set(_allowlist_ids(pd.Series(sorted(allowed)),
                                     source.allowlist_key, is_id=True))
        if sub.empty:
            out.append(GateResult(
                f"G-EDA1/allowlist/{source.name}", NA,
                f"no rows: {source.name} is not probed yet. An allowlist check "
                f"over zero files is a vacuous pass"))
            continue
        bad = ids[~ids.isin(allowed)]
        out.append(GateResult(
            f"G-EDA1/allowlist/{source.name}",
            FAIL if len(bad) else PASS,
            f"{len(bad)} of {len(sub)} rows are not in {source.allowlist}"
            if len(bad) else f"all {len(sub)} rows permitted"))
    return out or [GateResult("G-EDA1", NA, "no source declares an exclusion or an allowlist")]


def _g_eda2(cfg: EdaConfig, audit: pd.DataFrame | None) -> GateResult:
    """Shortcut. Metadata-only AUC below the threshold, per head."""
    if audit is None or audit.empty:
        return GateResult("G-EDA2", NA, "shortcut audit has not run")
    limit = cfg.gates.shortcut_auc_max
    scored = audit[audit["auc"].notna()]
    if scored.empty:
        return GateResult("G-EDA2", NA, "no head had two classes present")
    over = scored[scored["auc"] >= limit]
    verdict = FAIL if len(over) else PASS
    if len(scored) < len(audit):
        verdict = worst([verdict, NA])
    worst_row = scored.loc[scored["auc"].idxmax()]
    return GateResult(
        "G-EDA2", verdict,
        f"max AUC {worst_row['auc']:.3f} on {worst_row['head']} "
        f"(limit {limit}); {len(over)} head(s) over; "
        f"{len(audit) - len(scored)} head(s) unscored")


def _g_eda3(cfg: EdaConfig, groups: pd.DataFrame | None) -> GateResult:
    """Groups. Enough independent grouping atoms per source.

    🔴 The detail separates **two kinds of shortfall**, because only one of them
    is work. A source whose key was read from the publisher and still falls
    short has as many atoms as exist; a source with no key might have plenty and
    nobody has looked. Reporting them in one list is what made SONICS -- 5 real
    generators, reported as 1 -- indistinguishable from LJSpeech, which is 1 and
    always will be.
    """
    if groups is None or groups.empty:
        return GateResult("G-EDA3", NA, "grouping report has not run")
    short = groups[~groups["meets_floor"]]
    kinds = (short["group_key_kind"] if "group_key_kind" in short.columns
             else pd.Series("path", index=short.index))
    stated = short[kinds != "path"]
    unkeyed = short[kinds == "path"]

    def names(frame: pd.DataFrame) -> str:
        # 🔴 `.head(5)` silently truncated. The 6th source to fail was SONICS
        # and the detail read "6 of 9 source(s) below 6 ... : <five names>",
        # which reads as a transcription error rather than a cut list -- and
        # the missing name is the one the reader has not thought about yet.
        listed = ", ".join(frame["source_name"].head(5))
        return listed + (f" (+{len(frame) - 5} more)" if len(frame) > 5 else "")

    parts = [f"{len(short)} of {len(groups)} source(s) below "
             f"{cfg.gates.min_groups_per_role} groups"]
    if len(unkeyed):
        parts.append(f"{len(unkeyed)} with no publisher key, path depth only: "
                     f"{names(unkeyed)}")
    if len(stated):
        parts.append(f"{len(stated)} keyed and genuinely short -- this is the "
                     f"count, not a gap: {names(stated)}")
    return GateResult("G-EDA3", FAIL if len(short) else PASS, "; ".join(parts))


def _g_eda5(cfg: EdaConfig, dupes: dict | None) -> GateResult:
    """Duplicates. No byte-identical file shared across sources.

    Critical: `dupes` is `None` only when the sweep **has not run**. A sweep
    that ran and found nothing passes a zeroed summary, and reports `pass`. The
    two were indistinguishable while the detail table was the only artifact --
    a clean corpus wrote no parquet, so a later `eda gates` read `na` for what
    was actually a pass.
    """
    if dupes is None:
        return GateResult("G-EDA5", NA, "duplicate sweep has not run")
    missing = [k for k in SUMMARY_KEYS if k not in dupes]
    if missing:
        return GateResult("G-EDA5", NA,
                          f"duplicate summary is missing {missing}")
    n = dupes["cross_source_groups"]
    limit = cfg.gates.max_cross_source_duplicates
    return GateResult(
        "G-EDA5", FAIL if n > limit else PASS,
        f"{n} cross-source duplicate group(s), {dupes['cross_pool_groups']} of them "
        f"cross-pool; {dupes['groups']} groups over {dupes['rows']} rows total. "
        f"Byte identity only -- pass 2 (content fingerprint) is S tier")


def _g_eda6(cfg: EdaConfig, content: pd.DataFrame | None) -> GateResult:
    """Evidence. Every row's asserted components are evidenced, or explained.

    🔴 Two different shortfalls, and only one of them is the corpus's fault.

    A **contradiction** is a row whose audio disputes its own assertion: a
    pool-C file that asserts `voice_present = 0` with speech through 40% of it.
    07 says reassign (F-A1) before dropping, and the gate fails so that someone
    has to.

    An **unevidenceable** row is one the detector cannot speak to. Silero VAD
    detects speech, and SONICS sings: the publisher's own CSV reports
    `no_vocal = False` for all 49,074 rows while the VAD finds speech in 28% of
    them. Counting those as contradictions would claim 72% of cell 8 has no
    voice -- and acting on it would relabel 49k AI songs as voiceless, which is
    the defect the corpus was restructured to avoid. They are reported and do
    **not** fail the gate.

    ⚠️ Nor do they pass it silently. A source that is entirely unevidenceable
    has had no evidence gathered about it at all, which is `na`'s meaning in
    this repo's tri-state -- and `na` never reads as a pass.
    """
    if content is None or content.empty:
        return GateResult(
            "G-EDA6", NA, "component-evidence checks need the S tier "
            "(VAD/PANNs/energy) -- Phase 1. docs/EDA/07")
    bad = content[content["contradicted"] > 0].sort_values(
        "contradicted", ascending=False)
    total = int(content["contradicted"].sum())
    measured = int(content["measured"].sum())
    unevidenced = int(content.get("unevidenceable", pd.Series(dtype=int)).sum())
    note = (f"; {unevidenced} row(s) unevidenceable by a speech VAD "
            f"(sung: {', '.join(sorted(content.loc[content['sung'], 'source_name']))})"
            if unevidenced else "")
    if not total:
        return GateResult("G-EDA6", PASS,
                          f"0 of {measured} measured row(s) contradict their "
                          f"asserted components{note}")
    named = ", ".join(f"{r.source_name} {int(r.contradicted)}"
                      for r in list(bad.itertuples())[:5])
    more = f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""
    return GateResult(
        "G-EDA6", FAIL,
        f"{total} of {measured} measured row(s) contradict their asserted "
        f"components, over {len(bad)} source(s): {named}{more}{note}. "
        f"Reassign (F-A1) before dropping")


def run_gates(cfg: EdaConfig, files: pd.DataFrame, *, audit=None, groups=None,
              dupes=None, content=None) -> pd.DataFrame:
    """Every gate, plus the aggregate. Exit status is `worst` over the column."""
    results: list[GateResult] = []
    results += _g_eda1(cfg, files)
    results.append(_g_eda2(cfg, audit))
    results.append(_g_eda3(cfg, groups))
    results.append(GateResult(
        "G-EDA4", NA, "pair_id/dup_group vs fold boundaries -- needs a built fold "
        "table (Phase 2). docs/EDA/02 B1"))
    results.append(_g_eda5(cfg, dupes))
    results.append(_g_eda6(cfg, content))
    results.append(GateResult(
        "G-EDA7", NA, "filter-rate symmetry needs a filter; none is applied in "
        "Phase 0 by design (R2)"))
    df = pd.DataFrame([r.as_row() for r in results])
    df.attrs["aggregate"] = worst(df["verdict"])
    return df
