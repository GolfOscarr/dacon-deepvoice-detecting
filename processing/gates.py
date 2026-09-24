"""The EDA gates that could not be measured until the tables were built
(docs/EDA/07 §gates; docs/processing/03 §7 items 4 and 5).

``eda.gates`` runs G-EDA1..6 over the EDA's own artifacts and reports G-EDA4
and G-EDA7 as ``na`` because they bind on a fold table and on applied filters,
neither of which exists before OFF-1..OFF-5. This module measures them, plus
G-EDA3 over the atoms the folds were actually grouped by, over the built
``manifest.parquet`` / ``verdict.parquet`` / ``folds.parquet`` -- and writes the
two review tables docs/data/10 asks for: the G4 loss table (what every filter
dropped, per pool and per cell) and the G7 pack (what was reassigned, per
source, with examples to listen to).

Critical: every rate here is measured on the EDA's file table (every probed
file), not on the manifest (the survivors). A filter's rate per label is only
visible from the side that includes what it removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from training.manifest import POOL_IS_FAKE, ROLE_POOLS

__all__ = ["GateResult", "g4_table", "g7_pack", "g_eda3", "g_eda4", "g_eda7", "run_gates"]

PASS, FAIL, NA = "pass", "fail", "na"

#: G-EDA7: the largest |drop rate(FAKE) - drop rate(REAL)| a filter may have
#: on one role before it is manufacturing a cue. docs/EDA/07 asks for "a
#: stated tolerance" and states none; 0.10 is docs/data/10 G4's salvage-parity
#: line, adopted for this gate on 2026-09-24 (docs/processing/06 §0). The one
#: standing exception is the duration floor, recorded in the G-EDA7 ledger.
G_EDA7_TOL = 0.10
#: G4: a filter that drops more than this share of any pool escalates.
G4_MAX_DROP = 0.05
#: G-EDA3: independent grouping atoms per role and side.
MIN_ATOMS = 6


@dataclass(frozen=True)
class GateResult:
    gate: str
    verdict: str
    detail: str

    def as_row(self) -> dict[str, str]:
        return {"gate": self.gate, "verdict": self.verdict, "detail": self.detail}


def _label_of(frame: pd.DataFrame) -> pd.Series:
    """FILE_FAKE per row: the pool's fake status for a component row, the cell
    for a whole-file row (5 real, 8 fake). ``None`` where neither says."""
    pool = frame["pool"].map(POOL_IS_FAKE)
    cell = pd.to_numeric(frame["cell"], errors="coerce")
    by_cell = cell.map({5.0: False, 8.0: True})
    return pool.where(pool.notna(), by_cell)


# --------------------------------------------------------------------------- #
# G-EDA3 -- atoms per role, measured on what the folds grouped by


def g_eda3(manifest: pd.DataFrame, floor: int = MIN_ATOMS) -> GateResult:
    """At least ``floor`` grouping atoms on every (role, side). The atom is the
    one ``processing.corpus`` wrote into ``source_name`` (publisher: family /
    sub-corpus / speaker / artist) -- for a fake side, the generator family
    is the stricter atom and is reported beside it."""
    comp = manifest[manifest["row_kind"] == "component"]
    parts, short = [], []
    for role, pools in ROLE_POOLS.items():
        for pool in pools:
            sub = comp[comp["pool"] == pool]
            atoms = int(sub["source_name"].nunique())
            fam = int(sub["artifact_family"].nunique()) if POOL_IS_FAKE[pool] else None
            side = "fake" if POOL_IS_FAKE[pool] else "real"
            tag = f"{role}/{side} ({pool}) {atoms} atoms" + (f", {fam} families" if fam else "")
            parts.append(tag)
            n_eff = min(atoms, fam) if fam is not None else atoms
            if n_eff < floor:
                short.append(tag)
    detail = f"floor {floor}: " + "; ".join(parts)
    if short:
        detail += f" -- below the floor: {', '.join(short)}"
    return GateResult("G-EDA3", FAIL if short else PASS, detail)


# --------------------------------------------------------------------------- #
# G-EDA4 -- no pair or dup set straddles a fold boundary


def g_eda4(folds: pd.DataFrame) -> GateResult:
    """Every ``pair_id`` set and every ``dup_group`` set sits in ONE fold (and
    one slice: a pair with one row in PROBE and one in a fold straddles too).
    Verified on the built table, not on the intent (docs/EDA/07)."""
    bad: dict[str, int] = {}
    total: dict[str, int] = {}
    # Critical: a PROBE row has no fold; NaN would drop out of `nunique` and
    # hide a pair split between PROBE and a fold, so it is spelled out.
    fold = folds["fold"].map(lambda x: "-" if pd.isna(x) else str(int(x)))
    where = folds["slice"].astype(str) + "/" + fold
    for key in ("pair_id", "dup_group"):
        if key not in folds.columns:
            continue
        sets = folds[folds[key].notna()].groupby(key)
        sizes = sets.size()
        multi = sizes[sizes > 1].index
        total[key] = int(len(multi))
        if len(multi) == 0:
            bad[key] = 0
            continue
        spread = where[folds[key].isin(multi)].groupby(folds[key]).nunique()
        bad[key] = int((spread > 1).sum())
    detail = "; ".join(f"{k}: {bad[k]} of {total[k]} multi-row set(s) straddle a fold or slice"
                       for k in bad)
    if not bad:
        return GateResult("G-EDA4", NA, "no pair_id / dup_group column in the fold table")
    return GateResult("G-EDA4", FAIL if any(bad.values()) else PASS, detail)


# --------------------------------------------------------------------------- #
# G-EDA7 -- every filter's rate, per label


#: The role a row's pool or cell belongs to, for the per-role ledger.
_ROLE_OF = {"A": "voice", "B": "voice", "C": "music", "D": "music", "E": "noise"}


def g_eda7_ledger(verdict: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    """One row per (filter, role, label): rows judged, rows dropped, drop
    rate. ``files`` is the EDA file table (every probed file), which carries
    the pool and cell the label and role are read from. Per ROLE, because a
    filter that drops fake voice and real music at the same rate is not
    symmetric on either head (docs/processing/05 B13): pooled, the 4 s floor
    read as a 0.126 gap; within voice it is 0.187."""
    keyed = files.set_index("file_id")
    lab = _label_of(keyed)
    cell = pd.to_numeric(keyed["cell"], errors="coerce")
    role = keyed["pool"].map(_ROLE_OF).where(keyed["pool"].notna(),
                                             cell.map(lambda c: "file" if pd.notna(c) else None))
    v = verdict.copy()
    v["file_fake"] = v["file_id"].map(lab)
    v["role"] = v["file_id"].map(role)
    v = v[v["file_fake"].notna() & v["role"].notna()]
    v["dropped"] = v["verdict"].eq("drop")
    g = (v.groupby(["filter", "role", "file_fake"])["dropped"]
         .agg(n="size", dropped="sum").reset_index())
    g["rate"] = g["dropped"] / g["n"]
    g["file_fake"] = g["file_fake"].astype(bool)
    return g


def g_eda7(verdict: pd.DataFrame, files: pd.DataFrame, tol: float = G_EDA7_TOL
           ) -> tuple[GateResult, pd.DataFrame]:
    """Symmetry: for every (filter, role) with rows on both sides,
    |rate(FAKE) - rate(REAL)| <= ``tol``. A filter that only ever sees one
    label (a licence gate on real music, a label-evidence drop of fake voice)
    is reported as one-sided, not as a gap of 1.0. Returns the gate and the
    ledger it was read from."""
    ledger = g_eda7_ledger(verdict, files)
    parts, worst, one_sided = [], [], []
    for (name, role), sub in ledger.groupby(["filter", "role"]):
        rates = sub.set_index("file_fake")["rate"]
        if True not in rates.index or False not in rates.index:
            side = "fake" if True in rates.index else "real"
            one_sided.append(f"{name}/{role} ({side} only, rate {float(rates.iloc[0]):.3f})")
            continue
        r_fake, r_real = float(rates[True]), float(rates[False])
        gap = abs(r_fake - r_real)
        parts.append(f"{name}/{role}: fake {r_fake:.3f} vs real {r_real:.3f} (gap {gap:.3f})")
        if gap > tol:
            worst.append(f"{name}/{role}")
    detail = f"tol {tol:.2f}: " + "; ".join(parts)
    if one_sided:
        detail += " -- one-sided: " + ", ".join(one_sided)
    if worst:
        detail += f" -- asymmetric: {', '.join(worst)}"
    return GateResult("G-EDA7", FAIL if worst else PASS, detail), ledger


# --------------------------------------------------------------------------- #
# G4 -- the loss table


def g4_table(verdict: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    """Per (filter, pool-or-cell): rows judged, dropped, rate, hours dropped.
    A whole-file row is keyed by its cell (``cell5`` / ``cell8``)."""
    keyed = files.set_index("file_id")
    cell = pd.to_numeric(keyed["cell"], errors="coerce")
    unit = keyed["pool"].where(keyed["pool"].notna(),
                               cell.map(lambda c: f"cell{int(c)}" if pd.notna(c) else None))
    v = verdict.copy()
    v["unit"] = v["file_id"].map(unit)
    v["hours"] = v["file_id"].map(keyed["duration_s"]).fillna(0.0) / 3600.0
    v["dropped"] = v["verdict"].eq("drop")
    v = v[v["unit"].notna()]
    g = v.groupby(["filter", "unit"]).apply(
        lambda s: pd.Series({"n": len(s), "dropped": int(s["dropped"].sum()),
                             "hours": float(s["hours"].sum()),
                             "hours_dropped": float(s.loc[s["dropped"], "hours"].sum())}),
        include_groups=False).reset_index()
    g["rate"] = g["dropped"] / g["n"]
    g["escalates"] = g["rate"] > G4_MAX_DROP
    return g


# --------------------------------------------------------------------------- #
# G7 -- the reassignment pack


def g7_pack(manifest: pd.DataFrame, files: pd.DataFrame, n_examples: int = 10,
            seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (corpus, from-pool, to-cell): rows reassigned and the rate against
    the corpus's rows that sit in that pool or were moved out of it -- both
    counted on the MANIFEST, after the drops, so the rate is a share of what
    is used. Plus ``n_examples`` rows per (from, to) to listen to.
    ``reassigned_from`` is the manifest column OFF-2 writes; ``corpus`` is the
    EDA's source name (the manifest's ``source_name`` is the finer publisher
    atom the folds group by)."""
    corpus_col = "corpus" if "corpus" in manifest.columns else "source_name"
    moved = manifest[manifest["reassigned_from"].notna()]
    by = (moved.groupby([corpus_col, "reassigned_from", "cell"]).size()
          .rename("reassigned").reset_index().rename(columns={corpus_col: "corpus"}))
    origin = manifest["pool"].where(manifest["pool"].notna(), manifest["reassigned_from"])
    pool_rows = (manifest.assign(origin=origin).dropna(subset=["origin"])
                 .groupby([corpus_col, "origin"]).size().rename("corpus_rows").reset_index()
                 .rename(columns={corpus_col: "corpus", "origin": "reassigned_from"}))
    table = by.merge(pool_rows, on=["corpus", "reassigned_from"], how="left")
    table["rate"] = table["reassigned"] / table["corpus_rows"]
    table["cell"] = pd.to_numeric(table["cell"]).astype(int)
    examples = (moved.groupby(["reassigned_from", "cell"], group_keys=False)
                .apply(lambda g: g.sample(min(n_examples, len(g)), random_state=seed),
                       include_groups=False)
                [["file_id", "path", "source_name", "duration_s"]].reset_index(drop=True))
    return table, examples


# --------------------------------------------------------------------------- #


def run_gates(manifest: pd.DataFrame, verdict: pd.DataFrame, folds: pd.DataFrame,
              files: pd.DataFrame, *, min_atoms: int = MIN_ATOMS) -> dict[str, Any]:
    """Every gate and table; ``aggregate`` is the worst verdict."""
    g7, ledger = g_eda7(verdict, files)
    gates = [g_eda3(manifest, min_atoms), g_eda4(folds), g7]
    g4 = g4_table(verdict, files)
    pack, examples = g7_pack(manifest, files)
    verdicts = [g.verdict for g in gates]
    return {"gates": pd.DataFrame([g.as_row() for g in gates]),
            "aggregate": FAIL if FAIL in verdicts else NA if NA in verdicts else PASS,
            "g_eda7_ledger": ledger, "g4_table": g4, "g7_pack": pack,
            "g7_examples": examples}
