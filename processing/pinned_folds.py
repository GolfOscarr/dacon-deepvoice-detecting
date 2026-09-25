"""Extend a built fold table to a grown manifest WITHOUT moving any existing row
(docs/training/11 §1.2).

`training.folds.build_folds` is a global greedy pass: new groups change its order
and tallies, so a rebuild can move a base group to another fold or into or out of
PROBE. That silently breaks (a) initialising fold k from the base run's fold-k
weights, (b) PROBE for models trained on the base corpus, and (c) scoring the new
run on the base run's exact VAL specs. Here every base row keeps its (slice, fold);
only grouping atoms with no base member are placed, and never into PROBE.

Placement of new atoms:
  * artifact families: an explicit `family_fold` map keyed by the family's last path
    element (the cloner), so one cloner is VAL in one fold in every language. A new
    family missing from the map is an error, not a default.
  * a real atom whose speaker prompted a clone: that clone family's fold (first
    family by name when several), so VAL holds the speaker and their clone.
  * other real atoms: water-filled by hours per (lang, pool), starting from the
    base table's VAL hours, largest atom first; ties go to the lowest fold.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from training.folds import FOLD_COLUMNS, grouping_atoms, validate_folds


class PinError(ValueError):
    pass


def extend_folds_pinned(manifest: pd.DataFrame, base: pd.DataFrame,
                        family_fold: Mapping[str, int], n_folds: int,
                        assigned_at: str, scheme_version: str) -> tuple[pd.DataFrame, dict]:
    ids = manifest["file_id"].astype(str)
    base = base.set_index("file_id")
    missing = sorted(set(base.index) - set(ids))
    if missing:
        raise PinError(f"{len(missing)} base row(s) absent from the manifest, e.g. {missing[:3]}")
    if (base["slice"] == "shadow").any():
        raise PinError("base has SHADOW rows; pinning them is not implemented")

    atoms = grouping_atoms(manifest)                     # file_id -> group
    is_base = ids.isin(base.index).to_numpy()
    grp = atoms.loc[ids].to_numpy()

    # 1. atoms holding base rows inherit their (slice, fold) -- and must agree
    b = pd.DataFrame({"group": grp[is_base], "file_id": ids[is_base].to_numpy()})
    b["slice"] = base.loc[b["file_id"], "slice"].to_numpy()
    b["fold"] = base.loc[b["file_id"], "fold"].astype("Int64").to_numpy()
    b["key"] = b["slice"].astype(str) + "/" + b["fold"].astype(str)
    per = b.groupby("group")["key"].nunique()
    bridged = per[per > 1]
    if len(bridged):
        g = bridged.index[0]
        raise PinError(f"{len(bridged)} atom(s) join base rows of different folds/slices, "
                       f"e.g. {g}: {sorted(b.loc[b.group == g, 'key'].unique())}")
    inherit = b.drop_duplicates("group").set_index("group")[["slice", "fold"]]

    # 2. new atoms
    m = manifest.assign(group=grp)
    new = m[~m["group"].isin(inherit.index)]
    fam_rows = new[new["artifact_family"].notna()]
    fam_fold: dict[str, int] = {}
    for g, rows in fam_rows.groupby("group"):
        cloners = {str(f).rsplit("/", 1)[-1] for f in rows["artifact_family"].unique()}
        unknown = sorted(c for c in cloners if c not in family_fold)
        if unknown:
            raise PinError(f"new family cloner(s) {unknown} not in family_fold "
                           f"(atom {g}); place them explicitly")
        folds = {int(family_fold[c]) for c in cloners}
        if len(folds) != 1:
            raise PinError(f"atom {g} spans cloners placed in folds {sorted(folds)}")
        fam_fold[g] = folds.pop()

    # prompt speaker -> fold of the first family (by name) that cloned them
    prompt_fold: dict[str, int] = {}
    pf = fam_rows[fam_rows["prompt_speaker"].notna()]
    for fam, rows in sorted(pf.groupby("artifact_family"), key=lambda t: str(t[0])):
        k = int(family_fold[str(fam).rsplit("/", 1)[-1]])
        for spk in rows["prompt_speaker"].astype(str).unique():
            prompt_fold.setdefault(spk, k)

    real_new = new[~new["group"].isin(fam_fold)]
    hours = real_new.groupby("group")["duration_s"].sum() / 3600
    first = real_new.drop_duplicates("group").set_index("group")
    spk_of = real_new.groupby("group")["speaker_ref_id"].agg(
        lambda s: sorted(set(s.dropna().astype(str))))

    # water-fill tallies start from the base VAL hours per (lang, pool)
    bm = manifest.loc[is_base, ["file_id", "lang", "pool", "duration_s"]].copy()
    bm["fold"] = base.loc[bm["file_id"], "fold"].to_numpy()
    bm = bm[base.loc[bm["file_id"], "slice"].to_numpy() == "train_val"]
    tally: dict[tuple, np.ndarray] = {}
    for (lang, pool, fold), h in (bm.groupby(["lang", "pool", "fold"])["duration_s"].sum()
                                  / 3600).items():
        tally.setdefault((str(lang), str(pool)), np.zeros(n_folds))[int(fold)] += h

    real_fold: dict[str, int] = {}
    by_prompt = 0
    for g in hours.index:
        ks = sorted({prompt_fold[s] for s in spk_of[g] if s in prompt_fold})
        if ks:
            real_fold[g] = ks[0]
            by_prompt += 1
    for g, h in sorted(hours.items(), key=lambda t: (-t[1], t[0])):
        key = (str(first.at[g, "lang"]), str(first.at[g, "pool"]))
        t = tally.setdefault(key, np.zeros(n_folds))
        if g not in real_fold:
            real_fold[g] = int(np.argmin(t))
        t[real_fold[g]] += h

    # the report's VAL hours include the placed families too
    for g, k in fam_fold.items():
        rows = new[new["group"] == g]
        for (lang, pool), h in (rows.groupby(["lang", "pool"])["duration_s"].sum()
                                / 3600).items():
            tally.setdefault((str(lang), str(pool)), np.zeros(n_folds))[k] += h

    # 3. emit, in FOLD_COLUMNS, base rows copied verbatim
    fold_of_group = {**{g: int(f) for g, f in fam_fold.items()}, **real_fold}
    out = pd.DataFrame({"file_id": ids.to_numpy(), "row_kind": manifest["row_kind"].to_numpy()})
    sl = np.empty(len(out), dtype=object)
    fo = pd.array([pd.NA] * len(out), dtype="Int64")
    inh_slice = inherit["slice"].to_dict()
    inh_fold = inherit["fold"].to_dict()
    for i, (fid, g) in enumerate(zip(out["file_id"], grp)):
        if fid in base.index:
            sl[i] = base.at[fid, "slice"]
            fo[i] = base.at[fid, "fold"]
        elif g in inh_slice:
            sl[i] = inh_slice[g]
            fo[i] = inh_fold[g]
        else:
            sl[i] = "train_val"
            fo[i] = fold_of_group[g]
    out["slice"] = sl
    out["shadow_kind"] = None
    out["shadow_of"] = None
    out["fold"] = fo
    for col in ("artifact_family", "source_name", "speaker_ref_id", "pair_id",
                "dup_group", "domain_key"):
        out[col] = manifest[col].to_numpy()
    out["cell"] = manifest["cell"].astype("Int64").to_numpy()
    out["assigned_at"] = assigned_at
    out["scheme_version"] = scheme_version
    out = validate_folds(out[list(FOLD_COLUMNS)])

    # self-checks: base rows verbatim; PROBE is exactly the base PROBE
    o = out.set_index("file_id")
    ob = o.loc[base.index]
    if not (ob["slice"].to_numpy() == base["slice"].to_numpy()).all():
        raise PinError("a base row's slice changed")
    if not ob["fold"].astype("Int64").equals(base["fold"].astype("Int64")):
        raise PinError("a base row's fold changed")
    new_probe = sorted(set(o.index[o["slice"] == "probe"]) - set(base.index))
    placed = set(fold_of_group)
    if any(g in placed for g in atoms.loc[new_probe]):
        raise PinError("a newly placed atom landed in PROBE")
    report = {
        "rows": int(len(out)), "base_rows": int(is_base.sum()),
        "new_rows": int((~is_base).sum()),
        "new_rows_inheriting_base_atom": int((~is_base & np.isin(grp, inherit.index)).sum()),
        "new_rows_in_probe_via_base_atom": len(new_probe),
        "new_family_atoms": {str(g): k for g, k in sorted(fam_fold.items())},
        "new_real_atoms": int(len(real_fold)),
        "new_real_atoms_placed_by_prompt": by_prompt,
        "prompt_speakers": len(prompt_fold),
        "val_hours_after": {f"{k[0]}/{k[1]}": [round(x, 1) for x in v]
                            for k, v in sorted(tally.items())},
    }
    return out, report
