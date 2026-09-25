import pandas as pd
import pytest

from processing.pinned_folds import PinError, extend_folds_pinned
from training.folds import FOLD_COLUMNS

FF = {"fishspeech": 0, "maskgct": 1}


def _row(fid, pool, lang, *, fam=None, src=None, spk=None, prompt=None, dur=3600.0):
    return {"file_id": fid, "row_kind": "component", "pool": pool, "lang": lang,
            "artifact_family": fam, "source_name": src or fam or fid, "speaker_ref_id": spk,
            "pair_id": None, "dup_group": None, "cell": None, "domain_key": None,
            "duration_s": dur, "prompt_speaker": prompt, "scheme_version": "t"}


def _base(rows, where):
    m = pd.DataFrame(rows)
    b = pd.DataFrame({"file_id": m.file_id, "row_kind": m.row_kind})
    b["slice"] = [("probe" if where[f] is None else "train_val") for f in m.file_id]
    b["shadow_kind"] = None
    b["shadow_of"] = None
    b["fold"] = pd.array([where[f] for f in m.file_id], dtype="Int64")
    for c in ("artifact_family", "source_name", "speaker_ref_id", "pair_id", "dup_group",
              "domain_key"):
        b[c] = m[c]
    b["cell"] = pd.array([pd.NA] * len(m), dtype="Int64")
    b["assigned_at"] = "x"
    b["scheme_version"] = "t"
    return b[list(FOLD_COLUMNS)]


BASE_ROWS = [_row("r0", "A", "ko", spk="s0"), _row("r1", "A", "ko", spk="s1"),
             _row("f0", "B", "ko", fam="old/x", spk="s0"), _row("p0", "A", "en", spk="sp")]
WHERE = {"r0": 0, "r1": 1, "f0": 0, "p0": None}


def _run(new_rows, ff=FF):
    m = pd.DataFrame(BASE_ROWS + new_rows)
    return extend_folds_pinned(m, _base(BASE_ROWS, WHERE), ff, 2, "now", "t")


def test_base_rows_verbatim_and_inheritance():
    out, rep = _run([_row("proc:r1", "A", "ko", src="enc/r1", spk="s1"),
                     _row("proc:p0", "A", "en", src="enc/p0", spk="sp")])
    o = out.set_index("file_id")
    assert o.at["r0", "fold"] == 0 and o.at["r1", "fold"] == 1 and o.at["p0", "slice"] == "probe"
    assert o.at["proc:r1", "fold"] == 1                    # joins its source's atom
    assert o.at["proc:p0", "slice"] == "probe"             # a processed PROBE file stays sealed
    assert rep["new_rows_in_probe_via_base_atom"] == 1


def test_family_map_and_prompt_speaker_follow():
    out, rep = _run([_row("e1", "A", "ko", spk="emilia/X"),
                     _row("c1", "B", "ko", fam="ko-synth2/maskgct", spk="ko-synth2/maskgct/emilia/X",
                          prompt="emilia/X"),
                     _row("c2", "B", "en", fam="en-synth2/maskgct", spk="en-synth2/maskgct/Y")])
    o = out.set_index("file_id")
    assert o.at["c1", "fold"] == 1 and o.at["c2", "fold"] == 1
    assert o.at["e1", "fold"] == 1 and rep["new_real_atoms_placed_by_prompt"] == 1


def test_waterfill_goes_to_the_emptier_fold():
    out, _ = _run([_row("e2", "A", "en", spk="emilia/Z")])
    # base en-real VAL hours: fold 0 = 0 (p0 is PROBE), fold 1 = 0 -> tie -> fold 0
    assert out.set_index("file_id").at["e2", "fold"] == 0
    out, _ = _run([_row("e3", "A", "ko", spk="emilia/W")])
    # ko real: fold 0 has r0 (1 h), fold 1 has r1 (1 h) -> tie -> 0; then check fill
    assert out.set_index("file_id").at["e3", "fold"] == 0


def test_bridge_aborts():
    with pytest.raises(PinError, match="different folds"):
        _run([_row("x", "A", "ko", src="bridge", spk="s0"),
              _row("y", "A", "ko", src="bridge", spk="s1")])


def test_unknown_cloner_aborts():
    with pytest.raises(PinError, match="not in family_fold"):
        _run([_row("c9", "B", "ko", fam="ko-synth2/newtts", spk="ko-synth2/newtts/Q")])


def test_new_family_merged_into_base_atom_aborts():
    # a new family row that shares a speaker key with a base atom would inherit its fold
    with pytest.raises(PinError, match="must not inherit"):
        _run([_row("c5", "B", "ko", fam="ko-synth2/maskgct", spk="s0")])
