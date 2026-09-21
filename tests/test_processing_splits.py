"""OFF-5 as built: the folds over a manifest, checked and written."""

import json

import pytest

from processing.config import ProcessingConfig, load_processing_config
from processing.splits import build_and_check, write_outputs
from training.folds import FoldConfig
from training.synthetic import synthetic_manifest


@pytest.fixture(scope="module")
def manifest():
    return synthetic_manifest(n_per_pool=240, n_whole_file=200, seed=0, n_families=24, n_sources=8)


def test_build_and_check_returns_a_vg1_clean_table_and_a_summary(manifest, tmp_path):
    plan, report, summary = build_and_check(manifest, FoldConfig(assigned_at="x"))
    assert report.ok
    assert summary["n_folds"] == 5 and set(summary["rows_per_fold"]) == set(range(5))
    assert summary["rows"]["probe"] > 0 and 0 < summary["probe_row_share"] < 0.5
    write_outputs(tmp_path, plan, report, summary)
    assert (tmp_path / "folds.parquet").exists()
    assert json.loads((tmp_path / "folds_report.json").read_text())["n_folds"] == 5
    assert "PASS" in (tmp_path / "folds.vg1.txt").read_text()


def test_the_v1_folds_section_is_four_folds_with_probe():
    cfg = load_processing_config("configs/processing_v1.yaml").folds
    assert cfg == ProcessingConfig().folds
    assert cfg.n_folds == 4 and cfg.probe_share == 0.10 and not cfg.allow_no_probe
