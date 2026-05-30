from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_stage_summary_matches_paper_claims() -> None:
    rows = _read_json("embedded/paper_validation/embedded_stage_summary.json")
    by_stage = {row["stage"]: row for row in rows}

    feature = by_stage["feature_vector_replay"]
    block = by_stage["block_window_replay"]
    continuous = by_stage["continuous_timed_sample_replay"]

    assert feature["rows_or_windows"] == 476
    assert feature["agreement"] == 1.0
    assert round(feature["balanced_accuracy"], 4) == 0.7115
    assert round(feature["macro_f1"], 4) == 0.7004

    assert block["rows_or_windows"] == 476
    assert block["agreement"] == 1.0
    assert round(block["balanced_accuracy"], 4) == 0.7115
    assert round(block["macro_f1"], 4) == 0.7004

    assert continuous["rows_or_windows"] == 50
    assert continuous["agreement"] == 1.0
    assert round(continuous["balanced_accuracy"], 4) == 0.6405
    assert round(continuous["macro_f1"], 4) == 0.6173


def test_block_window_confusion_matrix_sums_to_full_heldout_set() -> None:
    path = ROOT / "embedded/paper_validation/block_window_476_confusion_matrix.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    total = 0
    labels = ["baseline", "low_1_2_3", "high_4_5_6"]
    for row in rows:
        total += sum(int(row[label]) for label in labels)
    assert total == 476


def test_repeatability_timing_is_within_hop_budget() -> None:
    summary = _read_json("embedded/paper_validation/stage2_repeatability_summary.json")
    assert summary["prediction_agreement_min"] == 1.0
    assert summary["prediction_mismatches_total"] == 0
    assert summary["malformed_packets_max"] == 0
    assert summary["dropped_sequences_max"] == 0
    assert summary["buffer_overruns_max"] == 0
    assert summary["total_us_max"] < summary["hop_budget_us"]


def test_no_private_absolute_paths_in_release_metadata() -> None:
    forbidden = ("C:" + "\\\\" + "Users", "Documents" + "\\\\" + "Conference", "REPLACE" + "_WITH_REPOSITORY")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or path.is_dir():
            continue
        if path.suffix.lower() not in {".md", ".json", ".csv", ".py", ".yml", ".yaml", ".cff", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text for token in forbidden), path
