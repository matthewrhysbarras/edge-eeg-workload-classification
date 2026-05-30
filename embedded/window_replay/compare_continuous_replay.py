from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from compare_block_window_replay import _balanced_accuracy, _macro_f1


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "embedded" / "window_replay" / "data" / "continuous_replay_reference.csv"
DEFAULT_MODEL = ROOT / "embedded" / "model_export" / "logreg_replay_model.json"


def _load_reference(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {(int(row["stream_id"]), int(row["window_index"])): row for row in csv.DictReader(f)}


def compare(log_path: Path, reference_path: Path, model_path: Path, out_path: Path | None) -> dict[str, object]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    labels = [str(x) for x in model["metric_labels"]]
    reference = _load_reference(reference_path)

    y_true: list[str] = []
    y_pred: list[str] = []
    mismatches: list[dict[str, object]] = []
    feature_us: list[float] = []
    classifier_us: list[float] = []
    total_us: list[float] = []
    malformed: list[int] = []
    dropped: list[int] = []
    overruns: list[int] = []
    seen: set[tuple[int, int]] = set()

    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("stream_id"):
                continue
            key = (int(row["stream_id"]), int(row["window_index"]))
            ref = reference.get(key)
            if ref is None:
                continue
            seen.add(key)
            true_label = ref["true_label"]
            pred_label = row["pred_label"]
            expected = ref["python_pred_from_window_features"]
            y_true.append(true_label)
            y_pred.append(pred_label)
            if pred_label != expected:
                mismatches.append(
                    {
                        "stream_id": key[0],
                        "window_index": key[1],
                        "replay_row_index": int(ref["replay_row_index"]),
                        "expected": expected,
                        "actual": pred_label,
                        "true_label": true_label,
                    }
                )
            feature_us.append(float(row["feature_us"]))
            classifier_us.append(float(row["classifier_us"]))
            total_us.append(float(row["total_us"]))
            malformed.append(int(row["malformed_packets"]))
            dropped.append(int(row["dropped_sequences"]))
            overruns.append(int(row["buffer_overrun_count"]))

    missing = sorted(set(reference) - seen)
    rows = len(y_pred)
    summary: dict[str, object] = {
        "log_path": str(log_path),
        "reference_path": str(reference_path),
        "windows_processed": rows,
        "windows_expected": len(reference),
        "missing_windows": [{"stream_id": x[0], "window_index": x[1]} for x in missing[:20]],
        "prediction_agreement": (rows - len(mismatches)) / rows if rows else None,
        "prediction_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels) if rows else None,
        "macro_f1": _macro_f1(y_true, y_pred, labels) if rows else None,
        "mean_feature_us": sum(feature_us) / len(feature_us) if feature_us else None,
        "max_feature_us": max(feature_us) if feature_us else None,
        "mean_classifier_us": sum(classifier_us) / len(classifier_us) if classifier_us else None,
        "max_classifier_us": max(classifier_us) if classifier_us else None,
        "mean_total_us": sum(total_us) / len(total_us) if total_us else None,
        "max_total_us": max(total_us) if total_us else None,
        "max_malformed_packets": max(malformed) if malformed else None,
        "max_dropped_sequences": max(dropped) if dropped else None,
        "max_buffer_overrun_count": max(overruns) if overruns else None,
        "safe_below_1_5s_hop": max(total_us) < 1_500_000 if total_us else None,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ESP32 continuous sample replay output against Python references.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = compare(
        args.log.resolve(),
        args.reference.resolve(),
        args.model.resolve(),
        args.out.resolve() if args.out else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
