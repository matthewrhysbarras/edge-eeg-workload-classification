from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "embedded" / "window_replay" / "data" / "window_replay_reference.csv"
DEFAULT_MODEL = ROOT / "embedded" / "model_export" / "logreg_replay_model.json"


def _balanced_accuracy(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    recalls = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return sum(recalls) / len(recalls)


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(scores) / len(scores)


def _load_reference(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row["replay_row_index"]): row for row in csv.DictReader(f)}


def compare(log_path: Path, reference_path: Path, model_path: Path, out_path: Path | None) -> dict[str, object]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    labels = [str(x) for x in model["metric_labels"]]
    reference = _load_reference(reference_path)

    y_true: list[str] = []
    y_pred: list[str] = []
    by_model: dict[str, dict[str, list[str]]] = {}
    agreement = 0
    agreement_window_feature_ref = 0
    mismatches_feature_ref: list[dict[str, object]] = []
    mismatches_window_ref: list[dict[str, object]] = []
    rows = 0
    max_errors: list[float] = []
    max_error_indices: list[int] = []
    mean_errors: list[float] = []
    rmse_errors: list[float] = []
    feature_us: list[float] = []
    classifier_us: list[float] = []
    total_us: list[float] = []
    missing_rows: list[int] = []

    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("replay_row_index"):
                continue
            replay_row_index = int(row["replay_row_index"])
            ref = reference.get(replay_row_index)
            if ref is None:
                missing_rows.append(replay_row_index)
                continue
            rows += 1
            true_label = ref["true_label"]
            pred_label = row["pred_label"]
            model_index = str(ref["model_index"])
            y_true.append(true_label)
            y_pred.append(pred_label)
            by_model.setdefault(model_index, {"true": [], "pred": []})
            by_model[model_index]["true"].append(true_label)
            by_model[model_index]["pred"].append(pred_label)
            feature_ref_match = pred_label == ref["python_pred_from_exported_features"]
            window_ref_match = pred_label == ref["python_pred_from_window_features"]
            agreement += int(feature_ref_match)
            agreement_window_feature_ref += int(window_ref_match)
            if not feature_ref_match:
                mismatches_feature_ref.append(
                    {
                        "replay_row_index": replay_row_index,
                        "model_index": int(model_index),
                        "true_label": true_label,
                        "expected": ref["python_pred_from_exported_features"],
                        "actual": pred_label,
                    }
                )
            if not window_ref_match:
                mismatches_window_ref.append(
                    {
                        "replay_row_index": replay_row_index,
                        "model_index": int(model_index),
                        "true_label": true_label,
                        "expected": ref["python_pred_from_window_features"],
                        "actual": pred_label,
                    }
                )
            max_errors.append(float(row["max_abs_feature_error"]))
            max_error_indices.append(int(row["max_feature_error_index"]))
            mean_errors.append(float(row["mean_abs_feature_error"]))
            rmse_errors.append(float(row["rmse_feature_error"]))
            feature_us.append(float(row["feature_us"]))
            classifier_us.append(float(row["classifier_us"]))
            total_us.append(float(row["total_us"]))

    split_bacc = [_balanced_accuracy(v["true"], v["pred"], labels) for v in by_model.values()]
    split_f1 = [_macro_f1(v["true"], v["pred"], labels) for v in by_model.values()]

    summary: dict[str, object] = {
        "log_path": str(log_path),
        "reference_path": str(reference_path),
        "rows": rows,
        "missing_rows": missing_rows[:20],
        "class_counts": dict(Counter(y_true)),
        "agreement_with_feature_row_python_reference": agreement / rows if rows else None,
        "agreement_with_window_feature_python_reference": agreement_window_feature_ref / rows if rows else None,
        "prediction_mismatches_vs_feature_row_reference": len(mismatches_feature_ref),
        "prediction_mismatches_vs_window_feature_reference": len(mismatches_window_ref),
        "mismatch_examples_vs_feature_row_reference": mismatches_feature_ref[:20],
        "mismatch_examples_vs_window_feature_reference": mismatches_window_ref[:20],
        "pooled_balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels) if rows else None,
        "pooled_macro_f1": _macro_f1(y_true, y_pred, labels) if rows else None,
        "mean_split_balanced_accuracy": sum(split_bacc) / len(split_bacc) if split_bacc else None,
        "mean_split_macro_f1": sum(split_f1) / len(split_f1) if split_f1 else None,
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels) if rows else None,
        "macro_f1": _macro_f1(y_true, y_pred, labels) if rows else None,
        "max_feature_abs_error": max(max_errors) if max_errors else None,
        "max_feature_error_indices": sorted(set(max_error_indices)),
        "mean_feature_abs_error": sum(mean_errors) / len(mean_errors) if mean_errors else None,
        "mean_feature_rmse": sum(rmse_errors) / len(rmse_errors) if rmse_errors else None,
        "mean_feature_us": sum(feature_us) / len(feature_us) if feature_us else None,
        "max_feature_us": max(feature_us) if feature_us else None,
        "mean_classifier_us": sum(classifier_us) / len(classifier_us) if classifier_us else None,
        "max_classifier_us": max(classifier_us) if classifier_us else None,
        "mean_total_us": sum(total_us) / len(total_us) if total_us else None,
        "max_total_us": max(total_us) if total_us else None,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ESP32 block-window replay output against Python references.")
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
