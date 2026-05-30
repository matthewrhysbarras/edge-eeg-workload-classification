from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from reference_inference import _balanced_accuracy, _macro_f1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "embedded" / "model_export" / "logreg_replay_model.json"
DEFAULT_VECTORS = ROOT / "embedded" / "model_export" / "logreg_replay_vectors.csv"
DEFAULT_REFERENCE = ROOT / "embedded" / "model_export" / "logreg_replay_reference_predictions.csv"


def _load_vectors(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row["replay_row_index"]): row for row in csv.DictReader(f)}


def _load_reference(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row["replay_row_index"]): row["pred_label"] for row in csv.DictReader(f)}


def compare(log_path: Path, model_path: Path, vectors_path: Path, reference_path: Path, out_path: Path | None) -> dict[str, object]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    labels = [str(x) for x in model["metric_labels"]]
    vectors = _load_vectors(vectors_path)
    reference = _load_reference(reference_path)

    y_true: list[str] = []
    y_pred: list[str] = []
    by_model: dict[str, dict[str, list[str]]] = {}
    agreement_hits = 0
    agreement_total = 0
    timings: list[float] = []
    missing_rows: list[int] = []

    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            replay_row_index = int(row["replay_row_index"])
            if replay_row_index not in vectors:
                missing_rows.append(replay_row_index)
                continue
            true_label = vectors[replay_row_index]["true_label"]
            pred_label = row["pred_label"]
            y_true.append(true_label)
            y_pred.append(pred_label)
            model_key = str(row["model_index"])
            by_model.setdefault(model_key, {"true": [], "pred": []})
            by_model[model_key]["true"].append(true_label)
            by_model[model_key]["pred"].append(pred_label)
            if replay_row_index in reference:
                agreement_total += 1
                agreement_hits += int(reference[replay_row_index] == pred_label)
            try:
                timing_value = row.get("inference_us", "")
                if timing_value != "":
                    timings.append(float(timing_value))
            except (TypeError, ValueError):
                pass

    split_bacc = [_balanced_accuracy(v["true"], v["pred"], labels) for v in by_model.values()]
    split_f1 = [_macro_f1(v["true"], v["pred"], labels) for v in by_model.values()]
    summary: dict[str, object] = {
        "log_path": str(log_path),
        "rows": len(y_true),
        "missing_or_unknown_rows": missing_rows[:20],
        "class_counts": dict(Counter(y_true)),
        "agreement_with_python_reference": agreement_hits / agreement_total if agreement_total else None,
        "agreement_rows": agreement_total,
        "pooled_replay_balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels) if y_true else None,
        "pooled_replay_macro_f1": _macro_f1(y_true, y_pred, labels) if y_true else None,
        "mean_split_balanced_accuracy": sum(split_bacc) / len(split_bacc) if split_bacc else None,
        "mean_split_macro_f1": sum(split_f1) / len(split_f1) if split_f1 else None,
        "mean_inference_us_esp32": sum(timings) / len(timings) if timings else None,
        "max_inference_us_esp32": max(timings) if timings else None,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ESP32 serial replay logs with the Python reference predictions.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = compare(args.log.resolve(), args.model.resolve(), args.vectors.resolve(), args.reference.resolve(), args.out.resolve() if args.out else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
