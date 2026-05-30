from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "embedded" / "model_export" / "logreg_replay_model.json"
DEFAULT_VECTORS = ROOT / "embedded" / "model_export" / "logreg_replay_vectors.csv"
DEFAULT_EXPECTED = ROOT / "embedded" / "model_export" / "logreg_replay_reference_predictions.csv"


def _to_float(text: str) -> float:
    value = str(text).strip()
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _predict_one(model_package: dict[str, Any], model_index: int, raw_features: list[float]) -> dict[str, Any]:
    model = model_package["models"][model_index]
    x: list[float] = []
    for i, value in enumerate(raw_features):
        if not math.isfinite(value):
            value = float(model["imputer_median"][i])
        lo = model["clip_lower"][i]
        hi = model["clip_upper"][i]
        if lo is not None and math.isfinite(float(lo)) and value < float(lo):
            value = float(lo)
        if hi is not None and math.isfinite(float(hi)) and value > float(hi):
            value = float(hi)
        scale = float(model["scaler_scale"][i])
        if not math.isfinite(scale) or scale == 0.0:
            scale = 1.0
        x.append((value - float(model["scaler_center"][i])) / scale)

    active = model["active_feature_indices"]
    logits: list[float] = []
    for class_i, intercept in enumerate(model["intercept"]):
        total = float(intercept)
        coef = model["coef"][class_i]
        for coef_i, feature_i in enumerate(active):
            total += float(coef[coef_i]) * x[int(feature_i)]
        logits.append(total)

    max_logit = max(logits)
    exps = [math.exp(v - max_logit) for v in logits]
    denom = sum(exps)
    probs = [v / denom for v in exps]
    pred_i = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "pred_label": model["model_classes"][pred_i],
        "logits": logits,
        "probabilities": probs,
    }


def _balanced_accuracy(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    recalls: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return sum(recalls) / len(recalls)


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(scores) / len(scores)


def _load_expected(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row["replay_row_index"]): row["pred_label"] for row in csv.DictReader(f)}


def run(model_path: Path, vectors_path: Path, expected_path: Path, out_path: Path | None) -> dict[str, Any]:
    model_package = json.loads(model_path.read_text(encoding="utf-8"))
    feature_names = [str(x) for x in model_package["feature_names"]]
    labels = [str(x) for x in model_package["metric_labels"]]
    expected = _load_expected(expected_path)

    rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    by_model: dict[str, dict[str, list[str]]] = {}
    agreement_hits = 0
    agreement_total = 0
    elapsed_us: list[float] = []

    with vectors_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = [_to_float(row[name]) for name in feature_names]
            start = time.perf_counter_ns()
            pred = _predict_one(model_package, int(row["model_index"]), raw)
            elapsed_us.append((time.perf_counter_ns() - start) / 1000.0)
            replay_row_index = int(row["replay_row_index"])
            true_label = str(row["true_label"])
            pred_label = str(pred["pred_label"])
            y_true.append(true_label)
            y_pred.append(pred_label)
            model_key = str(row["model_index"])
            by_model.setdefault(model_key, {"true": [], "pred": []})
            by_model[model_key]["true"].append(true_label)
            by_model[model_key]["pred"].append(pred_label)
            if replay_row_index in expected:
                agreement_total += 1
                agreement_hits += int(expected[replay_row_index] == pred_label)
            rows.append(
                {
                    "replay_row_index": replay_row_index,
                    "dataset_row_index": row["dataset_row_index"],
                    "model_index": row["model_index"],
                    "participant_id": row["participant_id"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "logit_0": f"{pred['logits'][0]:.9g}",
                    "logit_1": f"{pred['logits'][1]:.9g}",
                    "logit_2": f"{pred['logits'][2]:.9g}",
                    "prob_0": f"{pred['probabilities'][0]:.9g}",
                    "prob_1": f"{pred['probabilities'][1]:.9g}",
                    "prob_2": f"{pred['probabilities'][2]:.9g}",
                }
            )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    split_bacc = [_balanced_accuracy(v["true"], v["pred"], labels) for v in by_model.values()]
    split_f1 = [_macro_f1(v["true"], v["pred"], labels) for v in by_model.values()]
    summary = {
        "rows": len(rows),
        "class_counts": dict(Counter(y_true)),
        "pooled_replay_balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels),
        "pooled_replay_macro_f1": _macro_f1(y_true, y_pred, labels),
        "mean_split_balanced_accuracy": sum(split_bacc) / len(split_bacc),
        "mean_split_macro_f1": sum(split_f1) / len(split_f1),
        "agreement_with_exported_reference": agreement_hits / agreement_total if agreement_total else None,
        "agreement_rows": agreement_total,
        "mean_inference_us_python": sum(elapsed_us) / len(elapsed_us),
        "max_inference_us_python": max(elapsed_us),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="No-scikit reference inference for the exported EEG logistic regression replay model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--out", type=Path, default=ROOT / "embedded" / "model_export" / "python_reference_predictions.csv")
    args = parser.parse_args()
    summary = run(args.model.resolve(), args.vectors.resolve(), args.expected.resolve(), args.out.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
