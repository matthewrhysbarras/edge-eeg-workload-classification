from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STAGE4_DIR = ROOT / "_reference_Arithmetic_Workload_Estimation" / "analysis_pipeline"
if str(STAGE4_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE4_DIR))

from stage4_extract_features import _compute_eeg_roi_features  # noqa: E402


ROI_ORDER = ["central", "frontal", "occipital", "parietal", "temporal"]
BASELINE_DELTA_BANDS = ("theta", "alpha", "beta")


def feature_order(model_path: Path = ROOT / "embedded" / "model_export" / "logreg_replay_model.json") -> list[str]:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    return [str(x) for x in payload["feature_names"]]


def compute_window_features(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    baseline_abs: dict[str, dict[str, float]],
    order: list[str] | None = None,
    round_decimals: int | None = 9,
) -> dict[str, float]:
    feat, roi_abs = _compute_eeg_roi_features(data.astype(np.float64), float(sfreq), ch_names)
    for roi, bands in baseline_abs.items():
        roi_current = roi_abs.get(roi, {})
        for band in BASELINE_DELTA_BANDS:
            cur = roi_current.get(band)
            base = bands.get(band)
            if cur is not None and base is not None and math.isfinite(cur) and math.isfinite(base):
                feat[f"eeg_abs_{band}_{roi}_delta_base"] = float(cur - base)
    def clean(value: float) -> float:
        out = float(value)
        if round_decimals is not None and math.isfinite(out):
            out = round(out, int(round_decimals))
        return out

    if order is None:
        return {k: clean(v) for k, v in feat.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
    return {name: clean(feat.get(name, math.nan)) for name in order}


def logits_and_prediction(model_package: dict[str, Any], model_index: int, features: list[float]) -> dict[str, Any]:
    model = model_package["models"][model_index]
    x: list[float] = []
    for i, value in enumerate(features):
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

    logits: list[float] = []
    for class_i, intercept in enumerate(model["intercept"]):
        total = float(intercept)
        coef = model["coef"][class_i]
        for coef_i, feature_i in enumerate(model["active_feature_indices"]):
            total += float(coef[coef_i]) * x[int(feature_i)]
        logits.append(total)
    max_logit = max(logits)
    exp_vals = [math.exp(v - max_logit) for v in logits]
    denom = sum(exp_vals)
    probs = [v / denom for v in exp_vals]
    pred_i = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "pred_label": model["model_classes"][pred_i],
        "logits": logits,
        "probabilities": probs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the Python reference 153 EEG features for one exported npz window.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--ch-names-json", type=Path, required=True)
    args = parser.parse_args()
    payload = np.load(args.npz)
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    ch_names = json.loads(args.ch_names_json.read_text(encoding="utf-8"))
    data = payload["data"]
    time = payload["time"]
    sfreq = 1.0 / float(np.median(np.diff(time)))
    names = feature_order()
    values = compute_window_features(data, sfreq, ch_names, baseline, names)
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
