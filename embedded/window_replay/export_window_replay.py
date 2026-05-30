from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from window_feature_reference import compute_window_features, feature_order, logits_and_prediction


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "embedded" / "window_replay" / "data"
FEATURE_TABLE = ROOT / "replication" / "runs" / "eeg_logreg_overlap_low_high" / "features" / "features_ml_eeg_target_eeg_overlap3s_50pct_preproc.tsv"
EPOCH_MANIFEST = ROOT / "replication" / "runs" / "eeg_logreg_overlap_low_high" / "reports" / "epoch_manifest.tsv"
REPLAY_VECTORS = ROOT / "embedded" / "model_export" / "logreg_replay_vectors.csv"
MODEL_JSON = ROOT / "embedded" / "model_export" / "logreg_replay_model.json"
REFERENCE_PREDICTIONS = ROOT / "embedded" / "model_export" / "logreg_replay_reference_predictions.csv"

MAGIC = 0x31574245  # E B W 1, little-endian bytes: EWB1
N_CHANNELS = 19
N_SAMPLES = 750
N_FEATURES = 153
N_BASELINE = 15
ROI_ORDER = ["central", "frontal", "occipital", "parietal", "temporal"]
BASELINE_BANDS = ["theta", "alpha", "beta"]


def _label_index(labels: list[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError:
        return 65535


def _baseline_from_feature_row(row: pd.Series) -> dict[str, dict[str, float]]:
    baseline: dict[str, dict[str, float]] = {}
    for roi in ROI_ORDER:
        baseline[roi] = {}
        for band in BASELINE_BANDS:
            abs_key = f"eeg_abs_{band}_{roi}"
            delta_key = f"eeg_abs_{band}_{roi}_delta_base"
            current = float(row[abs_key])
            delta = float(row[delta_key])
            baseline[roi][band] = current - delta
    return baseline


def _baseline_vector(baseline: dict[str, dict[str, float]]) -> list[float]:
    return [float(baseline[roi][band]) for roi in ROI_ORDER for band in BASELINE_BANDS]


def _load_ch_names(participant_id: str) -> list[str]:
    meta = (
        ROOT
        / "replication"
        / "runs"
        / "eeg_logreg_overlap_low_high"
        / "derivatives"
        / "epochs"
        / participant_id
        / "eeg"
        / f"{participant_id}_task-arithmetic_desc-epochs-eeg_meta.json"
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return [str(x) for x in payload["ch_names"]]


def _metric(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float]:
    recalls = []
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(recalls) / len(recalls), sum(f1s) / len(f1s)


def export(out_dir: Path, limit: int | None, include_all: bool) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    labels = [str(x) for x in model["metric_labels"]]
    order = feature_order(MODEL_JSON)
    feature_df = pd.read_csv(FEATURE_TABLE, sep="\t")
    manifest = pd.read_csv(EPOCH_MANIFEST, sep="\t")
    replay_df = pd.read_csv(REPLAY_VECTORS)
    expected_df = pd.read_csv(REFERENCE_PREDICTIONS)
    expected_by_row = dict(zip(expected_df["replay_row_index"].astype(int), expected_df["pred_label"].astype(str)))

    if include_all:
        selected = replay_df
    else:
        # Representative default: first two held-out windows per model.
        selected = replay_df.groupby("model_index", group_keys=False).head(2)
    if limit is not None:
        selected = selected.head(max(0, int(limit)))

    feature_by_ml = feature_df.set_index("ml_row_id", drop=False)
    manifest_by_epoch = manifest.set_index(["participant_id", "epoch_id"], drop=False)
    rows: list[dict[str, object]] = []
    packet_path = out_dir / "window_replay_blocks.bin"
    ref_csv = out_dir / "window_replay_reference.csv"
    meta_json = out_dir / "window_replay_metadata.json"

    y_true: list[str] = []
    y_pred_ref: list[str] = []
    y_pred_features: list[str] = []
    max_feature_abs_diff_vs_table = 0.0

    with packet_path.open("wb") as packet:
        for out_i, replay_row in selected.reset_index(drop=True).iterrows():
            ml_row_id = str(replay_row["ml_row_id"])
            frow = feature_by_ml.loc[ml_row_id]
            participant_id = str(frow["participant_id"])
            epoch_id = str(frow["epoch_id"])
            mrow = manifest_by_epoch.loc[(participant_id, epoch_id)]
            npz_path = Path(str(mrow["eeg_epoch_file"]))
            npz = np.load(npz_path)
            data = npz["data"].astype(np.float32, copy=False)
            time = npz["time"]
            if data.shape != (N_CHANNELS, N_SAMPLES):
                raise ValueError(f"Unexpected shape for {npz_path}: {data.shape}")
            sfreq = 1.0 / float(np.median(np.diff(time)))
            ch_names = _load_ch_names(participant_id)
            baseline = _baseline_from_feature_row(frow)
            features = compute_window_features(data, sfreq, ch_names, baseline, order)
            feature_values = [float(features[name]) for name in order]
            table_values = [float(frow[name]) for name in order]
            max_feature_abs_diff_vs_table = max(
                max_feature_abs_diff_vs_table,
                max(abs(a - b) for a, b in zip(feature_values, table_values) if math.isfinite(a) and math.isfinite(b)),
            )
            pred = logits_and_prediction(model, int(replay_row["model_index"]), feature_values)
            true_label = str(replay_row["true_label"])
            expected_pred = expected_by_row[int(replay_row["replay_row_index"])]
            y_true.append(true_label)
            y_pred_ref.append(expected_pred)
            y_pred_features.append(str(pred["pred_label"]))

            header = struct.pack(
                "<IIHHd",
                MAGIC,
                int(replay_row["replay_row_index"]),
                int(replay_row["model_index"]),
                _label_index(labels, true_label),
                float(sfreq),
            )
            packet.write(header)
            packet.write(np.asarray(_baseline_vector(baseline), dtype="<f4").tobytes())
            packet.write(np.asarray(feature_values, dtype="<f4").tobytes())
            packet.write(np.asarray(data, dtype="<f4").tobytes(order="C"))

            rows.append(
                {
                    "packet_index": out_i,
                    "replay_row_index": int(replay_row["replay_row_index"]),
                    "model_index": int(replay_row["model_index"]),
                    "participant_id": participant_id,
                    "epoch_id": epoch_id,
                    "true_label": true_label,
                    "python_pred_from_exported_features": expected_pred,
                    "python_pred_from_window_features": str(pred["pred_label"]),
                    "npz_path": str(npz_path),
                }
            )

    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bacc_ref, f1_ref = _metric(y_true, y_pred_ref, labels)
    bacc_features, f1_features = _metric(y_true, y_pred_features, labels)
    summary = {
        "packet_bin": str(packet_path),
        "reference_csv": str(ref_csv),
        "n_windows": len(rows),
        "n_channels": N_CHANNELS,
        "n_samples": N_SAMPLES,
        "sfreq_hz": 250.0,
        "n_features": N_FEATURES,
        "packet_bytes_per_window": 4 + 4 + 2 + 2 + 8 + N_BASELINE * 4 + N_FEATURES * 4 + N_CHANNELS * N_SAMPLES * 4,
        "max_feature_abs_diff_vs_stage4_table": max_feature_abs_diff_vs_table,
        "python_window_feature_prediction_agreement_with_feature_row_reference": sum(a == b for a, b in zip(y_pred_features, y_pred_ref)) / len(y_pred_ref),
        "balanced_accuracy_from_feature_row_reference": bacc_ref,
        "macro_f1_from_feature_row_reference": f1_ref,
        "balanced_accuracy_from_window_features": bacc_features,
        "macro_f1_from_window_features": f1_features,
        "labels": labels,
    }
    meta_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sketch_dir = ROOT / "embedded" / "window_replay" / "esp32_window_replay"
    shutil.copy2(ROOT / "embedded" / "model_export" / "model_data.h", sketch_dir / "model_data.h")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preprocessed EEG windows for ESP32 block-window replay.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Export all 476 held-out replay windows instead of the representative subset.")
    args = parser.parse_args()
    summary = export(args.out_dir.resolve(), args.limit, args.all)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
