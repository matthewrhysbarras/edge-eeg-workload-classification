from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from export_window_replay import (
    BASELINE_BANDS,
    EPOCH_MANIFEST,
    FEATURE_TABLE,
    MODEL_JSON,
    REPLAY_VECTORS,
    ROI_ORDER,
    ROOT,
    _baseline_from_feature_row,
    _baseline_vector,
    _load_ch_names,
    _metric,
)
from window_feature_reference import compute_window_features, feature_order, logits_and_prediction


DEFAULT_OUT = ROOT / "embedded" / "window_replay" / "data"
STREAM_MAGIC = 0x32535745  # bytes "EWS2"
SAMPLE_MAGIC = 0x32535753  # bytes "SWS2"
N_CHANNELS = 19
N_SAMPLES = 750
HOP_SAMPLES = 375
N_BASELINE = len(ROI_ORDER) * len(BASELINE_BANDS)


def _parse_epoch_id(epoch_id: str) -> tuple[str, int, int]:
    match = re.match(r"(sub-\d+)_trial-(\d+)_seg-(\d+)", str(epoch_id))
    if not match:
        raise ValueError(f"Could not parse epoch_id {epoch_id!r}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def _load_window(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(npz_path)
    data = payload["data"].astype(np.float32, copy=False)
    time = payload["time"]
    if data.shape != (N_CHANNELS, N_SAMPLES):
        raise ValueError(f"Unexpected EEG window shape for {npz_path}: {data.shape}")
    return data, time


def _contiguous_runs(merged: pd.DataFrame) -> list[pd.DataFrame]:
    runs: list[pd.DataFrame] = []
    for _, group in merged.groupby(["model_index", "participant_id", "trial"]):
        group = group.sort_values("seg")
        current: list[pd.Series] = []
        prev_seg: int | None = None
        for _, row in group.iterrows():
            seg = int(row["seg"])
            if prev_seg is None or seg == prev_seg + 1:
                current.append(row)
            else:
                if current:
                    runs.append(pd.DataFrame(current))
                current = [row]
            prev_seg = seg
        if current:
            runs.append(pd.DataFrame(current))
    return sorted(runs, key=lambda df: (int(df["model_index"].iloc[0]), str(df["participant_id"].iloc[0]), int(df["trial"].iloc[0]), int(df["seg"].iloc[0])))


def export(out_dir: Path, limit_windows: int, min_run_windows: int) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    labels = [str(x) for x in model["metric_labels"]]
    order = feature_order(MODEL_JSON)
    replay_df = pd.read_csv(REPLAY_VECTORS)
    feature_df = pd.read_csv(FEATURE_TABLE, sep="\t")
    manifest = pd.read_csv(EPOCH_MANIFEST, sep="\t")
    manifest_by_epoch = manifest.set_index(["participant_id", "epoch_id"], drop=False)
    feature_by_ml = feature_df.set_index("ml_row_id", drop=False)

    selected = replay_df[["replay_row_index", "model_index", "ml_row_id", "true_label"]].merge(
        feature_df[["ml_row_id", "participant_id", "epoch_id"]],
        on="ml_row_id",
        how="inner",
    )
    parsed = [_parse_epoch_id(x) for x in selected["epoch_id"]]
    selected["trial"] = [x[1] for x in parsed]
    selected["seg"] = [x[2] for x in parsed]

    runs = [run for run in _contiguous_runs(selected) if len(run) >= min_run_windows]
    if not runs:
        raise ValueError("No contiguous replay runs were found.")

    packet_path = out_dir / "continuous_replay_samples.bin"
    ref_path = out_dir / "continuous_replay_reference.csv"
    meta_path = out_dir / "continuous_replay_metadata.json"

    ref_rows: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    n_samples_total = 0
    sequence = 0
    max_overlap_error = 0.0
    max_feature_abs_diff_vs_table = 0.0
    stream_id = 0

    with packet_path.open("wb") as packet:
        for run in runs:
            if len(ref_rows) >= limit_windows:
                break
            remaining = limit_windows - len(ref_rows)
            run = run.head(remaining)

            first = run.iloc[0]
            model_index = int(first["model_index"])
            participant_id = str(first["participant_id"])
            trial = int(first["trial"])
            ch_names = _load_ch_names(participant_id)

            windows: list[np.ndarray] = []
            sfreqs: list[float] = []
            row_payloads: list[tuple[pd.Series, pd.Series, Path]] = []
            for _, replay_row in run.iterrows():
                frow = feature_by_ml.loc[str(replay_row["ml_row_id"])]
                mrow = manifest_by_epoch.loc[(participant_id, str(frow["epoch_id"]))]
                npz_path = Path(str(mrow["eeg_epoch_file"]))
                data, time = _load_window(npz_path)
                windows.append(data)
                sfreqs.append(1.0 / float(np.median(np.diff(time))))
                row_payloads.append((replay_row, frow, npz_path))

            for prev, curr in zip(windows, windows[1:]):
                max_overlap_error = max(max_overlap_error, float(np.max(np.abs(prev[:, HOP_SAMPLES:] - curr[:, :HOP_SAMPLES]))))

            stream_data = np.concatenate([windows[0], *[w[:, HOP_SAMPLES:] for w in windows[1:]]], axis=1).astype(np.float32, copy=False)
            sfreq = float(sfreqs[0])
            baseline = _baseline_from_feature_row(row_payloads[0][1])
            baseline_values = _baseline_vector(baseline)

            packet.write(struct.pack("<IIHHd", STREAM_MAGIC, stream_id, model_index, len(windows), sfreq))
            packet.write(np.asarray(baseline_values, dtype="<f4").tobytes())

            for sample_index in range(stream_data.shape[1]):
                flags = 0
                if sample_index == 0:
                    flags |= 1
                if sample_index == stream_data.shape[1] - 1:
                    flags |= 2
                packet.write(struct.pack("<IIIHH", SAMPLE_MAGIC, sequence, sample_index, stream_id, flags))
                packet.write(np.asarray(stream_data[:, sample_index], dtype="<f4").tobytes())
                sequence += 1
            n_samples_total += int(stream_data.shape[1])

            for window_index, (replay_row, frow, npz_path) in enumerate(row_payloads):
                features = compute_window_features(windows[window_index], sfreqs[window_index], ch_names, baseline, order)
                feature_values = [float(features[name]) for name in order]
                table_values = [float(frow[name]) for name in order]
                max_feature_abs_diff_vs_table = max(
                    max_feature_abs_diff_vs_table,
                    max(abs(a - b) for a, b in zip(feature_values, table_values) if math.isfinite(a) and math.isfinite(b)),
                )
                pred = logits_and_prediction(model, model_index, feature_values)
                true_label = str(replay_row["true_label"])
                y_true.append(true_label)
                y_pred.append(str(pred["pred_label"]))
                ref_rows.append(
                    {
                        "stream_id": stream_id,
                        "window_index": window_index,
                        "start_sample_index": window_index * HOP_SAMPLES,
                        "end_sample_index": window_index * HOP_SAMPLES + N_SAMPLES - 1,
                        "replay_row_index": int(replay_row["replay_row_index"]),
                        "model_index": model_index,
                        "participant_id": participant_id,
                        "trial": trial,
                        "epoch_id": str(frow["epoch_id"]),
                        "true_label": true_label,
                        "python_pred_from_window_features": str(pred["pred_label"]),
                        "npz_path": str(npz_path),
                    }
                )

            stream_rows.append(
                {
                    "stream_id": stream_id,
                    "model_index": model_index,
                    "participant_id": participant_id,
                    "trial": trial,
                    "n_windows": len(windows),
                    "n_samples": int(stream_data.shape[1]),
                    "sfreq_hz": sfreq,
                    "first_replay_row_index": int(run.iloc[0]["replay_row_index"]),
                    "last_replay_row_index": int(run.iloc[-1]["replay_row_index"]),
                }
            )
            stream_id += 1

    with ref_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ref_rows)

    bacc, macro_f1 = _metric(y_true, y_pred, labels)
    summary = {
        "packet_bin": str(packet_path),
        "reference_csv": str(ref_path),
        "n_streams": len(stream_rows),
        "n_windows": len(ref_rows),
        "n_samples_total": n_samples_total,
        "n_channels": N_CHANNELS,
        "window_samples": N_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "target_sample_rate_hz": 250.0,
        "sample_packet_bytes": 4 + 4 + 4 + 2 + 2 + N_CHANNELS * 4,
        "stream_header_bytes": 4 + 4 + 2 + 2 + 8 + N_BASELINE * 4,
        "max_overlap_error": max_overlap_error,
        "max_feature_abs_diff_vs_stage4_table": max_feature_abs_diff_vs_table,
        "balanced_accuracy_from_window_features": bacc,
        "macro_f1_from_window_features": macro_f1,
        "labels": labels,
        "streams": stream_rows,
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sketch_dir = ROOT / "embedded" / "window_replay" / "esp32_continuous_replay"
    sketch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "embedded" / "model_export" / "model_data.h", sketch_dir / "model_data.h")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export contiguous preprocessed EEG sample streams for ESP32 Stage 2 replay.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-windows", type=int, default=26)
    parser.add_argument("--min-run-windows", type=int, default=2)
    args = parser.parse_args()
    summary = export(args.out_dir.resolve(), args.limit_windows, args.min_run_windows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
