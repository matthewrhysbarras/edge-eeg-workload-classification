from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "embedded" / "paper_validation"
RAW = OUT / "raw_logs"
PLOTS = OUT / "plots"
LABELS = ["baseline", "low_1_2_3", "high_4_5_6"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _confusion(reference_csv: Path, log_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    log = pd.read_csv(log_csv)
    merged = log.merge(ref, on=["stream_id", "window_index"], suffixes=("_esp32", "_ref"))
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for _, row in merged.iterrows():
        matrix.loc[str(row["true_label"]), str(row["pred_label"])] += 1
    return matrix


def _block_confusion(reference_csv: Path, log_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    log = pd.read_csv(log_csv)
    merged = log.merge(ref, on="replay_row_index", suffixes=("_esp32", "_ref"))
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for _, row in merged.iterrows():
        matrix.loc[str(row["true_label"]), str(row["pred_label"])] += 1
    return matrix


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    feature_vector = _load_json(ROOT / "embedded" / "logs" / "esp32_replay_full_summary.json")
    block = _load_json(ROOT / "embedded" / "window_replay" / "logs" / "esp32_window_block_476_full_chunked_summary.json")
    repeat_paths = [
        ROOT / "embedded" / "window_replay" / "logs" / f"esp32_continuous_26_repeat{i}_summary.json"
        for i in (1, 2, 3)
    ]
    repeat_summaries = [_load_json(path) for path in repeat_paths]
    meta26 = _load_json(ROOT / "embedded" / "paper_validation" / "continuous_26_metadata.json")
    meta50 = _load_json(ROOT / "embedded" / "paper_validation" / "continuous_50_metadata.json")
    meta124 = _load_json(ROOT / "embedded" / "paper_validation" / "continuous_124_candidate_data" / "continuous_replay_metadata.json")
    scaled50 = _load_json(ROOT / "embedded" / "window_replay" / "logs" / "esp32_continuous_50_scaled_summary.json")

    stage_rows = [
        {
            "stage": "feature_vector_replay",
            "esp32_work": "classifier only: imputation, clipping, scaling, active-feature masking, logistic regression, softmax",
            "input_type": "precomputed 153-feature rows",
            "rows_or_windows": feature_vector["rows"],
            "agreement": feature_vector["agreement_with_python_reference"],
            "balanced_accuracy": feature_vector["mean_split_balanced_accuracy"],
            "macro_f1": feature_vector["mean_split_macro_f1"],
            "mean_feature_us": "",
            "max_feature_us": "",
            "mean_classifier_or_inference_us": feature_vector["mean_inference_us_esp32"],
            "max_classifier_or_inference_us": feature_vector["max_inference_us_esp32"],
            "mean_total_us": feature_vector["mean_inference_us_esp32"],
            "max_total_us": feature_vector["max_inference_us_esp32"],
            "ram_bytes": 30884,
            "ram_percent": 9.4,
            "flash_bytes": 336181,
            "flash_percent": 25.6,
            "limitation": "uses PC-generated features; no ESP32 feature extraction",
        },
        {
            "stage": "block_window_replay",
            "esp32_work": "153-feature extraction from one preprocessed 19x750 EEG window, then classifier",
            "input_type": "complete preprocessed 19-channel 3 s windows",
            "rows_or_windows": block["rows"],
            "agreement": block["agreement_with_window_feature_python_reference"],
            "balanced_accuracy": block["mean_split_balanced_accuracy"],
            "macro_f1": block["mean_split_macro_f1"],
            "pooled_balanced_accuracy": block["pooled_balanced_accuracy"],
            "pooled_macro_f1": block["pooled_macro_f1"],
            "mean_feature_us": block["mean_feature_us"],
            "max_feature_us": block["max_feature_us"],
            "mean_classifier_or_inference_us": block["mean_classifier_us"],
            "max_classifier_or_inference_us": block["max_classifier_us"],
            "mean_total_us": block["mean_total_us"],
            "max_total_us": block["max_total_us"],
            "ram_bytes": 87436,
            "ram_percent": 26.7,
            "flash_bytes": 340025,
            "flash_percent": 25.9,
            "limitation": "preprocessed windows supplied by PC; no continuous buffering or ADC",
        },
        {
            "stage": "continuous_timed_sample_replay",
            "esp32_work": "sample buffering, 3 s ring windowing every 1.5 s, 153-feature extraction, classifier",
            "input_type": "timed binary stream of preprocessed 19-channel EEG samples",
            "rows_or_windows": scaled50["windows_processed"],
            "agreement": scaled50["prediction_agreement"],
            "balanced_accuracy": scaled50["balanced_accuracy"],
            "macro_f1": scaled50["macro_f1"],
            "mean_feature_us": scaled50["mean_feature_us"],
            "max_feature_us": scaled50["max_feature_us"],
            "mean_classifier_or_inference_us": scaled50["mean_classifier_us"],
            "max_classifier_or_inference_us": scaled50["max_classifier_us"],
            "mean_total_us": scaled50["mean_total_us"],
            "max_total_us": scaled50["max_total_us"],
            "ram_bytes": 115364,
            "ram_percent": 35.2,
            "flash_bytes": 340997,
            "flash_percent": 26.0,
            "limitation": "input samples are preprocessed offline; no live ADC acquisition",
        },
    ]
    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(OUT / "embedded_stage_summary.csv", index=False)
    (OUT / "embedded_stage_summary.json").write_text(json.dumps(stage_rows, indent=2), encoding="utf-8")

    repeat_rows = []
    all_windows = []
    for i, summary in enumerate(repeat_summaries, start=1):
        log_path = ROOT / "embedded" / "window_replay" / "logs" / f"esp32_continuous_26_repeat{i}.csv"
        _copy(log_path, RAW / log_path.name)
        _copy(ROOT / "embedded" / "window_replay" / "logs" / f"esp32_continuous_26_repeat{i}_summary.json", RAW / f"esp32_continuous_26_repeat{i}_summary.json")
        log = pd.read_csv(log_path)
        log["run"] = i
        all_windows.append(log)
        repeat_rows.append(
            {
                "run": i,
                "windows_processed": summary["windows_processed"],
                "windows_expected": summary["windows_expected"],
                "prediction_agreement": summary["prediction_agreement"],
                "prediction_mismatches": summary["prediction_mismatches"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "macro_f1": summary["macro_f1"],
                "mean_feature_us": summary["mean_feature_us"],
                "max_feature_us": summary["max_feature_us"],
                "mean_classifier_us": summary["mean_classifier_us"],
                "max_classifier_us": summary["max_classifier_us"],
                "mean_total_us": summary["mean_total_us"],
                "max_total_us": summary["max_total_us"],
                "malformed_packets": summary["max_malformed_packets"],
                "dropped_sequences": summary["max_dropped_sequences"],
                "buffer_overruns": summary["max_buffer_overrun_count"],
            }
        )
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(OUT / "stage2_repeatability_summary.csv", index=False)

    windows_df = pd.concat(all_windows, ignore_index=True)
    windows_df.to_csv(OUT / "stage2_repeatability_windows.csv", index=False)
    timing = {
        "n_runs": len(repeat_summaries),
        "n_windows_per_run": int(repeat_summaries[0]["windows_processed"]),
        "total_window_observations": int(len(windows_df)),
        "prediction_agreement_min": float(repeat_df["prediction_agreement"].min()),
        "prediction_mismatches_total": int(repeat_df["prediction_mismatches"].sum()),
        "malformed_packets_max": int(repeat_df["malformed_packets"].max()),
        "dropped_sequences_max": int(repeat_df["dropped_sequences"].max()),
        "buffer_overruns_max": int(repeat_df["buffer_overruns"].max()),
        "feature_us_mean": float(windows_df["feature_us"].mean()),
        "feature_us_std": float(windows_df["feature_us"].std(ddof=1)),
        "feature_us_max": float(windows_df["feature_us"].max()),
        "classifier_us_mean": float(windows_df["classifier_us"].mean()),
        "classifier_us_std": float(windows_df["classifier_us"].std(ddof=1)),
        "classifier_us_max": float(windows_df["classifier_us"].max()),
        "total_us_mean": float(windows_df["total_us"].mean()),
        "total_us_std": float(windows_df["total_us"].std(ddof=1)),
        "total_us_max": float(windows_df["total_us"].max()),
        "hop_budget_us": 1_500_000,
        "max_total_fraction_of_hop": float(windows_df["total_us"].max() / 1_500_000),
        "continuous_50_scaled_windows": scaled50["windows_processed"],
        "continuous_50_scaled_agreement": scaled50["prediction_agreement"],
        "continuous_50_scaled_balanced_accuracy": scaled50["balanced_accuracy"],
        "continuous_50_scaled_macro_f1": scaled50["macro_f1"],
        "continuous_50_scaled_packet_errors": {
            "malformed": scaled50["max_malformed_packets"],
            "dropped_sequences": scaled50["max_dropped_sequences"],
            "buffer_overruns": scaled50["max_buffer_overrun_count"],
        },
        "continuous_124_candidate_windows": meta124["n_windows"],
        "continuous_124_candidate_note": "A larger contiguous candidate was available, but the paper-grade continuous timing claim uses the clean 50-window replay with separate 26-window repeatability checks.",
    }
    (OUT / "stage2_repeatability_summary.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    conf = _confusion(ROOT / "embedded" / "paper_validation" / "continuous_50_reference.csv", ROOT / "embedded" / "window_replay" / "logs" / "esp32_continuous_50_scaled.csv")
    conf.to_csv(OUT / "stage2_confusion_matrix_50.csv")
    block_conf = _block_confusion(ROOT / "embedded" / "window_replay" / "data" / "window_replay_reference.csv", ROOT / "embedded" / "window_replay" / "logs" / "esp32_window_block_476_full_chunked.csv")
    block_conf.to_csv(OUT / "block_window_476_confusion_matrix.csv")

    try:
        import matplotlib.pyplot as plt

        plot_df = windows_df.reset_index().copy()
        plot_df["total_ms"] = plot_df["total_us"] / 1000.0
        plot_df["feature_ms"] = plot_df["feature_us"] / 1000.0
        ax = plot_df.plot(x="index", y="total_ms", kind="line", color="black", linewidth=1.4, label="Total", figsize=(5.6, 3.0))
        plot_df.plot(x="index", y="feature_ms", kind="line", color="0.45", linewidth=1.0, label="Feature extraction", ax=ax)
        ax.set_xlabel("Window observation")
        ax.set_ylabel("Processing time (ms)")
        ax.set_ylim(240, 265)
        ax.text(0.02, 0.94, "1.5 s hop budget = 1500 ms\nWorst case = 259 ms (17.3%)", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title("Continuous replay timing")
        plt.tight_layout()
        plt.savefig(PLOTS / "stage2_processing_time_per_window.png", dpi=300)
        plt.close()

        timing_means = pd.DataFrame(
            {
                "component": ["Feature extraction", "Classifier"],
                "mean_us": [windows_df["feature_us"].mean(), windows_df["classifier_us"].mean()],
            }
        )
        ax = timing_means.plot(x="component", y="mean_us", kind="bar", legend=False, figsize=(5, 3))
        ax.set_ylabel("Mean time (us)")
        ax.set_title("ESP32 Stage 2 Timing Components")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS / "stage2_timing_components.png", dpi=200)
        plt.close()

        ax = stage_df.plot(x="stage", y="agreement", kind="bar", legend=False, figsize=(6, 3))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Prediction agreement")
        ax.set_title("ESP32 Agreement by Validation Stage")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS / "embedded_prediction_agreement_summary.png", dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.axis("off")
        xs = [0.12, 0.38, 0.64, 0.88]
        labels = [
            "PC offline\npreprocessing",
            "Feature-vector\nclassifier parity",
            "Block-window\nfeature parity",
            "Continuous timed\nsample replay",
        ]
        details = [
            "BIDS/MNE +\npreprocessed EEG",
            "476 rows\n1.0000 agreement",
            "476 windows\nBA 0.7115",
            "50 windows\nreal-time timing",
        ]
        for x, label, detail in zip(xs, labels, details):
            ax.text(x, 0.62, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f1fb", edgecolor="#315a7d"))
            ax.text(x, 0.26, detail, ha="center", va="center", fontsize=8)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.annotate("", xy=(x1 - 0.08, 0.62), xytext=(x0 + 0.08, 0.62), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.set_title("Staged ESP32 Embedded Validation Pipeline")
        plt.tight_layout()
        plt.savefig(PLOTS / "embedded_staged_validation_pipeline.png", dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(conf.values, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("ESP32 predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Stage 2 Confusion Matrix")
        for r in range(conf.shape[0]):
            for c in range(conf.shape[1]):
                ax.text(c, r, int(conf.iat[r, c]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(PLOTS / "stage2_confusion_matrix_50.png", dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(block_conf.values, cmap="Greens")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("ESP32 predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Block-Window 476 Confusion Matrix")
        for r in range(block_conf.shape[0]):
            for c in range(block_conf.shape[1]):
                ax.text(c, r, int(block_conf.iat[r, c]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(PLOTS / "block_window_476_confusion_matrix.png", dpi=200)
        plt.close()
    except Exception as exc:
        (OUT / "plot_generation_error.txt").write_text(str(exc), encoding="utf-8")

    _copy(ROOT / "embedded" / "logs" / "esp32_replay_full.csv", RAW / "esp32_feature_vector_replay_full.csv")
    _copy(ROOT / "embedded" / "logs" / "esp32_replay_full_summary.json", RAW / "esp32_feature_vector_replay_full_summary.json")
    _copy(ROOT / "embedded" / "window_replay" / "logs" / "esp32_window_block_476_full_chunked.csv", RAW / "esp32_window_block_476_full_chunked.csv")
    _copy(ROOT / "embedded" / "window_replay" / "logs" / "esp32_window_block_476_full_chunked_summary.json", RAW / "esp32_window_block_476_full_chunked_summary.json")
    _copy(ROOT / "embedded" / "window_replay" / "logs" / "esp32_continuous_50_scaled.csv", RAW / "esp32_continuous_50_scaled.csv")
    _copy(ROOT / "embedded" / "window_replay" / "logs" / "esp32_continuous_50_scaled_summary.json", RAW / "esp32_continuous_50_scaled_summary.json")

    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
