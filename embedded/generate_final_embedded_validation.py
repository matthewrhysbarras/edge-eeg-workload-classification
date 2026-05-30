from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "embedded" / "paper_validation"
RAW = OUT / "raw_logs"
PLOTS = OUT / "plots"
LABELS = ["baseline", "low_1_2_3", "high_4_5_6"]
DISPLAY_LABELS = ["Baseline", "Low workload", "High workload"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_stage_summary() -> pd.DataFrame:
    # Release-only mode: use included paper summaries rather than unreleased working-tree paths.
    existing = OUT / "embedded_stage_summary.json"
    stage_rows = _load_json(existing)
    df = pd.DataFrame(stage_rows)
    df.to_csv(OUT / "embedded_stage_summary.csv", index=False)
    existing.write_text(json.dumps(stage_rows, indent=2), encoding="utf-8")
    return df


def _write_repeatability_summary() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    repeat_rows = []
    all_windows = []
    for i in (1, 2, 3):
        summary = _load_json(RAW / f"esp32_continuous_26_repeat{i}_summary.json")
        log = pd.read_csv(RAW / f"esp32_continuous_26_repeat{i}.csv")
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
    windows_df = pd.concat(all_windows, ignore_index=True)
    repeat_df.to_csv(OUT / "stage2_repeatability_summary.csv", index=False)
    windows_df.to_csv(OUT / "stage2_repeatability_windows.csv", index=False)

    scaled50 = _load_json(RAW / "esp32_continuous_50_scaled_summary.json")
    timing = {
        "n_runs": len(repeat_rows),
        "n_windows_per_run": int(repeat_df["windows_processed"].iloc[0]),
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
        "continuous_124_candidate_windows": 124,
        "continuous_124_candidate_note": "A larger contiguous candidate was exported during development, but the paper-grade continuous timing claim uses the clean 50-window replay with separate 26-window repeatability checks. The large derived binary is not included in this release.",
        "note": "Release-only regeneration uses included ESP32 logs and summary JSON files. Raw OpenNeuro data and large replay binaries are intentionally excluded.",
    }
    (OUT / "stage2_repeatability_summary.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return repeat_df, windows_df, timing


def _block_confusion_from_log() -> pd.DataFrame:
    log = pd.read_csv(RAW / "esp32_window_block_476_full_chunked.csv")
    vectors = pd.read_csv(ROOT / "embedded" / "model_export" / "logreg_replay_vectors.csv", usecols=["replay_row_index", "true_label"])
    merged = log.merge(vectors, on="replay_row_index", how="inner")
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for _, row in merged.iterrows():
        matrix.loc[str(row["true_label"]), str(row["pred_label"])] += 1
    matrix.to_csv(OUT / "block_window_476_confusion_matrix.csv")
    return matrix


def _continuous_confusion_from_log() -> pd.DataFrame:
    # The release does not include the full continuous reference table to avoid dataset path metadata.
    # The published confusion matrix is included and copied through as the release source of truth.
    path = OUT / "stage2_confusion_matrix_50.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame(0, index=LABELS, columns=LABELS)


def _write_plots(stage_df: pd.DataFrame, windows_df: pd.DataFrame, block_conf: pd.DataFrame, cont_conf: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
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
        labels = ["PC offline\npreprocessing", "Feature-vector\nclassifier parity", "Block-window\nfeature parity", "Continuous timed\nsample replay"]
        details = ["BIDS/MNE +\npreprocessed EEG", "476 rows\n1.0000 agreement", "476 windows\nBA 0.7115", "50 windows\nreal-time timing"]
        for x, label, detail in zip(xs, labels, details):
            ax.text(x, 0.62, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f1fb", edgecolor="#315a7d"))
            ax.text(x, 0.26, detail, ha="center", va="center", fontsize=8)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.annotate("", xy=(x1 - 0.08, 0.62), xytext=(x0 + 0.08, 0.62), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.set_title("Staged ESP32 Embedded Validation Pipeline")
        plt.tight_layout()
        plt.savefig(PLOTS / "embedded_staged_validation_pipeline.png", dpi=200)
        plt.close()

        for matrix, filename, title, cmap in [
            (cont_conf, "stage2_confusion_matrix_50.png", "Stage 2 Confusion Matrix", "Blues"),
            (block_conf, "block_window_476_confusion_matrix.png", "476-window block replay", "Greens"),
        ]:
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(matrix.values, cmap=cmap)
            ax.set_xticks(range(len(DISPLAY_LABELS)), DISPLAY_LABELS, rotation=30, ha="right")
            ax.set_yticks(range(len(DISPLAY_LABELS)), DISPLAY_LABELS)
            ax.set_xlabel("ESP32 predicted label")
            ax.set_ylabel("True label")
            ax.set_title(title)
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    ax.text(c, r, int(matrix.iat[r, c]), ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(PLOTS / filename, dpi=200)
            plt.close()
    except Exception as exc:
        (OUT / "plot_generation_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    stage_df = _write_stage_summary()
    _, windows_df, timing = _write_repeatability_summary()
    block_conf = _block_confusion_from_log()
    cont_conf = _continuous_confusion_from_log()
    _write_plots(stage_df, windows_df, block_conf, cont_conf)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
