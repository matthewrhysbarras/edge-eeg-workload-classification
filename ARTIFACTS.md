# Claim-To-Artifact Map

This table maps each paper claim to the released artefacts that support it.

| Paper claim | Primary artefact(s) | Verification command |
|---|---|---|
| Feature-vector replay processed 476 / 476 rows with 1.0000 ESP32-vs-Python agreement. | `embedded/paper_validation/raw_logs/esp32_feature_vector_replay_full.csv`; `embedded/paper_validation/raw_logs/esp32_feature_vector_replay_full_summary.json` | `python embedded/compare_esp32_log.py --log embedded/paper_validation/raw_logs/esp32_feature_vector_replay_full.csv --out embedded/paper_validation/raw_logs/esp32_feature_vector_replay_full_summary.json` |
| Feature-vector replay reproduced mean split balanced accuracy 0.7115 and mean split macro-F1 0.7004. | `embedded/model_export/logreg_replay_reference_predictions.csv`; `embedded/paper_validation/embedded_stage_summary.csv` | `python embedded/generate_final_embedded_validation.py` |
| Full held-out block-window replay processed 476 / 476 windows. | `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked.csv`; `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked_summary.json` | `python embedded/window_replay/compare_block_window_replay.py --log embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked.csv --out embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked_summary.json` |
| ESP32 block-window predictions matched the Python reference with 1.0000 agreement and zero mismatches. | `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked_summary.json` | Inspect `agreement_with_feature_row_python_reference`, `agreement_with_window_feature_python_reference`, and mismatch counts. |
| Full block-window mean split BA/F1 were 0.7115/0.7004; pooled BA/F1 were 0.7083/0.6975. | `embedded/paper_validation/embedded_stage_summary.csv`; `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked_summary.json` | `python embedded/generate_final_embedded_validation.py` |
| Block-window feature errors were small but not bitwise identical. | `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked_summary.json` | Inspect `max_feature_abs_error` and `mean_feature_abs_error`. |
| Continuous timed replay processed 50 / 50 windows with 1.0000 agreement and zero packet errors. | `embedded/paper_validation/raw_logs/esp32_continuous_50_scaled.csv`; `embedded/paper_validation/raw_logs/esp32_continuous_50_scaled_summary.json` | `python embedded/window_replay/compare_continuous_replay.py --log embedded/paper_validation/raw_logs/esp32_continuous_50_scaled.csv --out embedded/paper_validation/raw_logs/esp32_continuous_50_scaled_summary.json` |
| Continuous replay timing remained below the 1.5 s hop budget. | `embedded/paper_validation/stage2_repeatability_summary.json`; `embedded/paper_validation/stage2_repeatability_windows.csv`; `embedded/paper_validation/plots/stage2_processing_time_per_window.png` | `python embedded/generate_final_embedded_validation.py` |
| The 476-window confusion matrix used in the paper. | `embedded/paper_validation/block_window_476_confusion_matrix.csv`; `embedded/paper_validation/plots/block_window_476_confusion_matrix.png` | `python embedded/generate_final_embedded_validation.py` |
| Post-hoc grouped trial/block sensitivity result documents the row-level overlap caveat. | `embedded/paper_validation/grouped_trial_sensitivity_summary.csv`; `embedded/paper_validation/grouped_trial_sensitivity_summary.json` | Inspect the sensitivity files; this is an offline caveat, not an ESP32 replay claim. |
| Release excludes raw OpenNeuro data and large replay binaries. | `.gitignore`; `LIMITATIONS.md`; `README.md` | Confirm no `.bin`, `.npz`, `.edf`, `.bdf`, `.set`, `.fdt` files are present. |

## Regeneration Scope

The released files support high-level reproduction of ESP32 replay validation and regeneration of the paper summary tables/plots from included logs.

The release does not include raw OpenNeuro data, large replay binaries, or a vendored copy of the full offline benchmark preprocessing pipeline. Regenerating replay binaries from raw data requires OpenNeuro `ds007262` v1.0.6 and the separately released benchmark pipeline.
