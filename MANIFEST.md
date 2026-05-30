# Release Manifest

This manifest lists the files included in the release package and their purpose.

## Top Level

| File | Purpose |
|---|---|
| `README.md` | Project overview, validation scope and repository guide. |
| `REPRODUCIBILITY.md` | Commands for replay validation stages and figure/table generation. |
| `LIMITATIONS.md` | Explicit scope limits and caveats. |
| `ARTIFACTS.md` | Paper claim-to-artifact map. |
| `CHECKSUMS.md` | Checksum documentation. |
| `checksums_sha256.txt` | SHA-256 checksums for release files. |
| `MANIFEST.md` | This file. |
| `requirements.txt` | Python package assumptions. |
| `.gitignore` | Excludes caches, builds, raw datasets and large replay binaries. |
| `.github/workflows/release-checks.yml` | GitHub Actions release consistency checks. |
| `tests/` | Lightweight pytest release checks. |
| `LICENSE` | MIT software license. |
| `CITATION.cff` | Citation metadata for GitHub/Zenodo. |

## Firmware

| File/Directory | Purpose |
|---|---|
| `embedded/esp32_feature_replay/` | Feature-vector replay firmware and exported model header. |
| `embedded/window_replay/esp32_window_replay_full/` | Full held-out block-window replay firmware with ESP32-side feature extraction. |
| `embedded/window_replay/esp32_continuous_replay/` | Continuous timed sample replay firmware with ring buffering and ESP32-side feature extraction. |

## Python Scripts

| File | Purpose |
|---|---|
| `embedded/export_logreg_for_esp32.py` | Export logistic-regression model artefacts for ESP32 replay. |
| `embedded/reference_inference.py` | No-scikit Python reference inference for exported feature rows. |
| `embedded/serial_replay.py` | Host feature-vector serial replay client. |
| `embedded/compare_esp32_log.py` | Compare ESP32 feature-vector logs with Python reference. |
| `embedded/generate_final_embedded_validation.py` | Generate paper-ready summary tables and plots from released evidence. |
| `embedded/window_replay/export_window_replay.py` | Export block-window replay data from preprocessed windows. |
| `embedded/window_replay/block_window_replay.py` | Host block-window replay client. |
| `embedded/window_replay/compare_block_window_replay.py` | Compare block-window ESP32 predictions with Python references. |
| `embedded/window_replay/export_continuous_replay.py` | Export continuous replay sample streams from preprocessed windows. |
| `embedded/window_replay/continuous_sample_replay.py` | Host continuous sample replay client. |
| `embedded/window_replay/compare_continuous_replay.py` | Compare continuous replay ESP32 predictions with Python reference. |
| `embedded/window_replay/window_feature_reference.py` | Python window-to-feature reference implementation. |
| `embedded/window_replay/spectral_reference_fixture.py` | Spectral feature fixture helper. |

## Model Artefacts

| File | Purpose |
|---|---|
| `embedded/model_export/logreg_replay_model.json` | Exported model package: feature order, class labels, imputation, clipping, scaler, active masks, coefficients and intercepts. |
| `embedded/model_export/model_data.h` | C/C++ model header generated from the exported model package. |
| `embedded/model_export/logreg_replay_vectors.csv` | 476 held-out feature rows used for feature-vector replay. |
| `embedded/model_export/logreg_replay_reference_predictions.csv` | Python reference predictions for the 476 held-out feature rows. |
| `embedded/model_export/python_reference_compare_summary.json` | Summary of no-scikit Python reference comparison. |

## Paper Validation Evidence

| File/Directory | Purpose |
|---|---|
| `embedded/paper_validation/embedded_stage_summary.csv` | Three-stage embedded validation summary table. |
| `embedded/paper_validation/embedded_stage_summary.json` | JSON version of the stage summary. |
| `embedded/paper_validation/block_window_476_confusion_matrix.csv` | Confusion matrix for the 476-window block replay. |
| `embedded/paper_validation/grouped_trial_sensitivity_summary.csv` | Post-hoc grouped trial/block sensitivity summary. |
| `embedded/paper_validation/grouped_trial_sensitivity_summary.json` | JSON version of the grouped sensitivity summary. |
| `embedded/paper_validation/stage2_repeatability_summary.csv` | Continuous replay repeatability summary. |
| `embedded/paper_validation/stage2_repeatability_summary.json` | JSON version of continuous repeatability summary. |
| `embedded/paper_validation/stage2_repeatability_windows.csv` | Per-window timing data for repeatability runs. |
| `embedded/paper_validation/stage2_confusion_matrix_50.csv` | Confusion matrix for the 50-window continuous subset. |
| `embedded/paper_validation/plots/` | Final plots used in the paper. |
| `embedded/paper_validation/raw_logs/` | Final ESP32 logs and comparison summaries supporting the paper tables. |

## Complete File List

| File | Purpose |
|---|---|
| `.github/workflows/release-checks.yml` | GitHub Actions workflow for release consistency checks. |
| `.gitignore` | Git ignore rules for caches, builds and raw datasets. |
| `ARTIFACTS.md` | Paper claim-to-artifact map. |
| `CHECKSUMS.md` | Checksum documentation. |
| `checksums_sha256.txt` | SHA-256 checksums for release files. |
| `CITATION.cff` | Citation metadata. |
| `embedded/compare_esp32_log.py` | Python export, replay, comparison or plotting script. |
| `embedded/esp32_feature_replay/esp32_feature_replay.ino` | ESP32 firmware source. |
| `embedded/esp32_feature_replay/model_data.h` | C/C++ exported model header for firmware. |
| `embedded/esp32_feature_replay/platformio.ini` | PlatformIO firmware build configuration. |
| `embedded/export_logreg_for_esp32.py` | Python export, replay, comparison or plotting script. |
| `embedded/generate_final_embedded_validation.py` | Python export, replay, comparison or plotting script. |
| `embedded/model_export/logreg_replay_model.json` | Exported model package or reference summary. |
| `embedded/model_export/logreg_replay_reference_predictions.csv` | Held-out feature rows, predictions or exported model artefact. |
| `embedded/model_export/logreg_replay_vectors.csv` | Held-out feature rows, predictions or exported model artefact. |
| `embedded/model_export/model_data.h` | C/C++ exported model header for firmware. |
| `embedded/model_export/python_reference_compare_summary.json` | Exported model package or reference summary. |
| `embedded/paper_validation/block_window_476_confusion_matrix.csv` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/embedded_stage_summary.csv` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/embedded_stage_summary.json` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/grouped_trial_sensitivity_summary.csv` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/grouped_trial_sensitivity_summary.json` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/plots/block_window_476_confusion_matrix.png` | Paper-ready validation plot. |
| `embedded/paper_validation/plots/embedded_prediction_agreement_summary.png` | Paper-ready validation plot. |
| `embedded/paper_validation/plots/embedded_staged_validation_pipeline.png` | Paper-ready validation plot. |
| `embedded/paper_validation/plots/stage2_confusion_matrix_50.png` | Paper-ready validation plot. |
| `embedded/paper_validation/plots/stage2_processing_time_per_window.png` | Paper-ready validation plot. |
| `embedded/paper_validation/plots/stage2_timing_components.png` | Paper-ready validation plot. |
| `embedded/paper_validation/raw_logs/esp32_continuous_26_repeat1.csv` | Raw ESP32 validation log used as evidence. |
| `embedded/paper_validation/raw_logs/esp32_continuous_26_repeat1_summary.json` | Comparison summary for an ESP32 validation log. |
| `embedded/paper_validation/raw_logs/esp32_continuous_26_repeat2.csv` | Raw ESP32 validation log used as evidence. |
| `embedded/paper_validation/raw_logs/esp32_continuous_26_repeat2_summary.json` | Comparison summary for an ESP32 validation log. |
| `embedded/paper_validation/raw_logs/esp32_continuous_26_repeat3.csv` | Raw ESP32 validation log used as evidence. |
| `embedded/paper_validation/raw_logs/esp32_continuous_26_repeat3_summary.json` | Comparison summary for an ESP32 validation log. |
| `embedded/paper_validation/raw_logs/esp32_continuous_50_scaled.csv` | Raw ESP32 validation log used as evidence. |
| `embedded/paper_validation/raw_logs/esp32_continuous_50_scaled_summary.json` | Comparison summary for an ESP32 validation log. |
| `embedded/paper_validation/raw_logs/esp32_feature_vector_replay_full.csv` | Raw ESP32 validation log used as evidence. |
| `embedded/paper_validation/raw_logs/esp32_feature_vector_replay_full_summary.json` | Comparison summary for an ESP32 validation log. |
| `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked.csv` | Raw ESP32 validation log used as evidence. |
| `embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked_summary.json` | Comparison summary for an ESP32 validation log. |
| `embedded/paper_validation/stage2_confusion_matrix_50.csv` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/stage2_repeatability_summary.csv` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/stage2_repeatability_summary.json` | Paper-validation summary table or metadata. |
| `embedded/paper_validation/stage2_repeatability_windows.csv` | Paper-validation summary table or metadata. |
| `embedded/reference_inference.py` | Python export, replay, comparison or plotting script. |
| `embedded/serial_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/block_window_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/compare_block_window_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/compare_continuous_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/continuous_sample_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/esp32_continuous_replay/model_data.h` | C/C++ exported model header for firmware. |
| `embedded/window_replay/esp32_continuous_replay/platformio.ini` | PlatformIO firmware build configuration. |
| `embedded/window_replay/esp32_continuous_replay/src/main.cpp` | ESP32 firmware source. |
| `embedded/window_replay/esp32_window_replay_full/model_data.h` | C/C++ exported model header for firmware. |
| `embedded/window_replay/esp32_window_replay_full/platformio.ini` | PlatformIO firmware build configuration. |
| `embedded/window_replay/esp32_window_replay_full/src/main.cpp` | ESP32 firmware source. |
| `embedded/window_replay/export_continuous_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/export_window_replay.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/spectral_reference_fixture.py` | Python export, replay, comparison or plotting script. |
| `embedded/window_replay/window_feature_reference.py` | Python export, replay, comparison or plotting script. |
| `LICENSE` | MIT license. |
| `LIMITATIONS.md` | Scope and limitations. |
| `MANIFEST.md` | Release file manifest. |
| `README.md` | Project overview and validation scope. |
| `REPRODUCIBILITY.md` | Reproduction commands for validation stages. |
| `requirements.txt` | Python dependency assumptions. |
| `tests/test_release_summaries.py` | Pytest release consistency check. |
