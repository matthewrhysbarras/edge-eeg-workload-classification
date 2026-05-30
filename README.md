# Edge EEG Workload Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20452989.svg)](https://doi.org/10.5281/zenodo.20452989)

This repository contains the embedded validation artefacts for the conference paper **Edge EEG Workload Classification**.

The project validates whether a selected EEG workload-classification pipeline can run on ESP32-class hardware using replayed data. The source model is an EEG-only logistic-regression pipeline selected from an offline benchmark of OpenNeuro `ds007262`.

## What The Paper Validates

The validation is staged:

1. **Feature-vector replay:** the ESP32 receives precomputed 153-feature EEG rows and runs classifier preprocessing plus logistic regression.
2. **Full held-out block-window replay:** the ESP32 receives preprocessed 19-channel, 3 s EEG windows and computes all 153 EEG features on-device before classification.
3. **Continuous timed replay:** the ESP32 receives a timed stream of preprocessed 19-channel EEG samples, maintains a 3 s ring buffer, classifies every 1.5 s, computes features on-device, and logs timing.

The strongest embedded parity result is the full held-out block-window replay: 476 / 476 windows processed, 1.0000 ESP32-vs-Python prediction agreement, mean split balanced accuracy 0.7115, and mean split macro-F1 0.7004.

The continuous timed replay is an acquisition-like timing demonstration on a stable 50-window subset, not the full held-out accuracy claim.

## What Is Not Claimed

This is **not** live EEG acquisition and does **not** claim live ADC classification.

The PC-side pipeline still performs BIDS loading, EEG preprocessing, bad-channel handling, interpolation, montage normalization and average referencing before replay. The ESP32 performs embedded feature extraction and classification on replayed/preprocessed inputs.

The original within-participant benchmark split is row-level with overlapping 3 s windows. Overlapping-window leakage is acknowledged in the paper and in `LIMITATIONS.md`; the embedded result is primarily a parity and timing validation against that frozen benchmark target.

## Hardware Used

- ESP32 development target used for validation: ESP32-D0WD-V3.
- Serial port in the validation environment: `COM6`.
- Serial baud for block-window and continuous replay: `921600`.
- The paper frames this as testing the ESP32-side workload in the context of a broader STM32/ADS1299 mobile EEG/fNIRS architecture; the replay stream substitutes for the STM32/ADS1299 acquisition and filtering path.

## Software Assumptions

The validation environment used:

- Python 3.12
- NumPy 1.26
- pandas 2.3
- SciPy 1.14
- scikit-learn 1.5
- pyserial 3.5
- matplotlib 3.9
- PlatformIO Core 6.1

Install Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Repository Contents

- `embedded/esp32_feature_replay/`: feature-vector replay firmware.
- `embedded/window_replay/esp32_window_replay_full/`: full held-out block-window replay firmware.
- `embedded/window_replay/esp32_continuous_replay/`: continuous timed sample replay firmware.
- `embedded/model_export/`: exported logistic-regression model, feature rows, class labels, preprocessing parameters and C/C++ header.
- `embedded/paper_validation/`: paper-ready summary tables, plots and validation logs.
- `ARTIFACTS.md`: claim-to-artifact map linking paper claims to released evidence.
- `REPRODUCIBILITY.md`: exact commands used for the validation stages.
- `LIMITATIONS.md`: scope and caveats.
- `MANIFEST.md`: file-by-file release manifest.
- `checksums_sha256.txt`: SHA-256 checksums for release files.
- `.github/workflows/release-checks.yml` and `tests/`: lightweight checks that regenerate release summaries and verify key paper values.

## Citation

Archived release DOI:

https://doi.org/10.5281/zenodo.20452989

## Data

The original dataset is OpenNeuro `ds007262` version `1.0.6`:

https://doi.org/10.18112/openneuro.ds007262.v1.0.6

This release does not include the raw OpenNeuro dataset. It includes exported model artefacts and validation logs needed to support the embedded replay results. Large derived replay binaries are intentionally excluded and can be regenerated from the source dataset and preprocessing pipeline.

The release is intended to support embedded-validation auditability rather than full raw-data-to-model regeneration. Full regeneration from raw data requires the external benchmark preprocessing pipeline used to create the replay inputs. The associated offline benchmark code is available separately at:

https://github.com/LMBooth/Arithmetic_Workload_Estimation

The embedded release uses repo-relative metadata for the exported model, logs and summaries; no private local filesystem paths are required for the released validation checks.
