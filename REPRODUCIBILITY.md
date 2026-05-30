# Reproducibility

Run commands from the repository root.

The commands below assume an ESP32 is connected on `COM6`. Replace the port as needed.

## Python Environment

```powershell
python -m pip install -r requirements.txt
```

## 1. Feature-Vector Replay

This stage sends precomputed 153-feature rows to the ESP32. The ESP32 performs imputation, clipping, scaling, active-feature masking, logistic regression, softmax and timing.

Build/upload firmware:

```powershell
cd embedded\esp32_feature_replay
python -m platformio run -t upload
cd ..\..
```

Replay and compare:

```powershell
python embedded\serial_replay.py --port COM6 --baud 115200 --out embedded\paper_validation\raw_logs\esp32_feature_vector_replay_full.csv
python embedded\compare_esp32_log.py --log embedded\paper_validation\raw_logs\esp32_feature_vector_replay_full.csv --out embedded\paper_validation\raw_logs\esp32_feature_vector_replay_full_summary.json
```

Expected paper result:

- Rows processed: 476 / 476
- ESP32-vs-Python agreement: 1.0000
- Mean split balanced accuracy: 0.7115
- Mean split macro-F1: 0.7004

## 2. Full Held-Out Block-Window Replay

This stage sends complete preprocessed 19-channel, 750-sample EEG windows to the ESP32. The ESP32 computes all 153 EEG features on-device and then classifies.

Build/upload firmware:

```powershell
cd embedded\window_replay\esp32_window_replay_full
python -m platformio run -t upload
cd ..\..\..
```

If the derived replay block file `embedded/window_replay/data/window_replay_blocks.bin` is available, run:

```powershell
python embedded\window_replay\block_window_replay.py --port COM6 --out embedded\paper_validation\raw_logs\esp32_window_block_476_full_chunked.csv
python embedded\window_replay\compare_block_window_replay.py --log embedded\paper_validation\raw_logs\esp32_window_block_476_full_chunked.csv --out embedded\paper_validation\raw_logs\esp32_window_block_476_full_chunked_summary.json
```

The paper result was captured in reset-bounded block-transfer chunks. Chunking affected only serial transport capture; the firmware, model, packet format, input windows and predictions were unchanged.

Representative chunk commands used during validation:

```powershell
python embedded\window_replay\block_window_replay.py --port COM6 --start 0 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_000_049.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 50 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_050_099.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 100 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_100_149.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 150 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_150_199.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 200 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_200_249.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 250 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_250_299.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 300 --limit 50 --out embedded\window_replay\logs\block_476_chunks\block_300_349.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 350 --limit 25 --out embedded\window_replay\logs\block_476_chunks\block_350_374.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 375 --limit 25 --out embedded\window_replay\logs\block_476_chunks\block_375_399.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 400 --limit 25 --out embedded\window_replay\logs\block_476_chunks\block_400_424.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 425 --limit 25 --out embedded\window_replay\logs\block_476_chunks\block_425_449.csv
python embedded\window_replay\block_window_replay.py --port COM6 --start 450 --limit 26 --out embedded\window_replay\logs\block_476_chunks\block_450_475.csv
```

The released merged log is:

```text
embedded/paper_validation/raw_logs/esp32_window_block_476_full_chunked.csv
```

Expected paper result:

- Windows processed: 476 / 476
- ESP32-vs-Python feature-row reference agreement: 1.0000
- ESP32-vs-Python window-feature reference agreement: 1.0000
- Mean split balanced accuracy: 0.7115
- Mean split macro-F1: 0.7004

## 3. Continuous Timed Replay

This stage streams preprocessed 19-channel EEG samples to the ESP32 at acquisition-like timing. The ESP32 maintains a 3 s ring buffer and classifies every 1.5 s.

Build/upload firmware:

```powershell
cd embedded\window_replay\esp32_continuous_replay
python -m platformio run -t upload
cd ..\..\..
```

Run replay and compare:

```powershell
python embedded\window_replay\continuous_sample_replay.py --port COM6 --timeout 90 --out embedded\paper_validation\raw_logs\esp32_continuous_50_scaled.csv
python embedded\window_replay\compare_continuous_replay.py --log embedded\paper_validation\raw_logs\esp32_continuous_50_scaled.csv --out embedded\paper_validation\raw_logs\esp32_continuous_50_scaled_summary.json
```

The 50-window continuous replay used the exported stream file `embedded/window_replay/data/continuous_replay_samples.bin` during validation. That large derived binary is not included in this release.

The three 26-window repeatability runs were captured as:

```powershell
python embedded\window_replay\continuous_sample_replay.py --port COM6 --out embedded\paper_validation\raw_logs\esp32_continuous_26_repeat1.csv
python embedded\window_replay\continuous_sample_replay.py --port COM6 --out embedded\paper_validation\raw_logs\esp32_continuous_26_repeat2.csv
python embedded\window_replay\continuous_sample_replay.py --port COM6 --out embedded\paper_validation\raw_logs\esp32_continuous_26_repeat3.csv
```

The 124-window candidate was exported during development but is not used as a paper claim and its large binary is not included in this release.

Expected paper result:

- Windows processed: 50 / 50
- ESP32-vs-Python agreement: 1.0000
- Packet errors: 0
- Dropped sequences: 0
- Buffer overruns: 0
- Mean total processing time: about 256 ms
- Max total processing time: about 259 ms

## 4. Regenerate Paper Tables And Plots

```powershell
python embedded\generate_final_embedded_validation.py
```

Main outputs:

- `embedded/paper_validation/embedded_stage_summary.csv`
- `embedded/paper_validation/embedded_stage_summary.json`
- `embedded/paper_validation/plots/embedded_staged_validation_pipeline.png`
- `embedded/paper_validation/plots/stage2_processing_time_per_window.png`
- `embedded/paper_validation/plots/block_window_476_confusion_matrix.png`

## Notes

This release intentionally excludes raw OpenNeuro data and large derived replay binaries. To regenerate replay binaries from scratch, use OpenNeuro `ds007262` v1.0.6 and the preprocessing/export pipeline used in the associated benchmark analysis.

`embedded/generate_final_embedded_validation.py` is release-safe: it regenerates summary tables and plots from the logs and summaries included in this repository.

## 5. Post-Hoc Grouped Trial/Block Sensitivity

The frozen embedded parity target uses the original row-level within-participant split. A post-hoc grouped trial/block sensitivity result is included to document the expected effect of removing overlapping-window leakage:

- `embedded/paper_validation/grouped_trial_sensitivity_summary.csv`
- `embedded/paper_validation/grouped_trial_sensitivity_summary.json`

This sensitivity check is an offline analysis result, not an ESP32 replay result. It should be cited only as a caveat/sensitivity result. The embedded validation target remains the frozen 476-row/window benchmark used for parity with the exported model.

## 6. Release Checks

Run lightweight consistency checks:

```powershell
python embedded\generate_final_embedded_validation.py
python -m pytest -q
```

The GitHub Actions workflow `.github/workflows/release-checks.yml` runs the same summary regeneration and tests on push and pull request.
