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

If the derived replay block file is available, run:

```powershell
python embedded\window_replay\block_window_replay.py --port COM6 --out embedded\paper_validation\raw_logs\esp32_window_block_476_full_chunked.csv
python embedded\window_replay\compare_block_window_replay.py --log embedded\paper_validation\raw_logs\esp32_window_block_476_full_chunked.csv --out embedded\paper_validation\raw_logs\esp32_window_block_476_full_chunked_summary.json
```

The paper result was captured in reset-bounded block-transfer chunks. Chunking affected only serial transport capture; the firmware, model, packet format, input windows and predictions were unchanged.

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

