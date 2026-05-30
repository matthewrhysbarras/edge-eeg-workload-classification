# Limitations

## Replayed Data, Not Live ADC Acquisition

The validation uses replayed/preprocessed EEG data. It does not perform live ADC acquisition and does not validate electrode, analogue front-end, impedance, ADC, or live timing behaviour.

## PC-Side Preprocessing Remains Outside The ESP32 Path

The PC-side pipeline performs BIDS loading, EEG channel selection, unit conversion, 50 Hz notch filtering, 2-40 Hz bandpass filtering, bad-channel handling where applicable, montage normalization and average referencing. These preprocessing steps are not implemented on the ESP32.

The replay stream simulates the output that would otherwise be provided by the STM32/ADS1299 acquisition and filtering path in the broader mobile EEG/fNIRS device architecture.

## Overlapping-Window Leakage In The Original Benchmark Target

The replicated benchmark target uses within-participant, row-level stratified holdout with 3 s windows and 1.5 s overlap. Because the split is row-level rather than grouped by trial/block, overlapping windows from the same trial can appear in both training and test sets.

In the exported 476-row held-out set, 405 held-out windows had an adjacent overlapping segment in the training set. The reported 0.7115 mean split balanced accuracy and 0.7004 mean split macro-F1 should therefore be interpreted as benchmark-replication and embedded-parity targets, not leakage-free trial-level generalization estimates.

## Full Held-Out Embedded Parity Result

The 476-window block replay is the full held-out embedded parity result:

- preprocessed EEG window -> ESP32 feature extraction -> ESP32 classifier -> prediction
- 476 / 476 windows processed
- 1.0000 ESP32-vs-Python prediction agreement

This validates the embedded feature-extraction and classifier implementation against the frozen benchmark target.

## Continuous Timed Replay Subset

The 50-window continuous replay is an acquisition-like timing validation on a stable subset. It demonstrates that sample reception, buffering, 3 s windowing, 1.5 s hop triggering, feature extraction and classification run within the timing budget.

It is not used as the full held-out accuracy claim.

## Future Work

Future work should evaluate grouped trial/block splits, live STM32/ADS1299 acquisition, power and energy per inference, higher continuous replay loads, possible STM32 CMSIS-DSP feature extraction, and multimodal extension to fNIRS and other signals.

