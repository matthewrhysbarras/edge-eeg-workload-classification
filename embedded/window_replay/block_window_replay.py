from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLOCKS = ROOT / "embedded" / "window_replay" / "data" / "window_replay_blocks.bin"
DEFAULT_LOG_DIR = ROOT / "embedded" / "window_replay" / "logs"
PACKET_BYTES = 57692


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send preprocessed 19x750 EEG window blocks to the ESP32.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0, help="Zero-based packet index to start streaming from.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: python -m pip install pyserial") from exc

    payload = args.blocks.read_bytes()
    if len(payload) % PACKET_BYTES != 0:
        raise SystemExit(f"Unexpected block file size {len(payload)}; not divisible by {PACKET_BYTES}.")
    n_packets = len(payload) // PACKET_BYTES
    start_packet = max(0, int(args.start))
    if start_packet >= n_packets:
        raise SystemExit(f"--start {start_packet} is outside packet count {n_packets}.")
    if args.limit is not None:
        n_packets_to_send = min(n_packets - start_packet, max(0, int(args.limit)))
    else:
        n_packets_to_send = n_packets - start_packet

    out_path = args.out or DEFAULT_LOG_DIR / f"esp32_window_block_replay_{_timestamp()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "replay_row_index",
        "model_index",
        "pred_label",
        "max_abs_feature_error",
        "max_feature_error_index",
        "mean_abs_feature_error",
        "rmse_feature_error",
        "logit_0",
        "logit_1",
        "logit_2",
        "prob_0",
        "prob_1",
        "prob_2",
        "feature_us",
        "classifier_us",
        "total_us",
        "raw_line",
    ]

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser, out_path.open("w", newline="", encoding="utf-8") as f:
        ser.setDTR(False)
        ser.setRTS(False)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        time.sleep(2.0)
        ser.reset_input_buffer()
        for sent_i in range(n_packets_to_send):
            packet_i = start_packet + sent_i
            start = packet_i * PACKET_BYTES
            ser.write(payload[start : start + PACKET_BYTES])
            ser.flush()
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            parts = raw.split(",")
            while raw.startswith("#") or not raw or len(parts) < 16:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                parts = raw.split(",")
            row = {"raw_line": raw}
            if len(parts) >= 16:
                for name, value in zip(fieldnames[:-1], parts[:16]):
                    row[name] = value
            writer.writerow(row)
            if (sent_i + 1) % 10 == 0:
                print(f"{sent_i + 1}/{n_packets_to_send} windows")
    print(out_path)


if __name__ == "__main__":
    main()
