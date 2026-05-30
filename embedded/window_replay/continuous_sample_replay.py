from __future__ import annotations

import argparse
import csv
import struct
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKETS = ROOT / "embedded" / "window_replay" / "data" / "continuous_replay_samples.bin"
DEFAULT_REFERENCE = ROOT / "embedded" / "window_replay" / "data" / "continuous_replay_reference.csv"
DEFAULT_LOG_DIR = ROOT / "embedded" / "window_replay" / "logs"
STREAM_MAGIC = 0x32535745
SAMPLE_MAGIC = 0x32535753
STREAM_HEADER_BYTES = 80
SAMPLE_PACKET_BYTES = 92


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _expected_windows(reference_path: Path) -> int:
    with reference_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _iter_packets(payload: bytes):
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise ValueError(f"Truncated packet magic at offset {offset}")
        (magic,) = struct.unpack_from("<I", payload, offset)
        if magic == STREAM_MAGIC:
            size = STREAM_HEADER_BYTES
            packet_type = "header"
        elif magic == SAMPLE_MAGIC:
            size = SAMPLE_PACKET_BYTES
            packet_type = "sample"
        else:
            raise ValueError(f"Unknown packet magic 0x{magic:08x} at offset {offset}")
        if offset + size > len(payload):
            raise ValueError(f"Truncated {packet_type} packet at offset {offset}")
        yield packet_type, payload[offset : offset + size]
        offset += size


def _drain_serial_lines(ser, line_buffer: bytearray) -> list[str]:
    available = ser.in_waiting
    if available:
        line_buffer.extend(ser.read(available))
    lines: list[str] = []
    while True:
        try:
            newline_index = line_buffer.index(10)
        except ValueError:
            break
        raw = bytes(line_buffer[:newline_index]).decode("utf-8", errors="replace").strip()
        del line_buffer[: newline_index + 1]
        lines.append(raw)
    return lines


def _write_log_line(raw: str, fieldnames: list[str], writer: csv.DictWriter) -> bool:
    parts = raw.split(",")
    if raw.startswith("#") or not raw or len(parts) < 18:
        return False
    row = {"raw_line": raw}
    for name, value in zip(fieldnames[:-1], parts[:18]):
        row[name] = value
    writer.writerow(row)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Timed serial replay of preprocessed 19-channel EEG samples to ESP32.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample-rate", type=float, default=250.0)
    parser.add_argument("--speedup", type=float, default=1.0, help="Use 1.0 for real-time 250 Hz replay; >1 sends faster.")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: python -m pip install pyserial") from exc

    packets = list(_iter_packets(args.packets.read_bytes()))
    expected_windows = _expected_windows(args.reference)
    out_path = args.out or DEFAULT_LOG_DIR / f"esp32_continuous_replay_{_timestamp()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stream_id",
        "window_index",
        "start_sample_index",
        "end_sample_index",
        "model_index",
        "pred_label",
        "logit_0",
        "logit_1",
        "logit_2",
        "prob_0",
        "prob_1",
        "prob_2",
        "feature_us",
        "classifier_us",
        "total_us",
        "malformed_packets",
        "dropped_sequences",
        "buffer_overrun_count",
        "raw_line",
    ]

    sample_interval = 1.0 / (args.sample_rate * args.speedup)
    sample_count = 0
    rows_seen = 0

    with serial.Serial(args.port, args.baud, timeout=0, write_timeout=5.0) as ser, out_path.open("w", newline="", encoding="utf-8") as f:
        ser.setDTR(False)
        ser.setRTS(False)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        time.sleep(2.0)
        ser.reset_input_buffer()

        next_sample_time = time.perf_counter()
        line_buffer = bytearray()
        for packet_type, packet in packets:
            if packet_type == "sample":
                now = time.perf_counter()
                if now < next_sample_time:
                    time.sleep(next_sample_time - now)
                elif now - next_sample_time > sample_interval:
                    next_sample_time = now
                next_sample_time += sample_interval
                sample_count += 1
            else:
                next_sample_time = time.perf_counter()

            ser.write(packet)

            for raw in _drain_serial_lines(ser, line_buffer):
                if _write_log_line(raw, fieldnames, writer):
                    rows_seen += 1
                    if rows_seen % 10 == 0:
                        print(f"{rows_seen}/{expected_windows} windows")

        ser.flush()
        deadline = time.time() + args.timeout
        while rows_seen < expected_windows and time.time() < deadline:
            for raw in _drain_serial_lines(ser, line_buffer):
                if _write_log_line(raw, fieldnames, writer):
                    rows_seen += 1
                    if rows_seen % 10 == 0:
                        print(f"{rows_seen}/{expected_windows} windows")
            time.sleep(0.005)

    print(f"samples_sent={sample_count}")
    print(f"windows_logged={rows_seen}/{expected_windows}")
    print(out_path)


if __name__ == "__main__":
    main()
