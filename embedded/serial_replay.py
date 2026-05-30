from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "embedded" / "model_export" / "logreg_replay_model.json"
DEFAULT_VECTORS = ROOT / "embedded" / "model_export" / "logreg_replay_vectors.csv"
DEFAULT_LOG_DIR = ROOT / "embedded" / "logs"


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream held-out EEG feature vectors to ESP32 over serial.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the first payload and exit without opening serial.")
    args = parser.parse_args()

    model_package = json.loads(args.model.read_text(encoding="utf-8"))
    feature_names = [str(x) for x in model_package["feature_names"]]
    out_path = args.out or (DEFAULT_LOG_DIR / f"esp32_replay_{_timestamp()}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with args.vectors.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        raise SystemExit("No replay rows found.")

    def payload(row: dict[str, str]) -> str:
        values = [row["replay_row_index"], row["model_index"]]
        values.extend(row[name] if row[name] else "nan" for name in feature_names)
        return ",".join(values) + "\n"

    if args.dry_run:
        print(payload(rows[0])[:1000])
        return

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for serial replay: python -m pip install pyserial") from exc

    fieldnames = [
        "replay_row_index",
        "model_index",
        "pred_label",
        "logit_0",
        "logit_1",
        "logit_2",
        "prob_0",
        "prob_1",
        "prob_2",
        "inference_us",
        "raw_line",
    ]
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser, out_path.open("w", newline="", encoding="utf-8") as f:
        ser.setDTR(False)
        ser.setRTS(False)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        time.sleep(2.0)
        ser.reset_input_buffer()
        for i, row in enumerate(rows, start=1):
            ser.write(payload(row).encode("utf-8"))
            ser.flush()
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            while raw.startswith("#") or not raw:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
            parts = raw.split(",")
            parsed = {"raw_line": raw}
            if len(parts) >= 10:
                parsed.update(
                    {
                        "replay_row_index": parts[0],
                        "model_index": parts[1],
                        "pred_label": parts[2],
                        "logit_0": parts[3],
                        "logit_1": parts[4],
                        "logit_2": parts[5],
                        "prob_0": parts[6],
                        "prob_1": parts[7],
                        "prob_2": parts[8],
                        "inference_us": parts[9],
                    }
                )
            writer.writerow(parsed)
            if i % 100 == 0:
                print(f"{i}/{len(rows)} rows")
    print(out_path)


if __name__ == "__main__":
    main()
