from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "embedded" / "window_replay" / "data" / "window_replay_reference.csv"
DEFAULT_OUT = ROOT / "embedded" / "window_replay" / "data" / "spectral_reference_fixture.json"


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    return float(np.trapz(psd[mask], freqs[mask]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Python spectral fixture values for ESP32 Welch debugging.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--row", type=int, default=None, help="Replay row index; default uses first reference row.")
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    row = ref.iloc[0] if args.row is None else ref.loc[ref["replay_row_index"] == args.row].iloc[0]
    npz = np.load(row["npz_path"])
    data = npz["data"].astype(np.float64)
    fs = 250.0
    freqs, psd = signal.welch(data, fs=fs, nperseg=500, axis=1)
    channel = 1  # Fz in the fixed channel order.
    fixture = {
        "replay_row_index": int(row["replay_row_index"]),
        "npz_path": row["npz_path"],
        "scipy_welch": {
            "window": "hann",
            "nperseg": 500,
            "noverlap": 250,
            "nfft": 500,
            "detrend": "constant",
            "scaling": "density",
            "return_onesided": True,
            "average": "mean",
            "frequency_step_hz": float(freqs[1] - freqs[0]),
        },
        "channel_index": channel,
        "channel_name": "Fz",
        "freqs_1_to_40_hz": freqs[(freqs >= 1.0) & (freqs < 40.0)].tolist(),
        "psd_1_to_40_hz": psd[channel, (freqs >= 1.0) & (freqs < 40.0)].tolist(),
        "band_power": {
            "delta_1_4": band_power(psd[channel], freqs, 1.0, 4.0),
            "theta_4_8": band_power(psd[channel], freqs, 4.0, 8.0),
            "alpha_8_13": band_power(psd[channel], freqs, 8.0, 13.0),
            "beta_13_30": band_power(psd[channel], freqs, 13.0, 30.0),
            "highbeta_30_40": band_power(psd[channel], freqs, 30.0, 40.0),
            "total_1_40": band_power(psd[channel], freqs, 1.0, 40.0),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
