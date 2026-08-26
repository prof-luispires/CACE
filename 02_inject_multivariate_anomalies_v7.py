#!/usr/bin/env python3
"""
Inject type-labelled anomalies into real multivariate IoT environmental time-series.

Version v4 for the paper:
    Expert-Agent Cooperative Reinforcement Learning / Expert-MARL

Input folder:
    real_multivariate/
        sensor_501_multivar.csv, ...

Expected columns:
    timestamp, temp_c, humidity, light

Output folder:
    real_multivariate_labeled/seed_<SEED>/
        sensor_501_multivar_labeled.csv, ...
        summary_labeled.csv

Main additions over previous versions:
    - physical limits: humidity in [0,100], light >= 0
    - explicit anomaly-type labels for each variable and globally:
        spike, drift, flat, dropout
    - a global binary label and a compact dominant anomaly type label

The original real signal is preserved in <var>_clean columns.
Only controlled anomalies are injected, so quantitative evaluation is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARIABLES = ["temp_c", "humidity", "light"]
TYPES = ["spike", "drift", "flat", "dropout"]
TYPE_CODE = {"normal": 0, "spike": 1, "drift": 2, "flat": 3, "dropout": 4, "mixed": 5}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="real_multivariate", help="Folder with clean multivariate sensor CSVs")
    p.add_argument("--out", default="real_multivariate_labeled", help="Output root folder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=0, help="0 = all files")
    p.add_argument("--spike-frac", type=float, default=0.010)
    p.add_argument("--drift-frac", type=float, default=0.070)
    p.add_argument("--flat-frac", type=float, default=0.035)
    p.add_argument("--dropout-frac", type=float, default=0.030)
    p.add_argument("--correlated-frac", type=float, default=0.015, help="small fraction of multivariate events")
    return p.parse_args()


def infer_files(in_dir: Path, max_files: int = 0):
    patterns = ["sensor_*_multivar.csv", "sensor_*_real.csv", "*.csv"]
    files = []
    for pat in patterns:
        files = sorted(in_dir.glob(pat))
        if files:
            break
    files = [f for f in files if "summary" not in f.name.lower() and "metadata" not in f.name.lower()]
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No sensor CSV files found in {in_dir}")
    return files


def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {"temp_avg": "temp_c", "humidity_avg": "humidity", "light_avg": "light"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = [v for v in VARIABLES if v not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}. Found: {df.columns.tolist()}")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.RangeIndex(len(df))
    for v in VARIABLES:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.dropna(subset=VARIABLES).reset_index(drop=True)
    return df


def clamp_variable(name: str, values: np.ndarray) -> np.ndarray:
    y = values.copy()
    if name == "humidity":
        y = np.clip(y, 0.0, 100.0)
    elif name == "light":
        y = np.maximum(y, 0.0)
    return y


def safe_segment(n: int, length: int, rng: np.random.Generator):
    length = int(max(2, min(length, n)))
    if n <= length + 1:
        return 0, n
    start = int(rng.integers(0, n - length))
    return start, start + length


def inject_for_variable(y_clean: np.ndarray, var: str, rng: np.random.Generator, args) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(y_clean)
    y = y_clean.astype(float).copy()
    labels = {t: np.zeros(n, dtype=int) for t in TYPES}
    std = float(np.nanstd(y_clean))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    data_range = float(np.nanpercentile(y_clean, 95) - np.nanpercentile(y_clean, 5))
    if not np.isfinite(data_range) or data_range < 1e-6:
        data_range = std

    # 1) isolated spikes
    k = max(1, int(n * args.spike_frac))
    idx = rng.choice(n, size=min(k, n), replace=False)
    for i in idx:
        sign = -1 if rng.random() < 0.5 else 1
        # variable-aware amplitude
        amp = rng.uniform(2.5, 5.5) * std
        if var == "light":
            amp = rng.uniform(0.15, 0.45) * max(data_range, 1.0)
        elif var == "humidity":
            amp = rng.uniform(8.0, 25.0)
        y[i] += sign * amp
        labels["spike"][i] = 1

    # 2) drift segment
    length = max(12, int(n * args.drift_frac))
    start, end = safe_segment(n, length, rng)
    if var == "humidity":
        amp = rng.uniform(8.0, 22.0)
    elif var == "light":
        amp = rng.uniform(0.15, 0.40) * max(data_range, 1.0)
    else:
        amp = rng.uniform(1.5, 4.0) * std
    if rng.random() < 0.5:
        amp = -amp
    y[start:end] += np.linspace(0.0, amp, end - start)
    labels["drift"][start:end] = 1

    # 3) flat-line / frozen segment
    length = max(8, int(n * args.flat_frac))
    start, end = safe_segment(n, length, rng)
    ref = float(y[max(0, start - 1)]) if start > 0 else float(np.nanmedian(y))
    noise = 0.002 * max(std, 1.0)
    y[start:end] = ref + rng.normal(0.0, noise, size=end - start)
    labels["flat"][start:end] = 1

    # 4) dropout / communication loss: hold previous or abrupt zero/near-zero for light
    length = max(8, int(n * args.dropout_frac))
    start, end = safe_segment(n, length, rng)
    if var == "light" and rng.random() < 0.5:
        value = 0.0
    else:
        value = float(y[start - 1]) if start > 0 else float(np.nanmedian(y))
    y[start:end] = value
    labels["dropout"][start:end] = 1

    y = clamp_variable(var, y)
    return y, labels


def add_correlated_events(df_out: pd.DataFrame, rng: np.random.Generator, frac: float) -> pd.DataFrame:
    """Inject a few multivariate events affecting several variables simultaneously."""
    n = len(df_out)
    if frac <= 0 or n < 50:
        return df_out
    length = max(8, int(n * frac))
    start, end = safe_segment(n, length, rng)
    # correlated environmental-like drift: temp up, humidity down, light perturbation
    df_out.loc[start:end-1, "temp_c"] = df_out.loc[start:end-1, "temp_c"].to_numpy() + np.linspace(0, 2.0, end-start)
    df_out.loc[start:end-1, "humidity"] = np.clip(df_out.loc[start:end-1, "humidity"].to_numpy() - np.linspace(0, 8.0, end-start), 0, 100)
    df_out.loc[start:end-1, "light"] = np.maximum(df_out.loc[start:end-1, "light"].to_numpy() + np.linspace(0, 0.15*np.nanmax(df_out["light"]), end-start), 0)
    for v in VARIABLES:
        df_out.loc[start:end-1, f"label_{v}_drift"] = 1
        df_out.loc[start:end-1, f"label_{v}"] = 1
    df_out.loc[start:end-1, "injected_correlated_event"] = 1
    return df_out


def dominant_type(row) -> int:
    active = [t for t in TYPES if row.get(f"label_global_{t}", 0) == 1]
    if not active:
        return TYPE_CODE["normal"]
    if len(active) > 1:
        return TYPE_CODE["mixed"]
    return TYPE_CODE[active[0]]


def process_file(path: Path, out_dir: Path, seed: int, args) -> dict:
    df = load_sensor_csv(path)
    rng = np.random.default_rng(seed)
    out = df.copy()
    if "boardid" not in out.columns:
        # infer boardid from filename if possible
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        out["boardid"] = int(digits[:3]) if digits else path.stem

    out["injected_correlated_event"] = 0
    for v in VARIABLES:
        clean = out[v].to_numpy(dtype=float)
        out[f"{v}_clean"] = clean
        injected, labels = inject_for_variable(clean, v, rng, args)
        out[v] = injected
        for t in TYPES:
            out[f"label_{v}_{t}"] = labels[t]
        out[f"label_{v}"] = np.maximum.reduce([labels[t] for t in TYPES]).astype(int)

    out = add_correlated_events(out, rng, args.correlated_frac)

    for t in TYPES:
        cols = [f"label_{v}_{t}" for v in VARIABLES]
        out[f"label_global_{t}"] = out[cols].max(axis=1).astype(int)
    out["label_global"] = out[[f"label_global_{t}" for t in TYPES]].max(axis=1).astype(int)
    out["anomaly_type_code"] = out.apply(dominant_type, axis=1).astype(int)

    # final physical limits
    out["humidity"] = np.clip(out["humidity"], 0.0, 100.0)
    out["light"] = np.maximum(out["light"], 0.0)

    sensor_id = str(out["boardid"].iloc[0])
    out_name = f"sensor_{sensor_id}_multivar_labeled.csv"
    out.to_csv(out_dir / out_name, index=False)

    row = {"sensor": sensor_id, "file": out_name, "n_samples": len(out)}
    row["global_anomaly_pct"] = 100 * float(out["label_global"].mean())
    for t in TYPES:
        row[f"global_{t}_pct"] = 100 * float(out[f"label_global_{t}"].mean())
    for v in VARIABLES:
        row[f"{v}_anomaly_pct"] = 100 * float(out[f"label_{v}"].mean())
    return row


def main():
    args = parse_args()
    in_dir = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_dir = out_root / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = infer_files(in_dir, args.max_files)
    rows = []
    rng_master = np.random.default_rng(args.seed)
    for f in files:
        file_seed = int(rng_master.integers(0, 2**31 - 1))
        try:
            row = process_file(f, out_dir, file_seed, args)
            row["seed"] = file_seed
            rows.append(row)
            print(f"[OK] {f.name} -> {row['file']} | global anomalies={row['global_anomaly_pct']:.2f}%")
        except Exception as exc:
            print(f"[SKIP] {f.name}: {exc}")
    if not rows:
        raise RuntimeError("No files processed.")
    pd.DataFrame(rows).to_csv(out_dir / "summary_labeled.csv", index=False)
    print(f"[OK] Labeled files saved to: {out_dir}")


if __name__ == "__main__":
    main()
