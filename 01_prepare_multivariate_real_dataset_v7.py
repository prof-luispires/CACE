#!/usr/bin/env python3
"""
01_prepare_multivariate_real_dataset.py
--------------------------------------
Prepare the DataVic real environmental IoT dataset for multivariate anomaly detection.

Input
-----
Raw CSV:
  sensor-readings-with-temperature-light-humidity-every-5-minutes-at-8-locations-t.csv

Output
------
real_multivariate/
  sensor_501_multivar.csv
  sensor_502_multivar.csv
  ...
  summary_per_sensor.csv
  metadata_nodes.csv
  global_summary.txt

Each exported sensor file contains:
  timestamp, boardid, location, temp_c, humidity, light, latitude, longitude, elevation

Rationale
---------
The original dataset contains real temperature, humidity and light readings from
multiple distributed sensor nodes. This script preserves the real temporal structure,
resamples to a regular 5-minute grid by default, and interpolates small gaps so that
the RL/MARL pipeline can operate on aligned time-series.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_CSV = "sensor-readings-with-temperature-light-humidity-every-5-minutes-at-8-locations-t.csv"
DEFAULT_OUT = "real_multivariate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare multivariate real IoT dataset.")
    p.add_argument("--csv", default=DEFAULT_CSV, help="Path to raw DataVic CSV file.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output folder.")
    p.add_argument("--resample-minutes", type=int, default=5, help="Regular sampling interval; use 0 to disable.")
    p.add_argument("--min-points", type=int, default=500, help="Minimum valid samples per sensor.")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    here = Path(__file__).parent.resolve()
    cand = here / p
    if cand.exists():
        return cand
    return p.resolve()


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "boardid", "temp_avg", "humidity_avg", "light_avg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    num_cols = [
        "boardid", "temp_avg", "humidity_avg", "light_avg",
        "latitude", "longitude", "elevation",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "boardid", "temp_avg", "humidity_avg", "light_avg"])
    return df.sort_values(["boardid", "timestamp"]).reset_index(drop=True)


def prepare_one_sensor(sdf: pd.DataFrame, resample_minutes: int) -> pd.DataFrame:
    keep = [
        "timestamp", "boardid", "location", "temp_avg", "humidity_avg", "light_avg",
        "latitude", "longitude", "elevation", "geolocation",
    ]
    keep = [c for c in keep if c in sdf.columns]
    out = sdf[keep].copy().sort_values("timestamp")
    out = out.rename(columns={
        "temp_avg": "temp_c",
        "humidity_avg": "humidity",
        "light_avg": "light",
    })

    # Physical plausibility screening before interpolation. Invalid raw values are
    # treated as missing rather than as clean environmental observations.
    out.loc[~out["humidity"].between(0.0, 100.0), "humidity"] = np.nan
    out.loc[out["light"] < 0.0, "light"] = np.nan
    out.loc[~out["temp_c"].between(-20.0, 60.0), "temp_c"] = np.nan

    if resample_minutes and resample_minutes > 0:
        static_cols = [c for c in ["boardid", "location", "latitude", "longitude", "elevation", "geolocation"] if c in out.columns]
        dyn_cols = ["temp_c", "humidity", "light"]
        indexed = out.set_index("timestamp")
        numeric = indexed[dyn_cols].resample(f"{resample_minutes}min").mean()
        numeric = numeric.interpolate(method="time", limit=3, limit_area="inside")
        rebuilt = numeric.reset_index()
        for col in static_cols:
            values = out[col].dropna()
            rebuilt[col] = values.iloc[0] if len(values) else np.nan
        out = rebuilt[["timestamp"] + static_cols + dyn_cols]

    return out.dropna(subset=["temp_c", "humidity", "light"]).reset_index(drop=True)


def summarize_sensor(df: pd.DataFrame) -> dict:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    row = {
        "boardid": int(df["boardid"].iloc[0]),
        "location": df["location"].iloc[0] if "location" in df.columns else None,
        "n_samples": int(len(df)),
        "start_time": ts.min(),
        "end_time": ts.max(),
        "median_step_minutes": float(diffs.median()) if len(diffs) else np.nan,
    }
    for col in ["temp_c", "humidity", "light"]:
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_std"] = float(df[col].std(ddof=0))
        row[f"{col}_min"] = float(df[col].min())
        row[f"{col}_max"] = float(df[col].max())
    return row


def main() -> None:
    args = parse_args()
    csv_path = resolve_path(args.csv)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset(csv_path)
    summary_rows = []
    metadata_rows = []
    exported = 0

    for boardid, sdf in raw.groupby("boardid"):
        prepared = prepare_one_sensor(sdf, args.resample_minutes)
        if len(prepared) < args.min_points:
            continue
        bid = int(boardid)
        out_csv = out_dir / f"sensor_{bid}_multivar.csv"
        prepared.to_csv(out_csv, index=False)
        summary_rows.append(summarize_sensor(prepared))
        metadata_rows.append({
            "boardid": bid,
            "location": prepared["location"].iloc[0] if "location" in prepared.columns else None,
            "latitude": prepared["latitude"].iloc[0] if "latitude" in prepared.columns else None,
            "longitude": prepared["longitude"].iloc[0] if "longitude" in prepared.columns else None,
            "elevation": prepared["elevation"].iloc[0] if "elevation" in prepared.columns else None,
            "source_file": out_csv.name,
        })
        exported += 1

    if not summary_rows:
        raise RuntimeError("No sensor files were exported. Check input file and min-points.")

    summary = pd.DataFrame(summary_rows).sort_values("boardid")
    metadata = pd.DataFrame(metadata_rows).sort_values("boardid")
    summary.to_csv(out_dir / "summary_per_sensor.csv", index=False)
    metadata.to_csv(out_dir / "metadata_nodes.csv", index=False)

    lines = [
        "Real multivariate IoT dataset summary",
        "======================================",
        f"Input file: {csv_path.name}",
        f"Raw rows: {len(raw)}",
        f"Exported sensors: {exported}",
        "Variables: temp_c, humidity, light",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"boardid={int(row['boardid'])}, n={int(row['n_samples'])}, "
            f"temp_mean={row['temp_c_mean']:.3f}, humidity_mean={row['humidity_mean']:.3f}, "
            f"light_mean={row['light_mean']:.3f}"
        )
    (out_dir / "global_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Exported {exported} multivariate sensor files to {out_dir}")
    print(f"[OK] Summary: {out_dir / 'summary_per_sensor.csv'}")


if __name__ == "__main__":
    main()
