#!/usr/bin/env python3
"""
Expert-based Cooperative MARL for multivariate IoT anomaly detection (v4).

Architecture:
    - Single-agent multivariate RL baseline: one agent selects one detector for all anomalies.
    - Expert agents:
        1) Spike expert
        2) Drift expert
        3) Flat-line expert
        4) Dropout expert
      All experts observe the same multivariate stream, but each is rewarded against its
      own anomaly-type label.
    - Learned coordinator: learns how to fuse expert decisions for the global anomaly label.

Why v4 is different:
    Cooperation is based on functional anomaly specialization, not on variable specialization.
    This is designed to demonstrate a clear gain over a single generic detector.

Input:
    real_multivariate_labeled/seed_42/sensor_*_multivar_labeled.csv

Output:
    expert_marl_results/seed_42_v4/
        overall_metrics.csv
        expert_metrics_per_sensor.csv
        coordinator_metrics_per_sensor.csv
        confusion_matrix_counts.csv
        ablation_metrics.csv
        histories/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARIABLES = ["temp_c", "humidity", "light"]
EXPERTS = ["spike", "drift", "flat", "dropout"]

# -----------------------------
# Metrics and reward
# -----------------------------
def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    reward = 0.60 * f1 + 0.30 * recall + 0.10 * precision - 0.20 * fpr
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "fnr": fnr, "reward": reward}

# -----------------------------
# Detection primitives
# -----------------------------
def rolling_median_flags(x: pd.Series, window=21, k=3.0):
    half = window // 2
    med = x.rolling(window, center=True, min_periods=half).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=half).median()
    sigma = 1.4826 * mad.replace(0, np.nan)
    return ((x - med).abs() > k * sigma).fillna(False).astype(int).to_numpy()


def rolling_z_flags(x: pd.Series, window=21, thr=3.0):
    half = window // 2
    mu = x.rolling(window, center=True, min_periods=half).mean()
    sd = x.rolling(window, center=True, min_periods=half).std(ddof=0).replace(0, np.nan)
    return (((x - mu).abs() / sd) > thr).fillna(False).astype(int).to_numpy()


def diff_z_flags(x: pd.Series, window=21, thr=3.0):
    d = x.diff().fillna(0.0)
    return rolling_z_flags(d, window=window, thr=thr)


def ewma_residual_flags(x: pd.Series, span=24, window=48, thr=2.5):
    baseline = x.ewm(span=span, adjust=False).mean()
    r = (x - baseline).abs()
    half = max(3, window // 2)
    mu = r.rolling(window, center=True, min_periods=half).mean()
    sd = r.rolling(window, center=True, min_periods=half).std(ddof=0).replace(0, np.nan)
    return ((r - mu) / sd > thr).fillna(False).astype(int).to_numpy()


def slope_flags(x: pd.Series, window=36, thr=2.2):
    # approximate rolling slope by difference over W samples
    s = (x - x.shift(window)) / max(window, 1)
    s = s.fillna(0.0).abs()
    return rolling_z_flags(s, window=max(21, window), thr=thr)


def mean_shift_flags(x: pd.Series, short=12, long=72, thr=2.0):
    s = x.rolling(short, min_periods=max(3, short//2)).mean()
    l = x.rolling(long, min_periods=max(6, long//2)).mean()
    r = (s - l).abs().fillna(0.0)
    return rolling_z_flags(r, window=long, thr=thr)


def flat_std_flags(x: pd.Series, window=18, quantile=0.03):
    sd = x.rolling(window, center=True, min_periods=max(3, window//2)).std(ddof=0)
    # avoid flagging naturally constant zero light at night too aggressively by requiring low diff too
    diff = x.diff().abs().rolling(window, center=True, min_periods=max(3, window//2)).mean()
    sd_thr = float(sd.quantile(quantile)) if sd.notna().any() else 0.0
    diff_thr = float(diff.quantile(quantile)) if diff.notna().any() else 0.0
    return ((sd <= sd_thr) & (diff <= diff_thr)).fillna(False).astype(int).to_numpy()


def low_range_flags(x: pd.Series, window=24, quantile=0.04):
    r = x.rolling(window, center=True, min_periods=max(3, window//2)).max() - x.rolling(window, center=True, min_periods=max(3, window//2)).min()
    thr = float(r.quantile(quantile)) if r.notna().any() else 0.0
    return (r <= thr).fillna(False).astype(int).to_numpy()


def dropout_flags(x: pd.Series, window=12):
    # repeated identical/near-identical values plus abrupt transition at segment boundary
    d = x.diff().abs().fillna(0.0)
    near_zero = d <= max(1e-6, float(d.quantile(0.02)))
    run = near_zero.rolling(window, center=True, min_periods=max(3, window//2)).mean()
    return (run > 0.85).fillna(False).astype(int).to_numpy()


def multivar_or(df: pd.DataFrame, func, **kwargs):
    flags = []
    for v in VARIABLES:
        if v in df.columns:
            flags.append(func(pd.to_numeric(df[v], errors="coerce").interpolate(limit=3, limit_area="inside").ffill(limit=1).bfill(limit=1), **kwargs))
    if not flags:
        return np.zeros(len(df), dtype=int)
    return np.maximum.reduce(flags).astype(int)

# -----------------------------
# Action spaces
# -----------------------------
ACTION_LIBRARY = {
    # spike-oriented
    "hampel_w21_k3": lambda df: multivar_or(df, rolling_median_flags, window=21, k=3.0),
    "hampel_w31_k2.5": lambda df: multivar_or(df, rolling_median_flags, window=31, k=2.5),
    "z_w21_t3": lambda df: multivar_or(df, rolling_z_flags, window=21, thr=3.0),
    "diffz_w21_t3": lambda df: multivar_or(df, diff_z_flags, window=21, thr=3.0),
    # drift-oriented
    "ewma_s24_w48_t2.5": lambda df: multivar_or(df, ewma_residual_flags, span=24, window=48, thr=2.5),
    "ewma_s48_w96_t2.2": lambda df: multivar_or(df, ewma_residual_flags, span=48, window=96, thr=2.2),
    "slope_w36_t2.2": lambda df: multivar_or(df, slope_flags, window=36, thr=2.2),
    "mean_shift_12_72": lambda df: multivar_or(df, mean_shift_flags, short=12, long=72, thr=2.0),
    # flat/dropout-oriented
    "flat_std_w18": lambda df: multivar_or(df, flat_std_flags, window=18, quantile=0.035),
    "flat_std_w36": lambda df: multivar_or(df, flat_std_flags, window=36, quantile=0.045),
    "low_range_w24": lambda df: multivar_or(df, low_range_flags, window=24, quantile=0.045),
    "dropout_w12": lambda df: multivar_or(df, dropout_flags, window=12),
    "dropout_w24": lambda df: multivar_or(df, dropout_flags, window=24),
}

EXPERT_ACTIONS = {
    "spike": ["hampel_w21_k3", "hampel_w31_k2.5", "z_w21_t3", "diffz_w21_t3"],
    "drift": ["ewma_s24_w48_t2.5", "ewma_s48_w96_t2.2", "slope_w36_t2.2", "mean_shift_12_72"],
    "flat": ["flat_std_w18", "flat_std_w36", "low_range_w24", "dropout_w12"],
    "dropout": ["dropout_w12", "dropout_w24", "flat_std_w18", "low_range_w24"],
}

SINGLE_ACTIONS = list(ACTION_LIBRARY.keys())
COORD_ACTIONS = ["or_all", "top2_or", "majority", "weighted_025", "weighted_035", "best_expert"]

# -----------------------------
# Data loading
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="real_multivariate_labeled/seed_42")
    p.add_argument("--out", default="expert_marl_results/seed_42_v4")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.90)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_files(data_dir: Path):
    files = sorted(data_dir.glob("sensor_*_multivar_labeled.csv"))
    if not files:
        files = sorted(data_dir.glob("*.csv"))
        files = [f for f in files if "summary" not in f.name.lower()]
    if not files:
        raise FileNotFoundError(f"No labeled sensor files in {data_dir}")
    return files


def load_df(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for v in VARIABLES:
        df[v] = pd.to_numeric(df[v], errors="coerce").interpolate(limit=3, limit_area="inside").ffill(limit=1).bfill(limit=1)
    for t in EXPERTS:
        col = f"label_global_{t}"
        if col not in df.columns:
            # fallback from variable labels
            sub = [f"label_{v}_{t}" for v in VARIABLES if f"label_{v}_{t}" in df.columns]
            df[col] = df[sub].max(axis=1) if sub else 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "label_global" not in df.columns:
        df["label_global"] = df[[f"label_global_{t}" for t in EXPERTS]].max(axis=1)
    df["label_global"] = pd.to_numeric(df["label_global"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=VARIABLES).reset_index(drop=True)


def sensor_id_from_path(p: Path):
    parts = p.stem.split("_")
    for part in parts:
        if part.isdigit():
            return part
    return p.stem

# -----------------------------
# Q-learning helpers
# -----------------------------
def train_q_over_sensors(files, actions, label_getter, args, rng, prefix="agent"):
    n_states = len(files)
    n_actions = len(actions)
    Q = np.zeros((n_states, n_actions), dtype=float)
    eps = args.epsilon_start
    history = []
    cache = {i: load_df(f) for i, f in enumerate(files)}
    pred_cache = {}
    for ep in range(args.episodes):
        s = int(rng.integers(0, n_states))
        df = cache[s]
        a = int(rng.integers(0, n_actions)) if rng.random() < eps else int(np.argmax(Q[s, :]))
        action_name = actions[a]
        key = (s, action_name)
        if key not in pred_cache:
            pred_cache[key] = ACTION_LIBRARY[action_name](df)
        y_pred = pred_cache[key]
        y_true = label_getter(df)
        m = metrics(y_true, y_pred)
        old = Q[s, a]
        Q[s, a] = (1 - args.alpha) * old + args.alpha * (m["reward"] + args.gamma * np.max(Q[s, :]))
        history.append({"episode": ep+1, "state": s, "sensor": sensor_id_from_path(files[s]), "action": action_name, **m, "epsilon": eps, "agent": prefix})
        eps = max(args.epsilon_end, eps * args.epsilon_decay)
    return Q, pd.DataFrame(history), cache


def best_predictions(Q, files, actions, cache):
    rows = []
    preds = {}
    for s, f in enumerate(files):
        df = cache[s]
        best_a = int(np.argmax(Q[s, :]))
        action_name = actions[best_a]
        y_pred = ACTION_LIBRARY[action_name](df)
        preds[s] = y_pred
        rows.append({"state": s, "sensor": sensor_id_from_path(f), "best_action": best_a, "best_detector": action_name, **{f"Q_{i}": Q[s, i] for i in range(Q.shape[1])}})
    return preds, pd.DataFrame(rows)

# -----------------------------
# Coordinator
# -----------------------------
def fuse_predictions(action: str, expert_preds: dict[str, np.ndarray], expert_weights: dict[str, float]) -> np.ndarray:
    arr = np.vstack([expert_preds[e].astype(float) for e in EXPERTS])
    if action == "or_all":
        return (arr.max(axis=0) > 0).astype(int)
    if action == "top2_or":
        top = sorted(EXPERTS, key=lambda e: expert_weights.get(e, 0.0), reverse=True)[:2]
        return np.maximum.reduce([expert_preds[e] for e in top]).astype(int)
    if action == "majority":
        return (arr.sum(axis=0) >= 2).astype(int)
    if action == "weighted_025":
        score = sum(expert_weights[e] * expert_preds[e] for e in EXPERTS)
        return (score >= 0.25).astype(int)
    if action == "weighted_035":
        score = sum(expert_weights[e] * expert_preds[e] for e in EXPERTS)
        return (score >= 0.35).astype(int)
    if action == "best_expert":
        best = max(EXPERTS, key=lambda e: expert_weights.get(e, 0.0))
        return expert_preds[best].astype(int)
    raise ValueError(action)


def evaluate_expert_set(files, cache, expert_pred_by_sensor, expert_weight_by_sensor):
    # train coordinator Q: state=sensor, action=fusion rule
    n_states = len(files)
    n_actions = len(COORD_ACTIONS)
    Q = np.zeros((n_states, n_actions), dtype=float)
    rng = np.random.default_rng(123)
    eps = 1.0
    hist = []
    for ep in range(500):
        s = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions)) if rng.random() < eps else int(np.argmax(Q[s, :]))
        df = cache[s]
        pred = fuse_predictions(COORD_ACTIONS[a], expert_pred_by_sensor[s], expert_weight_by_sensor[s])
        m = metrics(df["label_global"].to_numpy(), pred)
        Q[s, a] = (1 - 0.2) * Q[s, a] + 0.2 * (m["reward"] + 0.9 * np.max(Q[s, :]))
        hist.append({"episode": ep+1, "state": s, "sensor": sensor_id_from_path(files[s]), "coord_action": COORD_ACTIONS[a], **m, "epsilon": eps})
        eps = max(0.05, eps * 0.995)

    rows = []
    pred_final = {}
    for s, f in enumerate(files):
        action = COORD_ACTIONS[int(np.argmax(Q[s, :]))]
        pred = fuse_predictions(action, expert_pred_by_sensor[s], expert_weight_by_sensor[s])
        pred_final[s] = pred
        m = metrics(cache[s]["label_global"].to_numpy(), pred)
        rows.append({"state": s, "sensor": sensor_id_from_path(f), "coord_action": action, **m})
    return pred_final, pd.DataFrame(rows), pd.DataFrame(hist)


def aggregate_metrics(method_name, files, cache, preds):
    y_true_all = []
    y_pred_all = []
    per_rows = []
    for s, f in enumerate(files):
        y_true = cache[s]["label_global"].to_numpy()
        y_pred = preds[s]
        m = metrics(y_true, y_pred)
        per_rows.append({"method": method_name, "state": s, "sensor": sensor_id_from_path(f), **m})
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    overall = metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    overall["method"] = method_name
    return overall, per_rows


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(exist_ok=True)
    files = get_files(data_dir)
    print(f"[INFO] Found {len(files)} labeled files.")

    # Single-agent baseline
    Qs, hist_s, cache = train_q_over_sensors(files, SINGLE_ACTIONS, lambda df: df["label_global"].to_numpy(), args, rng, prefix="single")
    single_preds, single_summary = best_predictions(Qs, files, SINGLE_ACTIONS, cache)
    hist_s.to_csv(out_dir / "histories" / "single_agent_history.csv", index=False)
    single_summary.to_csv(out_dir / "single_agent_summary.csv", index=False)

    # Experts
    expert_preds_by_sensor = {s: {} for s in range(len(files))}
    expert_weights_by_sensor = {s: {} for s in range(len(files))}
    expert_summary_rows = []
    all_hist = []
    for expert in EXPERTS:
        actions = EXPERT_ACTIONS[expert]
        Qe, hist_e, _ = train_q_over_sensors(files, actions, lambda df, e=expert: df[f"label_global_{e}"].to_numpy(), args, rng, prefix=expert)
        preds_e, summary_e = best_predictions(Qe, files, actions, cache)
        hist_e.to_csv(out_dir / "histories" / f"{expert}_expert_history.csv", index=False)
        all_hist.append(hist_e)
        for s, f in enumerate(files):
            df = cache[s]
            pred = preds_e[s]
            y_type = df[f"label_global_{expert}"].to_numpy()
            type_metrics = metrics(y_type, pred)
            global_metrics = metrics(df["label_global"].to_numpy(), pred)
            expert_preds_by_sensor[s][expert] = pred
            # weight based on type-specific F1, with a small floor so no expert disappears
            expert_weights_by_sensor[s][expert] = 0.05 + type_metrics["f1"]
            row = {"expert": expert, "state": s, "sensor": sensor_id_from_path(f),
                   "best_detector": summary_e.loc[summary_e["state"] == s, "best_detector"].iloc[0],
                   **{f"type_{k}": v for k, v in type_metrics.items()},
                   **{f"global_{k}": v for k, v in global_metrics.items()}}
            expert_summary_rows.append(row)
    expert_summary = pd.DataFrame(expert_summary_rows)
    expert_summary.to_csv(out_dir / "expert_metrics_per_sensor.csv", index=False)
    if all_hist:
        pd.concat(all_hist, ignore_index=True).to_csv(out_dir / "histories" / "all_expert_histories.csv", index=False)

    # normalize weights per sensor
    for s in expert_weights_by_sensor:
        total = sum(expert_weights_by_sensor[s].values())
        if total <= 0:
            expert_weights_by_sensor[s] = {e: 1/len(EXPERTS) for e in EXPERTS}
        else:
            expert_weights_by_sensor[s] = {e: expert_weights_by_sensor[s][e] / total for e in EXPERTS}

    # Coordinator learned fusion
    coord_preds, coord_per, coord_hist = evaluate_expert_set(files, cache, expert_preds_by_sensor, expert_weights_by_sensor)
    coord_per.to_csv(out_dir / "coordinator_metrics_per_sensor.csv", index=False)
    coord_hist.to_csv(out_dir / "histories" / "coordinator_history.csv", index=False)

    # Fixed fusions and ablations
    fusion_preds = {"Single-agent RL": single_preds, "Learned coordinator": coord_preds}
    for action in ["or_all", "top2_or", "majority", "weighted_025", "weighted_035"]:
        fixed = {}
        for s in range(len(files)):
            fixed[s] = fuse_predictions(action, expert_preds_by_sensor[s], expert_weights_by_sensor[s])
        fusion_preds[action] = fixed

    # Oracle fusion from available coordinator actions per sensor (diagnostic upper bound)
    oracle = {}
    oracle_rows = []
    for s, f in enumerate(files):
        best_m = None; best_pred = None; best_action = None
        for action in COORD_ACTIONS:
            pred = fuse_predictions(action, expert_preds_by_sensor[s], expert_weights_by_sensor[s])
            m = metrics(cache[s]["label_global"].to_numpy(), pred)
            if best_m is None or m["f1"] > best_m["f1"]:
                best_m = m; best_pred = pred; best_action = action
        oracle[s] = best_pred
        oracle_rows.append({"state": s, "sensor": sensor_id_from_path(f), "oracle_action": best_action, **best_m})
    fusion_preds["Oracle fusion"] = oracle
    pd.DataFrame(oracle_rows).to_csv(out_dir / "oracle_fusion_per_sensor.csv", index=False)

    # Ablation: remove each expert from OR fusion
    ablation_rows = []
    for removed in ["none"] + EXPERTS:
        preds = {}
        active = [e for e in EXPERTS if e != removed]
        for s in range(len(files)):
            if not active:
                preds[s] = np.zeros(len(cache[s]), dtype=int)
            else:
                preds[s] = np.maximum.reduce([expert_preds_by_sensor[s][e] for e in active]).astype(int)
        overall, _ = aggregate_metrics(f"ablation_without_{removed}" if removed != "none" else "all_experts_or", files, cache, preds)
        ablation_rows.append(overall)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "ablation_metrics.csv", index=False)

    # Exports overall/per-sensor/confusion
    overall_rows = []
    per_rows = []
    conf_rows = []
    for name, preds in fusion_preds.items():
        overall, per = aggregate_metrics(name, files, cache, preds)
        overall_rows.append(overall)
        per_rows.extend(per)
        conf_rows.append({"method": name, "tp": overall["tp"], "tn": overall["tn"], "fp": overall["fp"], "fn": overall["fn"]})
    pd.DataFrame(overall_rows).to_csv(out_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(per_rows).to_csv(out_dir / "cooperative_metrics_per_sensor.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(out_dir / "confusion_matrix_counts.csv", index=False)

    # weights
    weight_rows = []
    for s, f in enumerate(files):
        row = {"state": s, "sensor": sensor_id_from_path(f)}
        row.update({f"w_{e}": expert_weights_by_sensor[s][e] for e in EXPERTS})
        weight_rows.append(row)
    pd.DataFrame(weight_rows).to_csv(out_dir / "expert_weights.csv", index=False)

    print("[OK] Results saved to:", out_dir)
    print(pd.DataFrame(overall_rows)[["method", "accuracy", "precision", "recall", "f1", "reward"]])


if __name__ == "__main__":
    main()
