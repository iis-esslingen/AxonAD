#!/usr/bin/env python3
"""Sensitivity sweep for AxonAD on the TSB-AD-M tuning split.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from TSB_AD.model_wrapper import run_AxonAD


@dataclass(frozen=True)
class Setting:
    name: str
    params: dict


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_train_index(filename: str) -> int:
    # Matches ..._tr_<N>_1st_...
    parts = filename.split("_")
    try:
        return int(parts[-3])
    except Exception:
        raise ValueError(f"Could not parse train split index from filename: {filename}")


def _load_series(dataset_dir: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(dataset_dir, filename)
    df = pd.read_csv(path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    return data, label


def _oat_settings(base: dict, values_by_key: dict[str, list]) -> list[Setting]:
    settings = [Setting(name="base", params=dict(base))]
    for k, vals in values_by_key.items():
        for v in vals:
            p = dict(base)
            p[k] = v
            settings.append(Setting(name=f"{k}={v}", params=p))
    return settings


def _grid_settings(base: dict, values_by_key: dict[str, list]) -> list[Setting]:
    keys = [k for k, vals in values_by_key.items() if vals]
    if not keys:
        return [Setting(name="base", params=dict(base))]

    grid = []
    for combo in itertools.product(*[values_by_key[k] for k in keys]):
        p = dict(base)
        name_parts = []
        for k, v in zip(keys, combo):
            p[k] = v
            name_parts.append(f"{k}={v}")
        grid.append(Setting(name="__".join(name_parts), params=p))

    return [Setting(name="base", params=dict(base))] + grid


def _aggregate(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    agg = {}
    for m in metric_cols:
        agg[m + "_mean"] = (m, "mean")
        agg[m + "_std"] = (m, "std")
        agg[m + "_count"] = (m, "count")

    return (
        df.groupby(["setting"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["setting"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensitivity sweep for AxonAD on TSB-AD-M tuning.")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        "--dataset_dir",
        default=os.path.join(repo_root, "Datasets/TSB-AD-M"),
        help="Directory containing TSB-AD-M CSV files",
    )
    ap.add_argument(
        "--file_list",
        default=os.path.join(repo_root, "Datasets/File_List/TSB-AD-M-Tuning.csv"),
        help="CSV listing tuning filenames (column file_name)",
    )
    ap.add_argument("--out_dir", default=os.path.join(repo_root, "outputs_sweeps"), help="Output folder")
    ap.add_argument("--mode", choices=["oat", "grid"], default="oat", help="OAT is default; grid can be expensive")

    # Sweep values (defaults chosen to be small but informative)
    ap.add_argument("--win_size", nargs="*", type=int, default=[], help="Override sweep values for win_size")
    ap.add_argument("--d_model", nargs="*", type=int, default=[], help="Override sweep values for d_model")
    ap.add_argument("--num_heads", nargs="*", type=int, default=[], help="Override sweep values for num_heads")
    ap.add_argument("--lr", nargs="*", type=float, default=[], help="Override sweep values for lr")
    ap.add_argument("--kl_tail_k", nargs="*", type=int, default=[3, 5, 10, 20], help="Sweep values for kl_tail_k")
    ap.add_argument(
        "--forecast_steps",
        nargs="*",
        type=int,
        default=[1, 3, 5, 25],
        help="Sweep values for forecast_steps",
    )

    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of series for quicker runs (0 = all)")
    ap.add_argument("--device", default="", help="Optional: force torch device string, e.g. cpu / mps / cuda")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device:
        torch.device(args.device)  # validates

    # Base params from repo optimal list
    base = dict(Optimal_Multi_algo_HP_dict.get("AxonAD", {}))
    if not base:
        raise SystemExit("Optimal_Multi_algo_HP_dict['AxonAD'] not found")

    # Build sweep dict; if user supplies list, use it; else only include defaults for keys we want to probe
    values_by_key = {
        "win_size": args.win_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "kl_tail_k": args.kl_tail_k,
        "forecast_steps": args.forecast_steps,
    }

    # Remove empty overrides except the two core defaults we want to sweep
    values_by_key = {k: v for k, v in values_by_key.items() if v}

    settings = _oat_settings(base, values_by_key) if args.mode == "oat" else _grid_settings(base, values_by_key)

    file_list = pd.read_csv(args.file_list)["file_name"].astype(str).tolist()
    if args.max_files and args.max_files > 0:
        file_list = file_list[: args.max_files]

    metric_cols = [
        "AUC-PR",
        "AUC-ROC",
        "VUS-PR",
        "VUS-ROC",
        "Standard-F1",
        "PA-F1",
        "Event-based-F1",
        "R-based-F1",
        "Affiliation-F",
    ]

    rows = []
    for setting in settings:
        print(f"[SETTING] {setting.name}  params={setting.params}")
        for filename in file_list:
            print(f"  - {filename}")
            data, label = _load_series(args.dataset_dir, filename)
            sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
            train_index = _parse_train_index(filename)
            data_train = data[:train_index, :]

            try:
                scores = run_AxonAD(data_train, data, **setting.params)
                metrics = get_metrics(scores, label, slidingWindow=sliding_window)
            except Exception:
                metrics = {k: 0.0 for k in metric_cols}

            row = {
                "setting": setting.name,
                "file": filename,
                "data_len": int(data.shape[0]),
                "train_split": int(train_index),
                **{k: float(metrics.get(k, 0.0)) for k in metric_cols},
            }
            rows.append(row)

    out_base = os.path.join(args.out_dir, f"axonad_tuning_sensitivity_{_timestamp()}")
    per_series_path = out_base + "_per_series.csv"
    agg_path = out_base + "_aggregated.csv"

    df = pd.DataFrame(rows)
    df.to_csv(per_series_path, index=False)

    agg = _aggregate(df, metric_cols)
    agg.to_csv(agg_path, index=False)

    print(f"Wrote: {per_series_path}")
    print(f"Wrote: {agg_path}")


if __name__ == "__main__":
    main()
