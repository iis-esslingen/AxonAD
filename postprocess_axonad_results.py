#!/usr/bin/env python3
"""Post-process TSB-AD per-series result CSVs into analysis CSVs.

Works with result files like:
    - tsb_ad_m_eval_results.csv
    - tsb_ad_m_tuning_axonad_ablations_results.csv

Expected schema (per-series):
    AD_Name, dataset, data_len, train_split, benchmark, split_mode, <metric columns...>

This script emits:
- per_series_with_family.csv: adds dataset_family + variant columns
- per_family_summary.csv: mean/std/count per dataset_family and AD_Name
- deltas_vs_reference_per_series.csv: per-series metric deltas vs a reference model
- win_rates_vs_reference.csv: wins/ties/losses + sign-test (and Wilcoxon if SciPy exists)

Typical usage:
    python postprocess_axonad_results.py \
        --input tsb_ad_m_eval_results.csv \
        --split_mode eval \
        --reference_model AxonAD

If you want to compare against a smaller baseline set:
    python postprocess_axonad_results.py \
        --input tsb_ad_m_eval_results.csv \
        --split_mode eval \
        --reference_model AxonAD \
        --compare_models TranAD USAD OmniAnomaly
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


META_COLS = {
    "AD_Name",
    "Model",  # aggregated schema
    "dataset",
    "data_len",
    "train_split",
    "benchmark",
    "split_mode",
}

DERIVED_NON_METRIC_COLS = {
    "variant",
    "dataset_family",
}


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_dataset_family(dataset_filename: str) -> str:
    """Extract dataset family (the benchmark's 'dataset' group) from a TSB-AD filename."""
    # Example: 004_MSL_id_3_Sensor_tr_530_1st_630.csv -> MSL
    m = re.match(r"^\d+_([^_]+)_id_", str(dataset_filename))
    if m:
        return m.group(1)
    # Fallback: second token often matches (MSL/MITDB/...) even if regex fails
    parts = str(dataset_filename).split("_")
    if len(parts) >= 2:
        return parts[1]
    return "UNKNOWN"


def parse_variant(ad_name: str) -> str:
    s = str(ad_name)
    if "__" in s:
        return s.split("__", 1)[1]
    return s


def _metric_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in META_COLS or c in DERIVED_NON_METRIC_COLS:
            continue
        if c.endswith("_mean") or c.endswith("_std") or c.endswith("_count"):
            # aggregated schema columns
            continue
        cols.append(c)
    return cols


def _two_sided_sign_test_p_value(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (ties removed)."""
    n = wins + losses
    if n <= 0:
        return 1.0

    k = min(wins, losses)

    # Compute 2 * P(X <= k) for X~Bin(n, 0.5)
    # Guard for numeric issues by capping at 1.
    p_leq = 0.0
    for i in range(0, k + 1):
        p_leq += math.comb(n, i) * (0.5 ** n)
    return float(min(1.0, 2.0 * p_leq))


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means.append(float(np.mean(sample)))
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


@dataclass(frozen=True)
class PairedTestRow:
    variant: str
    metric: str
    n: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    mean_delta: float
    median_delta: float
    sign_test_p: float
    wilcoxon_p: float
    ci_low: float
    ci_high: float
    test_used: str


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-process AxonAD result CSVs into analysis CSVs.")
    ap.add_argument("--input", default="tsb_ad_m_eval_results.csv", help="Per-series results CSV to post-process")
    ap.add_argument("--out_dir", default="analysis/axonad_post", help="Output folder")
    ap.add_argument(
        "--split_mode",
        default="",
        help="Optional: filter rows by split_mode (e.g. 'eval' or 'tuning'); empty keeps all",
    )
    ap.add_argument(
        "--benchmark",
        default="",
        help="Optional: filter rows by benchmark (e.g. 'M' or 'U'); empty keeps all",
    )
    ap.add_argument(
        "--reference_model",
        default="AxonAD",
        help="AD_Name to treat as reference for deltas + win-rates",
    )
    ap.add_argument(
        "--compare_models",
        nargs="*",
        default=[],
        help="Optional: list of AD_Name entries to compare vs reference (default: all other models)",
    )
    ap.add_argument(
        "--primary_metrics",
        nargs="*",
        default=["VUS-PR", "R-based-F1", "Affiliation-F", "AUC-PR", "AUC-ROC"],
        help="Metrics to include in win-rate + paired-tests outputs",
    )
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap replicates for mean-delta CI")
    ap.add_argument("--bootstrap_seed", type=int, default=0, help="Seed for bootstrap")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    df = pd.read_csv(args.input)

    if "AD_Name" not in df.columns or "dataset" not in df.columns:
        raise SystemExit(
            "Input must be the per-series results CSV with columns including AD_Name and dataset. "
            "(You passed a file that looks aggregated.)"
        )

    df = df.copy()

    if args.split_mode:
        df = df[df["split_mode"].astype(str).str.lower() == str(args.split_mode).lower()].copy()
    if args.benchmark:
        df = df[df["benchmark"].astype(str).str.upper() == str(args.benchmark).upper()].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering (check --split_mode / --benchmark)")

    df["variant"] = df["AD_Name"].astype(str).map(parse_variant)
    df["dataset_family"] = df["dataset"].astype(str).map(parse_dataset_family)

    metric_cols = _metric_cols(df)
    if not metric_cols:
        raise SystemExit("No metric columns detected in input CSV")

    # Ensure metrics are numeric (guard against object dtype surprises)
    for m in metric_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Persist enriched per-series table
    per_series_path = os.path.join(args.out_dir, "per_series_with_family.csv")
    df.to_csv(per_series_path, index=False)

    # Per-family summary (mean/std/count)
    agg = {}
    for m in metric_cols:
        agg[m + "_mean"] = (m, "mean")
        agg[m + "_std"] = (m, "std")
        agg[m + "_count"] = (m, "count")

    per_family = (
        df.groupby(["AD_Name", "variant", "dataset_family"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["variant", "dataset_family"])
    )
    per_family_path = os.path.join(args.out_dir, "per_family_summary.csv")
    per_family.to_csv(per_family_path, index=False)

    # Deltas vs reference per series
    ref = df[df["AD_Name"] == args.reference_model].copy()
    if ref.empty:
        available = sorted(df["AD_Name"].unique().tolist())
        raise SystemExit(
            f"Reference model not found in input: {args.reference_model}. "
            f"Available models include: {available[:20]}{' ...' if len(available) > 20 else ''}"
        )

    keep_cols = ["dataset", "dataset_family"] + [m for m in metric_cols if m in df.columns]
    ref = ref[keep_cols].rename(columns={m: f"{m}__ref" for m in metric_cols})

    merged = df.merge(ref, on=["dataset", "dataset_family"], how="left")
    for m in metric_cols:
        merged[f"delta__{m}"] = merged[m] - merged[f"{m}__ref"]

    deltas_path = os.path.join(args.out_dir, "deltas_vs_reference_per_series.csv")
    merged.to_csv(deltas_path, index=False)

    # Win-rates + paired tests (variant vs base) for selected metrics
    primary_metrics = [m for m in args.primary_metrics if m in metric_cols]
    if not primary_metrics:
        raise SystemExit(
            "None of --primary_metrics exist in the input CSV. "
            f"Available metrics: {metric_cols}"
        )

    # Optional Wilcoxon if SciPy exists
    wilcoxon_fn = None
    try:
        from scipy.stats import wilcoxon as _wilcoxon  # type: ignore

        wilcoxon_fn = _wilcoxon
    except Exception:
        wilcoxon_fn = None

    test_rows: list[PairedTestRow] = []

    all_models = sorted(df["AD_Name"].unique().tolist())
    if args.compare_models:
        compare_models = [m for m in args.compare_models if m in set(all_models)]
        missing = [m for m in args.compare_models if m not in set(all_models)]
        if missing:
            raise SystemExit(f"compare_models not found in input: {missing}")
    else:
        compare_models = [m for m in all_models if m != args.reference_model]

    for ad_name in compare_models:
        sub = merged[merged["AD_Name"] == ad_name].copy()
        for metric in primary_metrics:
            d = sub[f"delta__{metric}"].to_numpy(dtype=float)
            d = d[np.isfinite(d)]

            wins = int(np.sum(d > 0))
            losses = int(np.sum(d < 0))
            ties = int(np.sum(d == 0))
            n = int(d.size)
            win_rate = float(wins / n) if n else float("nan")

            sign_p = _two_sided_sign_test_p_value(wins=wins, losses=losses)
            wilcox_p = float("nan")
            test_used = "sign_test"

            if wilcoxon_fn is not None:
                try:
                    # Wilcoxon ignores zeros by default with zero_method='wilcox'
                    res = wilcoxon_fn(d, zero_method="wilcox", alternative="two-sided")
                    wilcox_p = float(res.pvalue)
                    test_used = "wilcoxon"
                except Exception:
                    # Fall back to sign test
                    wilcox_p = float("nan")
                    test_used = "sign_test"

            ci_lo, ci_hi = _bootstrap_ci(d, n_boot=args.bootstrap, seed=args.bootstrap_seed)

            test_rows.append(
                PairedTestRow(
                    variant=str(ad_name),
                    metric=str(metric),
                    n=n,
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    win_rate=win_rate,
                    mean_delta=float(np.mean(d)) if n else float("nan"),
                    median_delta=float(np.median(d)) if n else float("nan"),
                    sign_test_p=sign_p,
                    wilcoxon_p=wilcox_p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                    test_used=test_used,
                )
            )

    win_df = pd.DataFrame([r.__dict__ for r in test_rows]).sort_values(["metric", "variant"])
    win_path = os.path.join(args.out_dir, "win_rates_vs_reference.csv")
    win_df.to_csv(win_path, index=False)

    print(f"Wrote: {per_series_path}")
    print(f"Wrote: {per_family_path}")
    print(f"Wrote: {deltas_path}")
    print(f"Wrote: {win_path}")


if __name__ == "__main__":
    main()
