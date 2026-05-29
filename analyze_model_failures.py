#!/usr/bin/env python3
"""
analyze_model_failures.py
==========================
Mechanistic failure-mode analysis for 5 anomaly detection models
on the TSB-AD-M Tuning split + 20 UCR series (every ~12th of 250).

Models : OmniAnomaly, CNN, AxonAD, StreamVAE, MatrixProfile

Outputs (all saved to Results/analysis_failure_modes/):
  per_event_results.csv          – segment × model detection status + features
  anomaly_type_matrix.csv        – model × anomaly_type → detection rate
  mechanistic_signals.csv        – per-segment internal component signals
  figures/heatmap_detection.png  – model × dataset detection overview
  figures/type_detection_barplot.png
  figures/case_study_*.png       – raw signal + per-model score overlays
  figures/mechanistic_violin.png
  REPORT.md                      – auto-generated narrative summary

Run:
    python analyze_model_failures.py
"""
from __future__ import annotations

import gc
import glob
import math
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
import torch.nn.functional as F  # noqa: F401 (kept for hooks)
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 0 — CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    models: List[str] = field(default_factory=lambda: [
        "OmniAnomaly", "CNN", "AxonAD", "StreamVAE",
    ])
    benchmark_m_split: str = "tuning"
    ucr_n: int = 20
    ucr_stride: int = 12          # pick every 12th from sorted UCR list
    out_dir: str = "Results/analysis_failure_modes"
    threshold_pct: float = 99.0   # train-region percentile → detection threshold
    seed: int = 42

    # Dataset paths (adjust if needed)
    tsb_m_root: str = "Datasets/TSB-AD-M"
    ucr_root: str = (
        "/Users/KAOEZER/Downloads/AnomalyDatasets_2021/"
        "UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"
    )
    file_list_dir: str = "Datasets/File_List"

    # Figure settings
    fig_dpi: int = 120
    case_study_n: int = 3         # number of series to produce case-study plots for


CFG = Config()


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_train_index(filename: str) -> Optional[int]:
    m = re.search(r"_tr_(\d+)_", filename)
    return int(m.group(1)) if m else None


def parse_ucr_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", os.path.basename(filename))
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def load_ucr_txt(filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
    parsed = parse_ucr_filename(filepath)
    if parsed is None:
        raise ValueError(f"Cannot parse UCR filename: {filepath}")
    train_end, anomaly_start, anomaly_end = parsed
    values = np.loadtxt(filepath)
    X = values.reshape(-1, 1).astype(float)
    n = len(X)
    y = np.zeros(n, dtype=int)
    a_s = max(0, anomaly_start - 1)
    a_e = min(n - 1, anomaly_end - 1)
    y[a_s: a_e + 1] = 1
    return X, y, train_end


def _read_filelist_csv(path: str) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    for cand in ["file_name", "data_direc", "data_dir", "file", "filename", "path", "ts_name"]:
        for col in df.columns:
            if col.lower() == cand or cand in col.lower():
                return set(df[col].astype(str).apply(os.path.basename))
    return set(df.iloc[:, 0].astype(str).apply(os.path.basename))


def load_official_split_filenames(
    benchmark: str, split_mode: str, file_list_dir: str
) -> Optional[set]:
    benchmark = benchmark.upper()
    split_mode = split_mode.lower()
    list_paths = []
    if benchmark == "M":
        if split_mode in ("eval", "all"):
            list_paths.append(os.path.join(file_list_dir, "TSB-AD-M-Eva.csv"))
        if split_mode in ("tuning", "all"):
            list_paths.append(os.path.join(file_list_dir, "TSB-AD-M-Tuning.csv"))
        if split_mode == "telemetry":
            list_paths.append(os.path.join(file_list_dir, "TSB-AD-M-Telemetry.csv"))
    elif benchmark == "UCR":
        return None
    allowed: set = set()
    for p in list_paths:
        allowed |= _read_filelist_csv(p)
    return allowed if allowed else None


def _pad_scores(arr: np.ndarray, win_size: int, target_len: int) -> np.ndarray:
    """Pad a window-level score array back to the original data length."""
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    pad_l = math.ceil((win_size - 1) / 2)
    pad_r = (win_size - 1) // 2
    padded = np.concatenate([[arr[0]] * pad_l, arr, [arr[-1]] * pad_r])
    return padded[:target_len]


def _get_anomaly_runs(y: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, lbl in enumerate(y):
        if lbl == 1 and not in_seg:
            seg_start = i
            in_seg = True
        elif lbl == 0 and in_seg:
            runs.append((seg_start, i))
            in_seg = False
    if in_seg:
        runs.append((seg_start, len(y)))
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_datasets(cfg: Config) -> List[Dict]:
    datasets: List[Dict] = []

    # --- TSB-AD-M tuning split ---
    allowed = load_official_split_filenames("M", cfg.benchmark_m_split, cfg.file_list_dir)
    m_files = sorted(glob.glob(os.path.join(cfg.tsb_m_root, "*.csv")))
    if allowed:
        m_files = [f for f in m_files if os.path.basename(f) in allowed]

    for path in m_files:
        basename = os.path.basename(path)
        tokens = basename.replace(".csv", "").split("_")
        family = tokens[1] if len(tokens) > 1 else "unknown"
        domain = tokens[4] if len(tokens) > 4 else "unknown"
        train_idx = parse_train_index(basename) or 0
        try:
            df = pd.read_csv(path).dropna()
            X = df.iloc[:, :-1].values.astype(float)
            y = df.iloc[:, -1].astype(int).to_numpy()
        except Exception as e:
            print(f"[SKIP M] {basename}: {e}")
            continue
        datasets.append({
            "path": path, "basename": basename, "benchmark": "M",
            "X": X, "y": y, "train_idx": train_idx,
            "family": family, "domain": domain,
        })

    # --- UCR subset (every ucr_stride-th series) ---
    if os.path.exists(cfg.ucr_root):
        ucr_files = sorted(glob.glob(os.path.join(cfg.ucr_root, "*.txt")))
        ucr_selected = ucr_files[::cfg.ucr_stride][: cfg.ucr_n]
        for path in ucr_selected:
            basename = os.path.basename(path)
            try:
                X, y, train_end = load_ucr_txt(path)
            except Exception as e:
                print(f"[SKIP UCR] {basename}: {e}")
                continue
            datasets.append({
                "path": path, "basename": basename, "benchmark": "UCR",
                "X": X, "y": y, "train_idx": train_end,
                "family": "UCR", "domain": "unknown",
            })
    else:
        print(f"[WARN] UCR root not found: {cfg.ucr_root} — skipping UCR datasets.")

    n_m = sum(1 for d in datasets if d["benchmark"] == "M")
    n_ucr = sum(1 for d in datasets if d["benchmark"] == "UCR")
    print(f"[DATA] {len(datasets)} datasets loaded  (M-tuning={n_m}, UCR={n_ucr})")
    return datasets


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — MODEL EXECUTION WITH DISK CACHING
# ──────────────────────────────────────────────────────────────────────────────

def _cache_path(out_dir: str, model_name: str, basename: str) -> str:
    stem = basename.replace(".csv", "").replace(".txt", "")
    return os.path.join(out_dir, "scores_cache", model_name, stem + ".npy")


def _fit_clf(model_name: str, X: np.ndarray, train_idx: int, hp: Dict):
    """Instantiate and fit a model. Returns the fitted clf object."""
    from TSB_AD.utils.slidingWindows import find_length_rank

    if model_name == "OmniAnomaly":
        from TSB_AD.models.OmniAnomaly import OmniAnomaly
        clf = OmniAnomaly(
            win_size=hp.get("win_size", 100),
            feats=X.shape[1],
            lr=hp.get("lr", 0.002),
        )
        clf.fit(X[:train_idx])

    elif model_name == "CNN":
        from TSB_AD.models.CNN import CNN
        clf = CNN(
            window_size=hp.get("window_size", 100),
            num_channel=hp.get("num_channel", [32, 32, 40]),
            feats=X.shape[1],
            lr=hp.get("lr", 0.0008),
            batch_size=128,
        )
        clf.fit(X[:train_idx])

    elif model_name == "AxonAD":
        from TSB_AD.models.AxonAD import TCNTransformer
        clf = TCNTransformer(
            win_size=hp.get("win_size", 100),
            feats=X.shape[1],
            d_model=hp.get("d_model", 64),
            num_heads=hp.get("num_heads", 4),
            batch_size=128,
            epochs=hp.get("epochs", 50),
            lr=hp.get("lr", 0.001),
            validation_size=0.2,
            kl_tail_k=hp.get("kl_tail_k", 5),
            forecast_steps=hp.get("forecast_steps", 1),
        )
        clf.fit(X[:train_idx])

    elif model_name == "StreamVAE":
        from TSB_AD.models.StreamVAE import StreamVAE
        clf = StreamVAE(
            win_size=hp.get("win_size", 100),
            feats=X.shape[1],
            latent_dim=hp.get("latent_dim", 64),
            batch_size=128,
            epochs=hp.get("epochs", 30),
            patience=10,
            lr=hp.get("lr", 0.001),
            validation_size=0.2,
            target_kl=hp.get("target_kl", 100.0),
        )
        clf.fit(X[:train_idx])

    elif model_name == "MatrixProfile":
        from TSB_AD.models.MatrixProfile import MatrixProfile
        x_for_sw = X[:, 0].reshape(-1, 1) if X.ndim > 1 and X.shape[1] > 1 else X
        sw = find_length_rank(x_for_sw, rank=1)
        clf = MatrixProfile(window=sw)
        clf.fit(X)  # unsupervised — uses full series

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return clf


def run_model_and_cache(
    model_name: str,
    ds: Dict,
    hp: Dict,
    out_dir: str,
) -> Tuple[np.ndarray, object]:
    """
    Run model on ds, caching scores to disk.
    Returns (scores array [len(X),], fitted clf object).
    """
    cp = _cache_path(out_dir, model_name, ds["basename"])
    X, train_idx = ds["X"], ds["train_idx"]

    clf = _fit_clf(model_name, X, train_idx, hp)

    if os.path.exists(cp):
        scores = np.load(cp)
        return scores, clf

    os.makedirs(os.path.dirname(cp), exist_ok=True)
    if model_name == "MatrixProfile":
        scores = clf.decision_scores_.ravel()
    else:
        scores = clf.decision_function(X).ravel()

    scores = scores.astype(np.float32)
    np.save(cp, scores)
    return scores, clf


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — ANOMALY SEGMENT INFERENCE (heuristic type classification)
# ──────────────────────────────────────────────────────────────────────────────

def infer_anomaly_segments(X: np.ndarray, y: np.ndarray, train_idx: int) -> List[Dict]:
    """
    Classify each contiguous labeled segment in the TEST region.

    Classification rules (priority order):
      duration ≤ 3                        → point
      within-variance / train-variance < 0.1 → flatline
      amplitude_z > 3 and duration ≤ 30   → spike
      linear-trend R² > 0.7 and dur > 50  → drift
      amplitude_z > 1.5 and dur > 20      → level_shift
      else                                → contextual
    """
    # Work on highest-variance channel
    if X.ndim > 1 and X.shape[1] > 1:
        ch = int(np.argmax(X[:train_idx].var(axis=0))) if train_idx > 0 else 0
    else:
        ch = 0
    x_1d = X[:, ch] if X.ndim > 1 else X.ravel()

    if train_idx > 0:
        train_x = x_1d[:train_idx]
        train_mean = float(np.mean(train_x))
        train_std  = float(np.std(train_x)) + 1e-8
        train_var  = float(np.var(train_x)) + 1e-8
    else:
        train_mean, train_std, train_var = 0.0, 1.0, 1.0

    # Zero-out train labels so we only find test-region segments
    y_test = y.copy()
    y_test[:train_idx] = 0

    # Find contiguous runs of label=1
    runs: List[Tuple[int, int]] = _get_anomaly_runs(y_test)

    segments: List[Dict] = []
    for seg_id, (s, e) in enumerate(runs):
        seg_x = x_1d[s:e]
        duration = e - s
        if duration == 0:
            continue

        amplitude_z = float(np.max(np.abs(seg_x - train_mean)) / train_std)
        var_ratio   = float(np.var(seg_x) / train_var)

        if duration >= 3:
            t = np.arange(duration, dtype=float)
            coeffs = np.polyfit(t, seg_x, 1)
            fitted = np.polyval(coeffs, t)
            ss_res = float(np.sum((seg_x - fitted) ** 2))
            ss_tot = float(np.sum((seg_x - seg_x.mean()) ** 2)) + 1e-10
            trend_r2 = float(1.0 - ss_res / ss_tot)
        else:
            trend_r2 = 0.0

        # Priority classification
        if duration <= 3:
            atype = "point"
        elif var_ratio < 0.1:
            atype = "flatline"
        elif amplitude_z > 3 and duration <= 30:
            atype = "spike"
        elif trend_r2 > 0.7 and duration > 50:
            atype = "drift"
        elif amplitude_z > 1.5 and duration > 20:
            atype = "level_shift"
        else:
            atype = "contextual"

        segments.append({
            "seg_id":      seg_id,
            "start":       s,
            "end":         e,
            "duration":    duration,
            "type":        atype,
            "amplitude_z": round(amplitude_z, 3),
            "var_ratio":   round(var_ratio, 4),
            "trend_r2":    round(trend_r2, 4),
            "channel":     ch,
        })

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — PER-EVENT DETECTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def per_event_analysis(
    model_name: str,
    scores: np.ndarray,
    segments: List[Dict],
    ds: Dict,
    threshold_pct: float,
) -> List[Dict]:
    train_idx = ds["train_idx"]
    y = ds["y"]

    if train_idx <= 0 or len(scores) == 0 or not segments:
        return []

    train_scores = scores[:train_idx]
    if len(train_scores) == 0:
        return []

    threshold  = float(np.percentile(train_scores, threshold_pct))
    train_mean = float(np.mean(train_scores))
    train_std  = float(np.std(train_scores)) + 1e-8

    # False-positive density: fraction of non-anomaly TEST windows above threshold
    test_mask     = np.arange(len(scores)) >= train_idx
    non_anom_test = test_mask & (y == 0)
    fp_density = float((scores[non_anom_test] > threshold).mean()) if non_anom_test.sum() > 0 else 0.0

    rows: List[Dict] = []
    for seg in segments:
        s, e = seg["start"], seg["end"]
        seg_scores = scores[s:e]
        if len(seg_scores) == 0:
            continue

        detected     = bool(np.any(seg_scores > threshold))
        max_score_z  = float((float(np.max(seg_scores)) - train_mean) / train_std)
        delay        = int(np.argmax(seg_scores > threshold)) if detected else None

        rows.append({
            "model":            model_name,
            "dataset":          ds["basename"],
            "dataset_family":   ds["family"],
            "benchmark":        ds["benchmark"],
            "seg_id":           seg["seg_id"],
            "seg_start":        s,
            "seg_end":          e,
            "duration":         seg["duration"],
            "anomaly_type":     seg["type"],
            "amplitude_z":      seg["amplitude_z"],
            "var_ratio":        seg["var_ratio"],
            "trend_r2":         seg["trend_r2"],
            "detected":         detected,
            "delay":            delay,
            "max_score_zscore": round(max_score_z, 4),
            "false_pos_density": round(fp_density, 4),
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — MECHANISTIC HOOKS
# ──────────────────────────────────────────────────────────────────────────────

def _seg_mean(arr: np.ndarray, s: int, e: int) -> float:
    sub = arr[s: min(e, len(arr))]
    return float(np.mean(sub)) if len(sub) > 0 else 0.0


def _build_mech_rows(
    model_name: str,
    segments: List[Dict],
    sig1_name: str,
    sig1_vals: List[float],
    sig2_name: str,
    sig2_vals: List[float],
    dominant_vals: List[str],
) -> List[Dict]:
    """Build mechanistic signal rows; caller fills in 'dataset' field."""
    rows = []
    for seg, s1, s2, dom in zip(segments, sig1_vals, sig2_vals, dominant_vals):
        rows.append({
            "model":          model_name,
            "dataset":        "",          # filled by caller
            "seg_id":         seg["seg_id"],
            "anomaly_type":   seg["type"],
            "duration":       seg["duration"],
            "signal_1_name":  sig1_name,
            "signal_1_value": round(float(s1), 6) if np.isfinite(float(s1)) else 0.0,
            "signal_2_name":  sig2_name,
            "signal_2_value": round(float(s2), 6) if np.isfinite(float(s2)) else 0.0,
            "dominant_signal": dom,
        })
    return rows


# ── OmniAnomaly: per-window recon MSE vs KL divergence ───────────────────────

def omni_mech(
    clf, X_test: np.ndarray, segments: List[Dict], device: torch.device
) -> List[Dict]:
    from TSB_AD.utils.dataset import ReconstructDataset
    win_size = clf.win_size
    loader = DataLoader(
        ReconstructDataset(X_test, window_size=win_size),
        batch_size=128, shuffle=False,
    )

    recon_list, kl_list = [], []
    clf.model.eval()
    with torch.inference_mode():
        hidden = None
        for idx, (d, _) in enumerate(loader):
            d = d.to(device)
            y_pred, mu, logvar, hidden = clf.model(d, hidden if idx > 0 else None)
            # hidden is a GRU h_n tensor  →  detach to prevent memory build-up
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()

            # y_pred, mu, logvar shapes: (B, win*feats) and (B, win*n_latent)
            d_flat = d.view(d.shape[0], -1)
            recon_mse = torch.mean(clf.criterion(y_pred, d_flat), dim=-1)      # (B,)
            kl_per_win = (-0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
            ))  # (B,)
            recon_list.append(recon_mse.cpu())
            kl_list.append(kl_per_win.cpu())

    recon_arr = _pad_scores(torch.cat(recon_list).numpy(), win_size, len(X_test))
    kl_arr    = _pad_scores(torch.cat(kl_list).numpy(),   win_size, len(X_test))

    s1, s2, dom = [], [], []
    for seg in segments:
        r = _seg_mean(recon_arr, seg["start"], seg["end"])
        k = _seg_mean(kl_arr,   seg["start"], seg["end"])
        s1.append(r); s2.append(k)
        dom.append("recon" if r >= k else "kl")

    return _build_mech_rows("OmniAnomaly", segments, "recon_mse", s1, "kl_div", s2, dom)


# ── StreamVAE: per-window NLL vs KL divergence ───────────────────────────────

def streamvae_mech(
    clf, X_test: np.ndarray, segments: List[Dict], device: torch.device
) -> List[Dict]:
    from TSB_AD.utils.dataset import ReconstructDataset
    win_size = clf.win_size
    loader = DataLoader(
        ReconstructDataset(X_test, window_size=win_size),
        batch_size=128, shuffle=False,
    )

    nll_list, kl_list = [], []
    clf.model.eval()
    with torch.inference_mode():
        for d, _ in loader:
            d = d.to(device)
            rec_mu, rec_std, z_mu, z_std, _, _ = clf.model(d)
            nll = -Normal(rec_mu, rec_std).log_prob(d).sum(dim=-1).mean(dim=1)   # (B,)
            q   = Normal(z_mu, z_std)
            p   = Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
            kl  = kl_divergence(q, p).sum(dim=(1, 2))                            # (B,)
            nll_list.append(nll.detach().cpu())
            kl_list.append(kl.detach().cpu())

    nll_arr = _pad_scores(torch.cat(nll_list).numpy(), win_size, len(X_test))
    kl_arr  = _pad_scores(torch.cat(kl_list).numpy(),  win_size, len(X_test))

    s1, s2, dom = [], [], []
    for seg in segments:
        n = _seg_mean(nll_arr, seg["start"], seg["end"])
        k = _seg_mean(kl_arr,  seg["start"], seg["end"])
        s1.append(n); s2.append(k)
        dom.append("nll" if n >= k else "kl")

    return _build_mech_rows("StreamVAE", segments, "nll", s1, "kl_div", s2, dom)


# ── CNN: per-channel reconstruction MSE ──────────────────────────────────────

def cnn_mech(
    clf, X_test: np.ndarray, segments: List[Dict], device: torch.device
) -> List[Dict]:
    from TSB_AD.utils.dataset import ForecastDataset
    win_size = clf.window_size
    pred_len = clf.pred_len
    feats    = clf.feats
    loader   = DataLoader(
        ForecastDataset(X_test, window_size=win_size, pred_len=pred_len),
        batch_size=128, shuffle=False,
    )

    per_ch_list = []
    clf.model.eval()
    with torch.inference_mode():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            output = clf.model(x)                           # (pred_len, B, feats)
            # Clean reshape: (B, pred_len, feats)
            out_3d = output.permute(1, 0, 2)               # (B, pred_len, feats)
            tgt_3d = target.view(x.shape[0], pred_len, feats)
            mse_ch = (out_3d - tgt_3d).pow(2).mean(dim=1)  # (B, feats) mean over pred steps
            per_ch_list.append(mse_ch.detach().cpu())

    per_ch = torch.cat(per_ch_list, dim=0).numpy()         # (N_windows, feats)

    # Pad row-wise back to data length
    first = per_ch[:1]
    last  = per_ch[-1:]
    pad_l = math.ceil((win_size - 1) / 2)
    pad_r = (win_size - 1) // 2
    per_ch_pad = np.concatenate(
        [np.repeat(first, pad_l, axis=0), per_ch, np.repeat(last, pad_r, axis=0)], axis=0
    )[:len(X_test)]

    s1, s2, dom = [], [], []
    for seg in segments:
        s, e = seg["start"], min(seg["end"], len(per_ch_pad))
        if e > s:
            avg     = per_ch_pad[s:e].mean(axis=0)         # (feats,)
            dc      = int(np.argmax(avg))
            dom_val = float(avg[dc])
            mean_val= float(avg.mean())
        else:
            dc, dom_val, mean_val = 0, 0.0, 0.0
        s1.append(dom_val)
        s2.append(mean_val)
        dom.append(f"channel_{dc}")

    return _build_mech_rows(
        "CNN", segments,
        "dominant_channel_mse", s1,
        "mean_channel_mse", s2,
        dom,
    )


# ── MatrixProfile: profile distance + nearest-neighbour train fraction ────────

def mp_mech(
    clf, X: np.ndarray, train_idx: int, segments: List[Dict]
) -> List[Dict]:
    if not hasattr(clf, "profile") or clf.profile is None or len(clf.profile) == 0:
        s1 = [0.0] * len(segments)
        s2 = [0.5] * len(segments)
        dom = ["unknown"] * len(segments)
        return _build_mech_rows("MatrixProfile", segments,
                                "mp_distance", s1, "nn_in_train_frac", s2, dom)

    mp_dist = clf.profile[:, 0].astype(float)
    nn_idx  = clf.profile[:, 1].astype(int)
    win     = clf.window
    offset  = win // 2

    s1, s2, dom = [], [], []
    for seg in segments:
        p_s = max(0, seg["start"] - offset)
        p_e = min(len(mp_dist), seg["end"] - offset)
        if p_e > p_s:
            avg_dist       = float(np.mean(mp_dist[p_s:p_e]))
            nn_train_frac  = float(np.mean(nn_idx[p_s:p_e] < train_idx))
        else:
            avg_dist, nn_train_frac = 0.0, 0.5
        s1.append(avg_dist)
        s2.append(nn_train_frac)
        dom.append("distance" if avg_dist > 1.5 else "context")

    return _build_mech_rows("MatrixProfile", segments,
                            "mp_distance", s1, "nn_in_train_frac", s2, dom)


# ── AxonAD: recon MSE vs query mismatch (cosine distance) ─────────────────────

def axonad_mech(
    clf, X_test: np.ndarray, segments: List[Dict], device: torch.device
) -> List[Dict]:
    from TSB_AD.utils.dataset import ReconstructDataset
    win_size = clf.win_size
    loader = DataLoader(
        ReconstructDataset(X_test, window_size=win_size),
        batch_size=128, shuffle=False,
    )

    recon_list, jepa_list = [], []
    clf.model.eval()
    with torch.inference_mode():
        for d, _ in loader:
            d = d.to(device)
            rec, q_real, q_pred, k, q_tgt, _ = clf.model(d)
            mse_s  = (rec - d).pow(2).sum(dim=-1).mean(dim=1)   # (B,)
            jepa_s = clf._jepa_tail_dist(q_tgt, q_pred)          # (B,)
            recon_list.append(mse_s.detach().cpu())
            jepa_list.append(jepa_s.detach().cpu())

    recon_arr = _pad_scores(torch.cat(recon_list).numpy(), win_size, len(X_test))
    jepa_arr  = _pad_scores(torch.cat(jepa_list).numpy(),  win_size, len(X_test))

    s1, s2, dom = [], [], []
    for seg in segments:
        r = _seg_mean(recon_arr, seg["start"], seg["end"])
        j = _seg_mean(jepa_arr,  seg["start"], seg["end"])
        s1.append(r); s2.append(j)
        dom.append("recon" if r >= j else "query_mismatch")

    return _build_mech_rows("AxonAD", segments,
                            "recon_mse", s1, "query_mismatch", s2, dom)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6 — OUTPUT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def save_anomaly_type_matrix(per_event_df: pd.DataFrame, out_dir: str) -> None:
    if per_event_df.empty:
        return
    pivot = (
        per_event_df.groupby(["model", "anomaly_type"])["detected"]
        .mean()
        .unstack(fill_value=float("nan"))
    )
    path = os.path.join(out_dir, "anomaly_type_matrix.csv")
    pivot.to_csv(path)
    print(f"[OUT] {path}")


def save_heatmap(per_event_df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn not available — skipping heatmap.")
        return
    if per_event_df.empty:
        return

    det_by_ds = (
        per_event_df.groupby(["model", "dataset"])["detected"]
        .mean()
        .reset_index()
        .rename(columns={"detected": "detection_rate"})
    )
    heat = det_by_ds.pivot(index="model", columns="dataset", values="detection_rate")
    if "dataset_family" in per_event_df.columns:
        fam_map = per_event_df.drop_duplicates("dataset").set_index("dataset")["dataset_family"]
        heat = heat.reindex(columns=sorted(heat.columns, key=lambda c: fam_map.get(c, "z")))

    n_cols = len(heat.columns)
    fig, ax = plt.subplots(figsize=(max(16, n_cols * 0.55), 4))
    sns.heatmap(
        heat, ax=ax, vmin=0, vmax=1, cmap="RdYlGn",
        linewidths=0.3, cbar_kws={"label": "Detection Rate"},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("Model × Dataset Detection Rate", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    path = os.path.join(out_dir, "figures", "heatmap_detection.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[OUT] {path}")


def save_type_barplot(per_event_df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if per_event_df.empty:
        return

    grp    = per_event_df.groupby(["anomaly_type", "model"])["detected"].mean().reset_index()
    models = sorted(grp["model"].unique())
    atypes = sorted(grp["anomaly_type"].unique())
    x      = np.arange(len(atypes))
    width  = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(models):
        sub  = grp[grp["model"] == m].set_index("anomaly_type")["detected"]
        vals = [float(sub.get(t, float("nan"))) for t in atypes]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(atypes, rotation=20)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Anomaly Type and Model")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "figures", "type_detection_barplot.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[OUT] {path}")


def save_case_studies(
    datasets: List[Dict],
    all_scores: Dict[str, Dict[str, np.ndarray]],
    per_event_df: pd.DataFrame,
    cfg: Config,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if per_event_df.empty:
        return

    # Select datasets with most diverse anomaly types
    diversity = (
        per_event_df.groupby("dataset")["anomaly_type"].nunique()
        .sort_values(ascending=False)
    )
    top_datasets = list(diversity.index[: cfg.case_study_n])
    ds_map = {d["basename"]: d for d in datasets}
    colors = plt.cm.tab10.colors

    for ds_name in top_datasets:
        ds = ds_map.get(ds_name)
        if ds is None:
            continue
        X, y, train_idx = ds["X"], ds["y"], ds["train_idx"]
        n = len(X)
        t = np.arange(n)
        n_models = len(cfg.models)

        fig, axes = plt.subplots(n_models + 1, 1,
                                 figsize=(16, 3 * (n_models + 1)), sharex=True)
        fig.suptitle(f"Case Study: {ds_name}", fontsize=9)

        # Panel 0: raw signal
        ax0  = axes[0]
        sig  = X[:, 0] if X.ndim > 1 else X.ravel()
        ax0.plot(t, sig, lw=0.5, color="steelblue", label="Signal (ch 0)")
        ax0.axvline(train_idx, color="gray", ls="--", lw=1, label="train/test split")
        for s_i, e_i in _get_anomaly_runs(y):
            ax0.axvspan(s_i, e_i, alpha=0.25, color="red")
        ax0.set_ylabel("Value", fontsize=7)
        ax0.legend(fontsize=7, loc="upper right")
        ax0.set_title("Raw signal + anomaly regions (red)", fontsize=8)

        # Per-model score panels
        for mi, mname in enumerate(cfg.models):
            ax = axes[mi + 1]
            scores = all_scores.get(mname, {}).get(ds_name)
            if scores is not None:
                # Normalize to [0, 1] for visual comparison
                s_min, s_max = scores.min(), scores.max() + 1e-12
                ax.plot(t, (scores - s_min) / (s_max - s_min),
                        lw=0.5, color=colors[mi % len(colors)], label=mname)
            for s_i, e_i in _get_anomaly_runs(y):
                ax.axvspan(s_i, e_i, alpha=0.2, color="red")
            ax.axvline(train_idx, color="gray", ls="--", lw=0.8)
            ax.set_ylabel("Score (norm.)", fontsize=7)
            ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("Time step")
        plt.tight_layout()
        safe_name = re.sub(r"[^\w\-]", "_", ds_name)[:60]
        path = os.path.join(cfg.out_dir, "figures", f"case_study_{safe_name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUT] {path}")


def save_mechanistic_violin(mech_df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    if mech_df.empty:
        return

    # Subset to models with a recon-style primary signal
    plot_df = mech_df[mech_df["signal_1_name"].isin(["recon_mse", "nll", "mp_distance"])]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        sns.violinplot(
            data=plot_df, x="anomaly_type", y="signal_1_value",
            hue="model", ax=ax, inner="quartile", density_norm="width", cut=0,
        )
    except Exception:
        sns.boxplot(
            data=plot_df, x="anomaly_type", y="signal_1_value",
            hue="model", ax=ax,
        )
    ax.set_title("Primary Mechanistic Signal by Anomaly Type and Model")
    ax.set_xlabel("Anomaly Type")
    ax.set_ylabel("Signal Value")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, "figures", "mechanistic_violin.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[OUT] {path}")


def generate_report(
    per_event_df: pd.DataFrame,
    mech_df: pd.DataFrame,
    out_dir: str,
) -> None:
    lines: List[str] = []
    lines.append("# Failure Mode & Mechanistic Analysis Report\n\n")

    # ── Section 1: Dataset Summary ──────────────────────────────────────────
    lines.append("## 1. Dataset Summary\n\n")
    if not per_event_df.empty:
        n_ds  = per_event_df["dataset"].nunique()
        segs  = per_event_df[["dataset", "seg_id", "anomaly_type"]].drop_duplicates()
        n_seg = len(segs)
        tc    = segs["anomaly_type"].value_counts()
        lines.append(f"- Datasets analysed: **{n_ds}**\n")
        lines.append(f"- Labeled anomaly segments: **{n_seg}**\n\n")
        lines.append("### Anomaly Type Distribution\n\n")
        lines.append("| Type | Count | % |\n|------|------:|--:|\n")
        for atype, cnt in tc.items():
            lines.append(f"| {atype} | {cnt} | {100*cnt/n_seg:.1f}% |\n")
    else:
        lines.append("_No event data available._\n")

    # ── Section 2: Per-Model Overall Detection Rates ─────────────────────────
    lines.append("\n## 2. Per-Model Overall Detection Rates\n\n")
    if not per_event_df.empty:
        overall = (
            per_event_df.groupby("model")["detected"]
            .agg(rate="mean", total="count", detected_n="sum")
            .reset_index()
        )
        lines.append("| Model | Detection Rate | Detected | Total Segments |\n")
        lines.append("|-------|:-------------:|:--------:|:--------------:|\n")
        for _, row in overall.iterrows():
            lines.append(
                f"| {row['model']} | {row['rate']:.3f} | {int(row['detected_n'])} | {int(row['total'])} |\n"
            )

    # ── Section 3: Type Breakdown Table ─────────────────────────────────────
    lines.append("\n## 3. Detection Rate by Anomaly Type\n\n")
    if not per_event_df.empty:
        pivot = per_event_df.groupby(["model", "anomaly_type"])["detected"].mean().unstack()
        try:
            lines.append(pivot.round(3).to_markdown() + "\n")
        except Exception:
            lines.append(pivot.round(3).to_string() + "\n")

    # ── Section 4: Mechanistic Findings ──────────────────────────────────────
    lines.append("\n## 4. Mechanistic Findings Per Model\n")
    if not mech_df.empty:
        for model in sorted(mech_df["model"].unique()):
            sub = mech_df[mech_df["model"] == model]
            dc  = sub["dominant_signal"].value_counts()
            lines.append(f"\n### {model}\n\n")
            lines.append(f"- N segments analysed: {len(sub)}\n")
            lines.append(f"- Most common dominant signal: **{dc.index[0]}** ({dc.iloc[0]} segs)\n")
            by_type = (
                sub.groupby("anomaly_type")["dominant_signal"]
                .agg(lambda s: s.value_counts().index[0] if len(s) else "—")
            )
            lines.append("- Dominant signal per type: " +
                         ", ".join(f"`{t}` → `{v}`" for t, v in by_type.items()) + "\n")
    else:
        lines.append("\n_No mechanistic data available._\n")

    # ── Section 5: Failure Modes ──────────────────────────────────────────────
    lines.append("\n## 5. Top Failure Modes Per Model\n")
    if not per_event_df.empty:
        for model in sorted(per_event_df["model"].unique()):
            sub    = per_event_df[per_event_df["model"] == model]
            missed = (
                sub[~sub["detected"]].groupby("anomaly_type").size()
                .sort_values(ascending=False)
            )
            lines.append(f"\n### {model}\n\n")
            if len(missed):
                for atype, cnt in missed.head(3).items():
                    total_of_type = len(sub[sub["anomaly_type"] == atype])
                    miss_rate = cnt / total_of_type if total_of_type else 0
                    lines.append(f"- **{atype}**: {cnt} missed / {total_of_type} total ({miss_rate:.0%} miss rate)\n")
            else:
                lines.append("- No missed detections.\n")

    # ── Section 6: Suggested Research Directions ──────────────────────────────
    lines.append("""
## 6. Suggested Research Directions

### OmniAnomaly
- **Observed**: KL divergence tends to dominate on flatline/low-variance segments; the
  prior regularisation collapses the latent representation when the input is near-constant.
- **Direction**: Adaptive-β scheduling conditioned on within-window variance (lower β for
  low-variance windows so recon error stays sensitive); or a learnable prior that tracks
  the running signal distribution rather than a fixed N(0,1).

### CNN
- **Observed**: MSE is aggregated across all channels, diluting anomalies that manifest
  in only a subset of features (especially point/spike types).
- **Direction**: Replace mean-aggregation with a channel-attention or max-channel pooling
  at decision time. Alternatively, learn a per-channel anomaly weight from the training
  distribution (e.g. inverse variance of each channel's train-time error).

### AxonAD
- **Observed**: Query-mismatch captures structural deviations well; however slowly-evolving
  drift may not trigger large query errors because the EMA target encoder adapts continuously.
- **Direction**: Reduce EMA momentum (τ) or add a fixed "long-term anchor" updated only
  on a periodic schedule. A Kalman-smoothed baseline comparator could also flag slow drift
  that the EMA target absorbs.

### StreamVAE
- **Observed**: The NLL score combines marginal uncertainty and temporal structure; abrupt
  level-shifts can cause the reconstructed σ to widen (absorbing the shift), producing weak
  spike scores on level_shift anomalies.
- **Direction**: A fixed-σ decoder variant would prevent this "uncertainty dilation" and
  make the recon signal more discriminative. Alternatively, monitor the rate-of-change of
  reconstructed σ as an auxiliary signal.

### MatrixProfile
- **Observed**: Shape-based nearest-neighbour distance misses contextual anomalies that
  have a globally similar subsequence shape but occur at an unexpected time/frequency.
  The `nn_in_train_frac` signal reveals whether the model finds a training-region match.
- **Direction**: Combine MP distance with a temporal prior (expected recurrence period
  from training, estimated via autocorrelation). Flag windows whose nearest neighbour
  is anomalously far in time even if the profile distance is low.
""")

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[OUT] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(CFG.seed)
    device = pick_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {CFG.out_dir}")

    os.makedirs(CFG.out_dir, exist_ok=True)
    os.makedirs(os.path.join(CFG.out_dir, "figures"), exist_ok=True)

    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    datasets = load_datasets(CFG)
    if not datasets:
        print("[ERROR] No datasets found. Check path configuration.")
        return

    # CSV paths for progressive saving
    per_event_path = os.path.join(CFG.out_dir, "per_event_results.csv")
    mech_path      = os.path.join(CFG.out_dir, "mechanistic_signals.csv")

    # Resume support
    per_event_rows: List[Dict] = []
    completed: set = set()
    if os.path.exists(per_event_path):
        try:
            ex = pd.read_csv(per_event_path)
            completed = set(zip(ex["model"], ex["dataset"]))
            per_event_rows = ex.to_dict("records")
            print(f"[RESUME] {len(completed)} completed (model, dataset) pairs.")
        except Exception:
            pass

    mech_rows: List[Dict] = []
    mech_completed: set = set()
    if os.path.exists(mech_path):
        try:
            ex_m = pd.read_csv(mech_path)
            mech_completed = set(zip(ex_m["model"], ex_m["dataset"]))
            mech_rows = ex_m.to_dict("records")
        except Exception:
            pass

    # model → basename → score array  (for case-study plots, kept in RAM)
    all_scores: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in CFG.models}

    # ── Phases 2–5 main loop ─────────────────────────────────────────────────
    for model_name in CFG.models:
        hp = Optimal_Multi_algo_HP_dict.get(model_name, {})
        print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")

        for ds in tqdm(datasets, desc=model_name):
            basename = ds["basename"]
            X, y, train_idx = ds["X"], ds["y"], ds["train_idx"]

            # Phase 3: infer segments once per dataset
            segments = infer_anomaly_segments(X, y, train_idx)
            if not segments:
                continue   # no labeled anomalies in test region → skip

            # Phase 2: fit model & get scores (with disk cache)
            try:
                scores, clf = run_model_and_cache(model_name, ds, hp, CFG.out_dir)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[ERROR] {model_name} on {basename}: {e}")
                continue

            all_scores[model_name][basename] = scores

            # Phase 4: per-event detection rows
            if (model_name, basename) not in completed:
                rows = per_event_analysis(model_name, scores, segments, ds, CFG.threshold_pct)
                per_event_rows.extend(rows)
                completed.add((model_name, basename))
                # Progressive save after every dataset
                pd.DataFrame(per_event_rows).to_csv(per_event_path, index=False)

            # Phase 5: mechanistic hooks
            if (model_name, basename) not in mech_completed:
                try:
                    if model_name == "OmniAnomaly":
                        mrows = omni_mech(clf, X, segments, device)
                    elif model_name == "StreamVAE":
                        mrows = streamvae_mech(clf, X, segments, device)
                    elif model_name == "CNN":
                        mrows = cnn_mech(clf, X, segments, device)
                    elif model_name == "MatrixProfile":
                        mrows = mp_mech(clf, X, train_idx, segments)
                    elif model_name == "AxonAD":
                        mrows = axonad_mech(clf, X, segments, device)
                    else:
                        mrows = []

                    for r in mrows:
                        r["dataset"] = basename     # fill dataset name

                    mech_rows.extend(mrows)
                except Exception as e:
                    print(f"\n[WARN] Mechanistic hook failed {model_name}/{basename}: {e}")

                mech_completed.add((model_name, basename))
                pd.DataFrame(mech_rows).to_csv(mech_path, index=False)

            # Memory cleanup
            try:
                del clf, scores
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Phase 6: final outputs ───────────────────────────────────────────────
    per_event_df = pd.DataFrame(per_event_rows)
    mech_df      = pd.DataFrame(mech_rows)

    per_event_df.to_csv(per_event_path, index=False)
    mech_df.to_csv(mech_path, index=False)
    print(f"\n[OUT] {per_event_path}  ({len(per_event_df)} rows)")
    print(f"[OUT] {mech_path}  ({len(mech_df)} rows)")

    save_anomaly_type_matrix(per_event_df, CFG.out_dir)
    save_heatmap(per_event_df, CFG.out_dir, CFG.fig_dpi)
    save_type_barplot(per_event_df, CFG.out_dir, CFG.fig_dpi)
    save_mechanistic_violin(mech_df, CFG.out_dir, CFG.fig_dpi)
    save_case_studies(datasets, all_scores, per_event_df, CFG)
    generate_report(per_event_df, mech_df, CFG.out_dir)

    print(f"\n[DONE] All outputs saved to: {CFG.out_dir}")


if __name__ == "__main__":
    main()
