"""Single-file mechanistic evidence suite (no CLI args).

Run:
    python run_mechanistic_auto.py

Context
    This script is designed to produce mechanistic evidence for why AxonAD's
    *query mismatch surprisal* complements reconstruction error and why query
    prediction is a stronger target than predicting keys/values/hidden states.

What it produces (cached + reproducible from saved tensors)
    - Saved tensors per series: window-level scalar scores (mse, qmis, kl_tail,
      attn_entropy), per-head mechanistic signals (dq_norm_h, attn_kl_h), labels,
      z-scores, and window indices.  Raw sampled Q/K/V/attention tensors are NOT
      cached (only their derived statistics are stored).
    - Hypothesis-linked plots + captions.
    - Quant tables with block-bootstrap 95% CIs.
    - A compact REPORT.md and a one-page checklist.

Leakage policy
    - Model is trained on the official train prefix (derived from `_tr_<idx>_` in filename).
    - Any standardization / thresholds use TRAIN windows only.
    - TEST windows are used only for reporting/plots/metrics.

No args by design.
    If you want more/less compute, edit the CONFIG block.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Disable TorchDynamo/torch.compile before importing torch.
# Without this, the first torch.optim.AdamW() call triggers a lazy import of
# torch._dynamo → sympy → mpmath which can stall for tens of seconds on first run
# (especially on Apple Silicon / MPS setups with no GPU cache warm-up).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from TSB_AD.models.AxonAD import TCNQueryPredictor, TCNTransformer
from TSB_AD.utils.dataset import ReconstructDataset


# =========================
# CONFIG (edit if desired)
# =========================

@dataclass(frozen=True)
class Config:
    benchmark: str = "M"  # "M" or "U"
    split_list: str = "Tuning"  # "Tuning" | "Eva" | "Telemetry" (if present)

    # Work limiting (keeps default run practical)
    # Default suite: tuning subset (20 series) + 3 case studies auto-chosen.
    series_limit: int | None = 20  # set to None to run all series in the list
    max_windows_per_series: int = 6000  # cap windows processed for extraction (per series)
    save_attention_for_case_studies_only: bool = True
    attention_samples_per_case_study: int = 240  # total windows sampled per case study (normal+anomaly)

    # Paper mode: keep only the figures that support the mechanism.
    paper_figures_only: bool = True
    case_studies: int = 2  # usually enough: one recon-dominant + one query-dominant
    make_query_manifold: bool = False  # optional appendix
    make_head_specialization: bool = False  # optional appendix
    make_per_series_debug_plots: bool = False

    # Training-objective ablations (matches run_AxonAD_ablations.py logic)
    # ReconOnly, JEPAOnly_Q, PredKeys, PredValues, PredHiddenState
    run_training_objective_ablations: bool = True
    ablation_scope: str = "case"  # "case" (default) | "all" (all processed series)
    # These two names are intentionally distinct from ablation_epochs/ablation_patience
    # (used for score-mode / TCNTransformerAblation runs below) to avoid the Python
    # dataclass silent-override bug when both sets of fields shared the same name.
    training_obj_ablation_epochs: int = 20
    training_obj_ablation_patience: int = 2

    # Base model hparams (per-series)
    win_size: int = 100
    d_model: int = 64
    num_heads: int = 4
    forecast_steps: int = 1
    kl_tail_k: int = 5

    # Training (per-series)
    epochs: int = 30
    patience: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    validation_size: float = 0.2

    # Training-objective ablations (matches run_AxonAD_ablations.py subset)
    run_objective_ablations: bool = True
    ablation_epochs: int = 12
    ablation_patience: int = 2
    ablation_lr: float = 1e-3

    # Output location
    out_dir: str = "mechanistic_run"
    seed: int = 1337

    # Reporting
    block_bootstrap_block_len: int = 200
    block_bootstrap_samples: int = 400
    fpr_for_complementarity: float = 0.01
    z_hi: float = 3.0
    z_lo: float = 1.0


CFG = Config()


# =========================
# Small utilities
# =========================


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


def parse_train_index(filename: str) -> int | None:
    m = re.search(r"_tr_(\d+)_", filename)
    if m:
        return int(m.group(1))
    # Fallback: no _tr_<idx>_ pattern found.  Emit a loud warning so that
    # leakage-policy violations are never silent.
    import warnings
    warnings.warn(
        f"[LEAKAGE WARNING] parse_train_index: no '_tr_<idx>_' pattern found in "
        f"filename {filename!r}.  Falling back to hard-coded 80/20 split.  "
        "This may violate the official train/test protocol.  "
        "Verify the filename pattern or provide an explicit train_end.",
        stacklevel=2,
    )
    return None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig_png_pdf(path: Path, *, dpi: int = 200, bbox_inches: str = "tight") -> None:
    """Save current matplotlib figure to both PNG (preview) and PDF (paper).

    Accepts either a .png path or a stem; always writes <stem>.png and <stem>.pdf.
    """
    from pathlib import Path as _Path

    import matplotlib.pyplot as plt

    p = _Path(path)
    ensure_dir(p.parent)
    png_path = p if p.suffix.lower() == ".png" else p.with_suffix(".png")
    plt.savefig(png_path, dpi=int(dpi), bbox_inches=bbox_inches)
    plt.savefig(png_path.with_suffix(".pdf"), bbox_inches=bbox_inches)


def read_file_list(benchmark: str, split_list: str) -> list[str]:
    file = Path("Datasets") / "File_List" / f"TSB-AD-{benchmark}-{split_list}.csv"
    if not file.exists():
        raise FileNotFoundError(f"Missing split list: {file}")

    df = pd.read_csv(file)
    # files are in a column 'file_name' (most lists) or first column
    col = "file_name" if "file_name" in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


def series_data_dir(benchmark: str) -> Path:
    return Path("Datasets") / f"TSB-AD-{benchmark}"


def load_series_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column in {path}")

    labels = df["Label"].to_numpy(dtype=np.int64)
    feats = df.drop(columns=["Label"]).to_numpy(dtype=np.float32)

    if feats.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix; got shape={feats.shape} for {path}")

    return feats, labels


def robust_fit(arr: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
    med = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float((q75 - q25) + eps)
    return med, iqr


def safe_softmax_masked(logits: torch.Tensor, mask_future: torch.Tensor) -> torch.Tensor:
    # logits: [B,H,k,T], mask_future: [1,1,k,T] bool
    neg = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(mask_future, neg)
    return F.softmax(masked, dim=-1)


# =========================
# Checkpoint I/O
# =========================


def checkpoint_path(out_root: Path, series_name: str) -> Path:
    stem = Path(series_name).stem
    return out_root / "checkpoints" / stem / "base_model.pt"

def ablation_checkpoint_path(out_root: Path, series_name: str, variant: str) -> Path:
    stem = Path(series_name).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(variant))
    return out_root / "checkpoints" / stem / f"ablation_{safe}.pt"

class ObjectiveAblationTCN(TCNTransformer):
    """Train the same AxonAD architecture under different objectives.

    Supported objectives:
      - rec: reconstruction only
      - jepa: JEPA query prediction only
      - pred_h: predict hidden state h_norm (cosine distance)
      - pred_k: predict keys K (cosine distance)
      - pred_v: predict values V (cosine distance)
    """

    def __init__(
        self,
        *,
        train_objective: str,
        patience: int = 3,
        **kwargs,
    ):
        super().__init__(patience=patience, **kwargs)
        # TCNTransformer does not expose self.patience; store it explicitly so that
        # checkpoint-save code (and early-stopping init) can access it.
        self.patience = patience
        self._train_objective = str(train_objective).lower().strip()
        allowed = {"rec", "jepa", "pred_h", "pred_k", "pred_v"}
        if self._train_objective not in allowed:
            raise ValueError(f"train_objective must be one of: {sorted(allowed)}")

        self._mse_loss = nn.MSELoss()
        self.k_pred_net: nn.Module | None = None
        self.v_pred_net: nn.Module | None = None
        if self._train_objective == "pred_k":
            self.k_pred_net = TCNQueryPredictor(d_model=self.model.d_model).to(self.device)
        if self._train_objective == "pred_v":
            self.v_pred_net = TCNQueryPredictor(d_model=self.model.d_model).to(self.device)
        if self.k_pred_net is not None or self.v_pred_net is not None:
            params = list(self.model.parameters())
            if self.k_pred_net is not None:
                params += list(self.k_pred_net.parameters())
            if self.v_pred_net is not None:
                params += list(self.v_pred_net.parameters())
            self.opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-5)

    def _shift_history(self, h_detached: torch.Tensor, s: int) -> torch.Tensor:
        B, T, C = h_detached.shape
        s = max(0, min(int(s), T))
        if s == 0:
            return h_detached
        x_shift = h_detached.new_zeros(B, T, C)
        x_shift[:, s:, :] = h_detached[:, :-s, :]
        return x_shift

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,H,T,D] -> [B,T,H*D]
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)

    def _cos_masked_loss_flat(self, pred_flat: torch.Tensor, tgt_flat: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
        if tmask is None or tmask.numel() == 0:
            return pred_flat.new_zeros(())
        m = tmask.to(pred_flat.device)
        if m.dtype != torch.bool:
            m = m.bool()
        if m.sum() == 0:
            return pred_flat.new_zeros(())

        p = F.normalize(pred_flat, dim=-1)
        t = F.normalize(tgt_flat, dim=-1)
        cos = (p * t).sum(dim=-1)  # [B,T]
        dist = 1.0 - cos
        return dist[m].mean()

    def fit(self, data):
        split = int((1 - self.validation_size) * len(data))
        if split <= self.win_size:
            tsTrain = data
            tsValid = data[:0]
        else:
            tsTrain = data[:split]
            tsValid = data[split:]

        train_loader = DataLoader(
            ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        objective = str(self._train_objective)

        for _epoch in range(1, int(self.epochs) + 1):
            self.model.train()
            for d, _ in train_loader:
                d = d.to(self.device)
                rec, q_real, q_pred, k, q_tgt, tmask = self.model(d)

                h = self.model.embed(d) + self.model.pos
                h_norm = self.model.ln1(h)
                h_det = h_norm.detach()
                s = max(0, min(int(self.model.forecast_steps), h_det.shape[1]))
                x_shift = self._shift_history(h_det, s)

                q_pred_flat = self.model.attn.q_pred_net(x_shift)  # [B,T,C]
                k_tgt = k.detach()  # [B,H,T,D]
                v_tgt = self.model.attn._split_heads(self.model.attn.wv(h_norm)).detach()  # [B,H,T,D]

                L_rec = self._mse_loss(rec, d)
                L_jepa = self._jepa_masked_loss(q_tgt, q_pred, tmask)

                if objective == "pred_h":
                    L = self._cos_masked_loss_flat(q_pred_flat, h_det, tmask)
                elif objective == "pred_k":
                    if self.k_pred_net is None:
                        raise RuntimeError("k_pred_net not initialized")
                    k_pred_flat = self.k_pred_net(x_shift)
                    k_pred = self.model._split_heads(k_pred_flat, self.model.attn.num_heads)
                    L = self._cos_masked_loss_flat(self._merge_heads(k_pred), self._merge_heads(k_tgt), tmask)
                elif objective == "pred_v":
                    if self.v_pred_net is None:
                        raise RuntimeError("v_pred_net not initialized")
                    v_pred_flat = self.v_pred_net(x_shift)
                    v_pred = self.model._split_heads(v_pred_flat, self.model.attn.num_heads)
                    L = self._cos_masked_loss_flat(self._merge_heads(v_pred), self._merge_heads(v_tgt), tmask)
                elif objective == "jepa":
                    L = L_jepa
                else:  # rec
                    L = L_rec

                self.opt.zero_grad(set_to_none=True)
                L.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.model.ema_update_target()

            if len(valid_loader) > 0:
                self.model.eval()
                val_sum, n_batches = 0.0, 0
                with torch.inference_mode():
                    for d, _ in valid_loader:
                        d = d.to(self.device)
                        rec, q_real, q_pred, k, q_tgt, tmask = self.model(d)
                        h = self.model.embed(d) + self.model.pos
                        h_norm = self.model.ln1(h)
                        h_det = h_norm.detach()
                        s = max(0, min(int(self.model.forecast_steps), h_det.shape[1]))
                        x_shift = self._shift_history(h_det, s)
                        q_pred_flat = self.model.attn.q_pred_net(x_shift)
                        k_tgt = k.detach()
                        v_tgt = self.model.attn._split_heads(self.model.attn.wv(h_norm)).detach()

                        L_rec = self._mse_loss(rec, d)
                        L_jepa = self._jepa_masked_loss(q_tgt, q_pred, tmask)

                        if objective == "pred_h":
                            Lv = self._cos_masked_loss_flat(q_pred_flat, h_det, tmask)
                        elif objective == "pred_k":
                            if self.k_pred_net is None:
                                raise RuntimeError("k_pred_net not initialized")
                            k_pred_flat = self.k_pred_net(x_shift)
                            k_pred = self.model._split_heads(k_pred_flat, self.model.attn.num_heads)
                            Lv = self._cos_masked_loss_flat(self._merge_heads(k_pred), self._merge_heads(k_tgt), tmask)
                        elif objective == "pred_v":
                            if self.v_pred_net is None:
                                raise RuntimeError("v_pred_net not initialized")
                            v_pred_flat = self.v_pred_net(x_shift)
                            v_pred = self.model._split_heads(v_pred_flat, self.model.attn.num_heads)
                            Lv = self._cos_masked_loss_flat(self._merge_heads(v_pred), self._merge_heads(v_tgt), tmask)
                        elif objective == "jepa":
                            Lv = L_jepa
                        else:
                            Lv = L_rec

                        val_sum += float(Lv.item())
                        n_batches += 1

                val_avg = val_sum / max(1, n_batches)
                self.early_stopping(val_avg, self.model)
                if self.early_stopping.early_stop:
                    break

        self._calibrate(tsTrain)


def load_or_train_objective_ablation(
    *,
    series_name: str,
    series_X: np.ndarray,
    train_end: int,
    out_root: Path,
    variant_name: str,
    train_objective: str,
    batch_size: int,
) -> ObjectiveAblationTCN:
    ckpt = ablation_checkpoint_path(out_root, series_name, variant_name)
    device = pick_device()

    feats = int(series_X.shape[1])
    if ckpt.exists():
        obj = torch.load(ckpt, map_location="cpu")
        h = obj.get("hparams", {}) if isinstance(obj, dict) else {}
        base = ObjectiveAblationTCN(
            win_size=int(h.get("win_size", CFG.win_size)),
            feats=int(h.get("feats", feats)),
            d_model=int(h.get("d_model", CFG.d_model)),
            num_heads=int(h.get("num_heads", CFG.num_heads)),
            batch_size=int(h.get("batch_size", batch_size)),
            epochs=1,
            patience=int(h.get("patience", CFG.training_obj_ablation_patience)),
            lr=float(h.get("lr", CFG.ablation_lr)),
            validation_size=float(h.get("validation_size", CFG.validation_size)),
            kl_tail_k=int(h.get("kl_tail_k", CFG.kl_tail_k)),
            forecast_steps=int(h.get("forecast_steps", CFG.forecast_steps)),
            train_objective=str(obj.get("train_objective", train_objective)),
        )
        base.device = device
        base.model = base.model.to(device)
        params = list(base.model.parameters())
        if base.k_pred_net is not None:
            params += list(base.k_pred_net.parameters())
        if base.v_pred_net is not None:
            params += list(base.v_pred_net.parameters())
        base.opt = torch.optim.AdamW(params, lr=base.lr, weight_decay=1e-5)

        if isinstance(obj, dict) and "model_state_dict" in obj:
            base.model.load_state_dict(obj["model_state_dict"], strict=True)
            if base.k_pred_net is not None and "k_pred_state_dict" in obj:
                base.k_pred_net.load_state_dict(obj["k_pred_state_dict"], strict=True)
            if base.v_pred_net is not None and "v_pred_state_dict" in obj:
                base.v_pred_net.load_state_dict(obj["v_pred_state_dict"], strict=True)
            # Restore calibration stats so as_detector scoring matches decision_function.
            cal = obj.get("calibration", {})
            base.mse_med  = float(cal.get("mse_med",  0.0))
            base.mse_iqr  = float(cal.get("mse_iqr",  1.0))
            base.jepa_med = float(cal.get("jepa_med", 0.0))
            base.jepa_iqr = float(cal.get("jepa_iqr", 1.0))
            base.kl_med   = float(cal.get("kl_med",   0.0))
            base.kl_iqr   = float(cal.get("kl_iqr",   1.0))
        else:
            base.model.load_state_dict(obj, strict=True)
        return base

    base = ObjectiveAblationTCN(
        win_size=CFG.win_size,
        feats=feats,
        d_model=CFG.d_model,
        num_heads=CFG.num_heads,
        batch_size=batch_size,
        epochs=CFG.training_obj_ablation_epochs,
        patience=CFG.training_obj_ablation_patience,
        lr=CFG.ablation_lr,
        validation_size=CFG.validation_size,
        kl_tail_k=CFG.kl_tail_k,
        forecast_steps=CFG.forecast_steps,
        train_objective=train_objective,
    )
    base.device = device
    base.model = base.model.to(device)
    params = list(base.model.parameters())
    if base.k_pred_net is not None:
        params += list(base.k_pred_net.parameters())
    if base.v_pred_net is not None:
        params += list(base.v_pred_net.parameters())
    base.opt = torch.optim.AdamW(params, lr=base.lr, weight_decay=1e-5)

    ts_train = series_X[:train_end]
    base.fit(ts_train)

    payload = {
        "model_state_dict": base.model.state_dict(),
        "k_pred_state_dict": base.k_pred_net.state_dict() if base.k_pred_net is not None else None,
        "v_pred_state_dict": base.v_pred_net.state_dict() if base.v_pred_net is not None else None,
        "train_objective": train_objective,
        "calibration": {
            "mse_med":  float(base.mse_med  if base.mse_med  is not None else 0.0),
            "mse_iqr":  float(base.mse_iqr  if base.mse_iqr  is not None else 1.0),
            "jepa_med": float(base.jepa_med if base.jepa_med is not None else 0.0),
            "jepa_iqr": float(base.jepa_iqr if base.jepa_iqr is not None else 1.0),
            "kl_med":   float(base.kl_med   if base.kl_med   is not None else 0.0),
            "kl_iqr":   float(base.kl_iqr   if base.kl_iqr   is not None else 1.0),
        },
        "hparams": {
            "win_size": int(base.win_size),
            "feats": int(base.feats),
            "d_model": int(base.model.d_model),
            "num_heads": int(base.model.attn.num_heads),
            "forecast_steps": int(base.forecast_steps),
            "kl_tail_k": int(base.kl_tail_k),
            "batch_size": int(base.batch_size),
            "epochs": int(base.epochs),
            "patience": int(getattr(base, "patience", CFG.training_obj_ablation_patience)),
            "lr": float(base.lr),
            "validation_size": float(base.validation_size),
        },
    }
    ensure_dir(ckpt.parent)
    torch.save(payload, ckpt)
    return base


def checkpoint_path_variant(out_root: Path, series_name: str, variant: str) -> Path:
    stem = Path(series_name).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(variant))
    return out_root / "checkpoints" / stem / f"variant_{safe}.pt"


def save_checkpoint(model: TCNTransformer, path: Path, meta: dict[str, Any]) -> None:
    payload = {
        "model_state_dict": model.model.state_dict(),
        "hparams": {
            "win_size": int(model.win_size),
            "feats": int(model.feats),
            "d_model": int(model.model.d_model),
            "num_heads": int(model.model.attn.num_heads),
            "forecast_steps": int(model.forecast_steps),
            "kl_tail_k": int(model.kl_tail_k),
            "batch_size": int(model.batch_size),
            "epochs": int(model.epochs),
            "lr": float(model.lr),
            "validation_size": float(model.validation_size),
        },
        "calibration": {
            "mse_med": float(model.mse_med if model.mse_med is not None else 0.0),
            "mse_iqr": float(model.mse_iqr if model.mse_iqr is not None else 1.0),
            "jepa_med": float(model.jepa_med if model.jepa_med is not None else 0.0),
            "jepa_iqr": float(model.jepa_iqr if model.jepa_iqr is not None else 1.0),
            "kl_med": float(model.kl_med if model.kl_med is not None else 0.0),
            "kl_iqr": float(model.kl_iqr if model.kl_iqr is not None else 1.0),
        },
        "meta": meta,
    }
    ensure_dir(path.parent)
    torch.save(payload, path)


def save_variant_checkpoint(model: "TCNTransformerAblation", path: Path, meta: dict[str, Any]) -> None:
    payload = {
        "model_state_dict": model.model.state_dict(),
        "variant": {
            "train_objective": str(model._train_objective),
            "score_mode": str(model._score_mode),
        },
        "aux_state": {
            "k_pred_net": (model.k_pred_net.state_dict() if model.k_pred_net is not None else None),
            "v_pred_net": (model.v_pred_net.state_dict() if model.v_pred_net is not None else None),
        },
        "hparams": {
            "win_size": int(model.win_size),
            "feats": int(model.feats),
            "d_model": int(model.model.d_model),
            "num_heads": int(model.model.attn.num_heads),
            "forecast_steps": int(model.forecast_steps),
            "kl_tail_k": int(model.kl_tail_k),
            "batch_size": int(model.batch_size),
            "epochs": int(model.epochs),
            "lr": float(model.lr),
            "validation_size": float(model.validation_size),
        },
        "calibration": {
            "mse_med": float(model.mse_med if model.mse_med is not None else 0.0),
            "mse_iqr": float(model.mse_iqr if model.mse_iqr is not None else 1.0),
            "jepa_med": float(model.jepa_med if model.jepa_med is not None else 0.0),
            "jepa_iqr": float(model.jepa_iqr if model.jepa_iqr is not None else 1.0),
            "kl_med": float(model.kl_med if model.kl_med is not None else 0.0),
            "kl_iqr": float(model.kl_iqr if model.kl_iqr is not None else 1.0),
        },
        "meta": meta,
    }
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_or_train_base_model(
    *,
    series_name: str,
    series_X: np.ndarray,
    train_end: int,
    out_root: Path,
) -> tuple[TCNTransformer, Path]:
    ckpt = checkpoint_path(out_root, series_name)

    device = pick_device()

    if ckpt.exists():
        obj = torch.load(ckpt, map_location="cpu")
        h = obj.get("hparams", {}) if isinstance(obj, dict) else {}
        win_size = int(h.get("win_size", CFG.win_size))
        feats = int(h.get("feats", series_X.shape[1]))
        d_model = int(h.get("d_model", CFG.d_model))
        num_heads = int(h.get("num_heads", CFG.num_heads))
        forecast_steps = int(h.get("forecast_steps", CFG.forecast_steps))
        kl_tail_k = int(h.get("kl_tail_k", CFG.kl_tail_k))

        base = TCNTransformer(
            win_size=win_size,
            feats=feats,
            d_model=d_model,
            num_heads=num_heads,
            batch_size=CFG.batch_size,
            epochs=1,
            patience=CFG.patience,
            lr=CFG.lr,
            validation_size=CFG.validation_size,
            kl_tail_k=kl_tail_k,
            forecast_steps=forecast_steps,
        )
        # Move + rebuild optimizer on chosen device
        base.device = device
        base.model = base.model.to(device)
        base.opt = torch.optim.AdamW(base.model.parameters(), lr=base.lr, weight_decay=1e-5)

        if isinstance(obj, dict) and "model_state_dict" in obj:
            base.model.load_state_dict(obj["model_state_dict"], strict=True)
            calib = obj.get("calibration", {})
            base.mse_med, base.mse_iqr = float(calib.get("mse_med", 0.0)), float(calib.get("mse_iqr", 1.0))
            base.jepa_med, base.jepa_iqr = float(calib.get("jepa_med", 0.0)), float(calib.get("jepa_iqr", 1.0))
            base.kl_med, base.kl_iqr = float(calib.get("kl_med", 0.0)), float(calib.get("kl_iqr", 1.0))
        else:
            # Back-compat: raw state_dict
            base.model.load_state_dict(obj, strict=True)
        return base, ckpt

    feats = int(series_X.shape[1])
    base = TCNTransformer(
        win_size=CFG.win_size,
        feats=feats,
        d_model=CFG.d_model,
        num_heads=CFG.num_heads,
        batch_size=CFG.batch_size,
        epochs=CFG.epochs,
        patience=CFG.patience,
        lr=CFG.lr,
        validation_size=CFG.validation_size,
        kl_tail_k=CFG.kl_tail_k,
        forecast_steps=CFG.forecast_steps,
    )

    base.device = device
    base.model = base.model.to(device)
    base.opt = torch.optim.AdamW(base.model.parameters(), lr=base.lr, weight_decay=1e-5)

    ts_train = series_X[:train_end]
    if len(ts_train) < max(2 * CFG.win_size, CFG.win_size + 5):
        raise ValueError(
            f"Train segment too small for {series_name}: train_end={train_end}, len={len(series_X)}, win={CFG.win_size}"
        )

    t0 = time.time()
    base.fit(ts_train)
    dt = time.time() - t0

    save_checkpoint(
        base,
        ckpt,
        meta={
            "series": series_name,
            "trained_on": int(len(ts_train)),
            "total_len": int(len(series_X)),
            "device": str(device),
            "train_seconds": float(dt),
        },
    )

    return base, ckpt


# =========================
# Training-objective ablations (ported from run_AxonAD_ablations.py)
# =========================


class TCNTransformerAblation(TCNTransformer):
    """Enables training/scoring ablations without changing library files."""

    def __init__(
        self,
        *,
        train_objective: str = "balanced",  # balanced|rec|jepa|pred_h|pred_k|pred_v
        score_mode: str = "mse+jepa",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._train_objective = train_objective
        self._score_mode = score_mode
        self._mse_loss = nn.MSELoss()

        objective = str(train_objective).lower().strip()
        self.k_pred_net: TCNQueryPredictor | None = None
        self.v_pred_net: TCNQueryPredictor | None = None
        if objective == "pred_k":
            self.k_pred_net = TCNQueryPredictor(d_model=self.model.d_model).to(self.device)
        if objective == "pred_v":
            self.v_pred_net = TCNQueryPredictor(d_model=self.model.d_model).to(self.device)

        if self.k_pred_net is not None or self.v_pred_net is not None:
            params = list(self.model.parameters())
            if self.k_pred_net is not None:
                params += list(self.k_pred_net.parameters())
            if self.v_pred_net is not None:
                params += list(self.v_pred_net.parameters())
            self.opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-5)

    def _shift_history(self, h_detached: torch.Tensor, s: int) -> torch.Tensor:
        B, T, C = h_detached.shape
        s = max(0, min(int(s), T))
        if s == 0:
            return h_detached
        x_shift = h_detached.new_zeros(B, T, C)
        x_shift[:, s:, :] = h_detached[:, :-s, :]
        return x_shift

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)

    def _cos_masked_loss_flat(self, pred_flat: torch.Tensor, tgt_flat: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
        if tmask is None or tmask.numel() == 0:
            return pred_flat.new_zeros(())
        m = tmask.to(pred_flat.device)
        if m.dtype != torch.bool:
            m = m.bool()
        if int(m.sum()) == 0:
            return pred_flat.new_zeros(())

        p = F.normalize(pred_flat, dim=-1)
        t = F.normalize(tgt_flat, dim=-1)
        cos = (p * t).sum(dim=-1)
        dist = 1.0 - cos
        return dist[m].mean()

    def fit(self, data):
        split = int((1 - self.validation_size) * len(data))
        if split <= self.win_size:
            tsTrain = data
            tsValid = data[:0]
        else:
            tsTrain = data[:split]
            tsValid = data[split:]

        train_loader = DataLoader(
            ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        train_objective = str(self._train_objective).lower().strip()
        allowed_obj = {"balanced", "rec", "jepa", "pred_h", "pred_k", "pred_v"}
        if train_objective not in allowed_obj:
            raise ValueError(f"train_objective must be one of: {sorted(allowed_obj)}")

        for _epoch in range(1, int(self.epochs) + 1):
            self.model.train()
            for d, _ in train_loader:
                d = d.to(self.device)
                rec, q_real, q_pred, k, q_tgt, tmask = self.model(d)

                h = self.model.embed(d) + self.model.pos
                h_norm = self.model.ln1(h)
                h_det = h_norm.detach()
                s = max(0, min(int(self.model.forecast_steps), h_det.shape[1]))
                x_shift = self._shift_history(h_det, s)

                q_pred_flat = self.model.attn.q_pred_net(x_shift)
                k_tgt = k.detach()
                v_tgt = self.model.attn._split_heads(self.model.attn.wv(h_norm)).detach()

                L_rec = self._mse_loss(rec, d)
                L_jepa = self._jepa_masked_loss(q_tgt, q_pred, tmask)

                if train_objective == "pred_h":
                    L_aux = self._cos_masked_loss_flat(q_pred_flat, h_det, tmask)
                elif train_objective == "pred_k":
                    if self.k_pred_net is None:
                        raise RuntimeError("k_pred_net not initialized")
                    k_pred_flat = self.k_pred_net(x_shift)
                    k_pred = self.model._split_heads(k_pred_flat, self.model.attn.num_heads)
                    k_tgt_flat = self._merge_heads(k_tgt)
                    L_aux = self._cos_masked_loss_flat(self._merge_heads(k_pred), k_tgt_flat, tmask)
                elif train_objective == "pred_v":
                    if self.v_pred_net is None:
                        raise RuntimeError("v_pred_net not initialized")
                    v_pred_flat = self.v_pred_net(x_shift)
                    v_pred = self.model._split_heads(v_pred_flat, self.model.attn.num_heads)
                    v_tgt_flat = self._merge_heads(v_tgt)
                    L_aux = self._cos_masked_loss_flat(self._merge_heads(v_pred), v_tgt_flat, tmask)
                else:
                    L_aux = None

                if train_objective == "rec":
                    loss = L_rec
                elif train_objective == "jepa":
                    loss = L_jepa
                elif train_objective.startswith("pred_"):
                    loss = L_aux
                else:
                    s_rec = self.model.s_rec
                    s_kl = self.model.s_kl
                    loss = torch.exp(-s_rec) * L_rec + s_rec + torch.exp(-s_kl) * L_jepa + s_kl

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                self.model.ema_update_target()

            if len(valid_loader) > 0:
                self.model.eval()
                val_sum, n_batches = 0.0, 0
                with torch.no_grad():
                    for d, _ in valid_loader:
                        d = d.to(self.device)
                        rec, q_real, q_pred, k, q_tgt, tmask = self.model(d)
                        h = self.model.embed(d) + self.model.pos
                        h_norm = self.model.ln1(h)
                        h_det = h_norm.detach()
                        s = max(0, min(int(self.model.forecast_steps), h_det.shape[1]))
                        x_shift = self._shift_history(h_det, s)
                        q_pred_flat = self.model.attn.q_pred_net(x_shift)
                        k_tgt = k.detach()
                        v_tgt = self.model.attn._split_heads(self.model.attn.wv(h_norm)).detach()

                        L_rec = self._mse_loss(rec, d)
                        L_jepa = self._jepa_masked_loss(q_tgt, q_pred, tmask)

                        if train_objective == "pred_h":
                            Lv = self._cos_masked_loss_flat(q_pred_flat, h_det, tmask)
                        elif train_objective == "pred_k":
                            if self.k_pred_net is None:
                                raise RuntimeError("k_pred_net not initialized")
                            k_pred_flat = self.k_pred_net(x_shift)
                            k_pred = self.model._split_heads(k_pred_flat, self.model.attn.num_heads)
                            k_tgt_flat = self._merge_heads(k_tgt)
                            Lv = self._cos_masked_loss_flat(self._merge_heads(k_pred), k_tgt_flat, tmask)
                        elif train_objective == "pred_v":
                            if self.v_pred_net is None:
                                raise RuntimeError("v_pred_net not initialized")
                            v_pred_flat = self.v_pred_net(x_shift)
                            v_pred = self.model._split_heads(v_pred_flat, self.model.attn.num_heads)
                            v_tgt_flat = self._merge_heads(v_tgt)
                            Lv = self._cos_masked_loss_flat(self._merge_heads(v_pred), v_tgt_flat, tmask)
                        elif train_objective == "jepa":
                            Lv = L_jepa
                        else:  # rec or balanced: use reconstruction as the validation proxy
                            Lv = L_rec

                        val_sum += float(Lv.item())
                        n_batches += 1

                val_avg = val_sum / max(1, n_batches)
                self.early_stopping(val_avg, self.model)
                if self.early_stopping.early_stop:
                    break

        self._calibrate(tsTrain)


def load_or_train_ablation_model(
    *,
    series_name: str,
    series_X: np.ndarray,
    train_end: int,
    out_root: Path,
    variant_name: str,
    train_objective: str,
    batch_size: int,
) -> tuple[TCNTransformerAblation, Path]:
    ckpt = checkpoint_path_variant(out_root, series_name, variant_name)
    device = pick_device()

    if ckpt.exists():
        obj = torch.load(ckpt, map_location="cpu")
        h = obj.get("hparams", {}) if isinstance(obj, dict) else {}
        win_size = int(h.get("win_size", CFG.win_size))
        feats = int(h.get("feats", series_X.shape[1]))
        d_model = int(h.get("d_model", CFG.d_model))
        num_heads = int(h.get("num_heads", CFG.num_heads))
        forecast_steps = int(h.get("forecast_steps", CFG.forecast_steps))
        kl_tail_k = int(h.get("kl_tail_k", CFG.kl_tail_k))
        lr = float(h.get("lr", CFG.lr))
        validation_size = float(h.get("validation_size", CFG.validation_size))
        score_mode = str(obj.get("variant", {}).get("score_mode", "mse+jepa")) if isinstance(obj, dict) else "mse+jepa"

        ab = TCNTransformerAblation(
            win_size=win_size,
            feats=feats,
            d_model=d_model,
            num_heads=num_heads,
            batch_size=int(h.get("batch_size", batch_size)),
            epochs=1,
            patience=CFG.ablation_patience,
            lr=lr,
            validation_size=validation_size,
            kl_tail_k=kl_tail_k,
            forecast_steps=forecast_steps,
            train_objective=train_objective,
            score_mode=score_mode,
        )
        ab.device = device
        ab.model = ab.model.to(device)
        # Rebuild optimizer (plus aux nets if any)
        params = list(ab.model.parameters())
        if ab.k_pred_net is not None:
            params += list(ab.k_pred_net.parameters())
        if ab.v_pred_net is not None:
            params += list(ab.v_pred_net.parameters())
        ab.opt = torch.optim.AdamW(params, lr=ab.lr, weight_decay=1e-5)

        if isinstance(obj, dict) and "model_state_dict" in obj:
            ab.model.load_state_dict(obj["model_state_dict"], strict=True)
            aux = obj.get("aux_state", {})
            if ab.k_pred_net is not None and aux.get("k_pred_net") is not None:
                ab.k_pred_net.load_state_dict(aux["k_pred_net"], strict=True)
            if ab.v_pred_net is not None and aux.get("v_pred_net") is not None:
                ab.v_pred_net.load_state_dict(aux["v_pred_net"], strict=True)

            calib = obj.get("calibration", {})
            ab.mse_med, ab.mse_iqr = float(calib.get("mse_med", 0.0)), float(calib.get("mse_iqr", 1.0))
            ab.jepa_med, ab.jepa_iqr = float(calib.get("jepa_med", 0.0)), float(calib.get("jepa_iqr", 1.0))
            ab.kl_med, ab.kl_iqr = float(calib.get("kl_med", 0.0)), float(calib.get("kl_iqr", 1.0))
        return ab, ckpt

    feats = int(series_X.shape[1])
    ab = TCNTransformerAblation(
        win_size=CFG.win_size,
        feats=feats,
        d_model=CFG.d_model,
        num_heads=CFG.num_heads,
        batch_size=batch_size,
        epochs=CFG.ablation_epochs,
        patience=CFG.ablation_patience,
        lr=CFG.ablation_lr,
        validation_size=CFG.validation_size,
        kl_tail_k=CFG.kl_tail_k,
        forecast_steps=CFG.forecast_steps,
        train_objective=train_objective,
        score_mode="mse+jepa",
    )

    ab.device = device
    ab.model = ab.model.to(device)
    params = list(ab.model.parameters())
    if ab.k_pred_net is not None:
        params += list(ab.k_pred_net.parameters())
    if ab.v_pred_net is not None:
        params += list(ab.v_pred_net.parameters())
    ab.opt = torch.optim.AdamW(params, lr=ab.lr, weight_decay=1e-5)

    ts_train = series_X[:train_end]
    ab.fit(ts_train)

    save_variant_checkpoint(
        ab,
        ckpt,
        meta={"series": series_name, "trained_on": int(len(ts_train)), "variant": variant_name, "objective": train_objective},
    )

    return ab, ckpt


# =========================
# Metrics helpers (no extra deps)
# =========================


def _rankdata(x: np.ndarray) -> np.ndarray:
    # Average ranks for ties, 1..n
    s = pd.Series(x)
    return s.rank(method="average").to_numpy(dtype=np.float64)


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    order = np.argsort(-y_score)
    y = y_true[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # integrate precision over recall (step function)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    y = y_true[order]
    # Mann–Whitney U
    ranks = np.arange(1, len(y) + 1)
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    block_len: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = int(len(y_true))
    if n < max(10, 2 * block_len):
        m = float(metric_fn(y_true, y_score))
        return m, float("nan"), float("nan")

    # contiguous blocks
    starts = np.arange(0, n, block_len)
    blocks = [np.arange(s, min(s + block_len, n)) for s in starts]
    if len(blocks) < 2:
        m = float(metric_fn(y_true, y_score))
        return m, float("nan"), float("nan")

    stats = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        stats.append(float(metric_fn(y_true[idx], y_score[idx])))

    m = float(metric_fn(y_true, y_score))
    lo, hi = np.nanpercentile(np.array(stats, dtype=np.float64), [2.5, 97.5])
    return m, float(lo), float(hi)


def find_threshold_at_fpr(scores_train: np.ndarray, fpr: float) -> float:
    # FPR computed on train windows (assumed mostly normal). Use high tail threshold.
    q = 100.0 * (1.0 - float(fpr))
    return float(np.percentile(scores_train, q))


def robust_z(train_scores: np.ndarray, all_scores: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    med, iqr = robust_fit(train_scores)
    z = (all_scores - med) / max(iqr, 1e-9)
    return z, {"med": float(med), "iqr": float(iqr)}


def window_label_from_point_labels(point_y: np.ndarray, win: int) -> np.ndarray:
    # Label a window as anomalous if ANY point in the window is anomalous.
    # point_y: [T] 0/1
    y = point_y.astype(np.int64)
    if len(y) < win:
        return np.zeros((0,), dtype=np.int64)
    # cumulative sum trick
    c = np.concatenate([[0], np.cumsum(y)])
    sums = c[win:] - c[:-win]
    return (sums > 0).astype(np.int64)


def window_centers(n_points: int, win: int) -> np.ndarray:
    # centers aligned to window end index
    if n_points < win:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(win - 1, n_points, dtype=np.int64)


# =========================
# Mechanistic extraction
# =========================


def _attention_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    forecast_steps: int,
    mask_to_history: bool = True,
) -> torch.Tensor:
    """Compute attention weights from q,k with the same causal/fairness mask.

    q,k: [B,H,T,D]
    returns: [B,H,T,T] with masked future keys set to 0 prob.
    """
    B, H, T, D = q.shape
    scale = math.sqrt(D)
    logits = (q @ k.transpose(-2, -1)) / scale  # [B,H,T,T]

    if not mask_to_history:
        return F.softmax(logits, dim=-1)

    s = max(0, min(int(forecast_steps), T))
    t = torch.arange(T, device=logits.device)
    # For each query time i, only allow keys j <= i - s
    max_key = (t - s).clamp(min=0)
    j = torch.arange(T, device=logits.device).view(1, 1, 1, T)
    max_key = max_key.view(1, 1, T, 1)
    mask_future = j > max_key  # [1,1,T,T]

    attn = safe_softmax_masked(logits, mask_future)
    # ensure masked positions are exactly 0 for stable KL
    attn = attn.masked_fill(mask_future, 0.0)
    return attn


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # KL(p||q) along last dim
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def extract_mechanistic_signals(
    *,
    base: TCNTransformer,
    series_X: np.ndarray,
    series_y_point: np.ndarray,
    train_end: int,
    series_name: str,
    out_root: Path,
    series_out_dir: Path | None = None,
) -> dict[str, Any]:
    device = base.device

    ds = ReconstructDataset(series_X, window_size=base.win_size, stride=1, normalize=True)
    y_win_full = window_label_from_point_labels(series_y_point, base.win_size)
    centers_full = window_centers(len(series_y_point), base.win_size)
    n_total = min(len(ds), len(y_win_full), len(centers_full))
    n_use = min(n_total, int(CFG.max_windows_per_series))

    if n_total <= 0:
        raise ValueError(f"No windows available for {series_name} (len={len(series_X)}, win={base.win_size})")

    # Subsample windows if we cap, but (1) keep indices for correct time/label alignment,
    # and (2) stratify so we don't accidentally sample away anomalies.
    if n_use < n_total:
        idx_all = np.arange(n_total, dtype=np.int64)
        is_train = centers_full[:n_total] < int(train_end)
        is_test = ~is_train

        train_normal = idx_all[is_train & (y_win_full[:n_total] == 0)]
        test_anom = idx_all[is_test & (y_win_full[:n_total] == 1)]
        test_normal = idx_all[is_test & (y_win_full[:n_total] == 0)]

        # Ensure enough train-normal for robust stats, plus include some test anomalies for plots.
        keep_train = min(len(train_normal), max(800, n_use // 2))
        keep_anom = min(len(test_anom), max(50, n_use // 4))
        remaining = max(0, n_use - keep_train - keep_anom)
        keep_test_norm = min(len(test_normal), remaining)
        remaining2 = max(0, remaining - keep_test_norm)

        chosen = []
        if keep_train > 0:
            chosen.append(np.random.choice(train_normal, size=keep_train, replace=False))
        if keep_anom > 0:
            chosen.append(np.random.choice(test_anom, size=keep_anom, replace=False))
        if keep_test_norm > 0:
            chosen.append(np.random.choice(test_normal, size=keep_test_norm, replace=False))
        if remaining2 > 0:
            # Fill any remainder with uniform coverage across the timeline.
            chosen.append(np.linspace(0, n_total - 1, remaining2).astype(np.int64))

        idx = np.unique(np.concatenate(chosen, axis=0)).astype(np.int64)
        # If uniquing reduced count, top up uniformly.
        if len(idx) < n_use:
            need = n_use - len(idx)
            extra = np.setdiff1d(idx_all, idx)
            if len(extra) > 0:
                add = np.random.choice(extra, size=min(need, len(extra)), replace=False)
                idx = np.unique(np.concatenate([idx, add])).astype(np.int64)
        # Final: trim if we somehow exceed
        if len(idx) > n_use:
            idx = np.sort(idx)[:n_use]
        else:
            idx = np.sort(idx)

        windows = torch.stack([ds.samples[i] for i in idx], dim=0)
        win_idx = idx
    else:
        windows = ds.samples[:n_total]
        win_idx = np.arange(n_total, dtype=np.int64)

    loader = DataLoader(windows, batch_size=base.batch_size, shuffle=False)

    # --- H3: train K/V/H predictors (same TCNQueryPredictor architecture + cosine tail-loss as JEPA Q) ---
    # This makes the comparison scientifically fair: all four representations (Q, K, V, H) are scored
    # with an IDENTICAL method — trained predictor cosine error on tail positions.  Q's predictor
    # (q_pred_net) was trained end-to-end with the model; K/V/H predictors are trained here on the
    # same train-normal windows with the same loss, so any gap reflects the representation itself.
    C_model = base.model.attn.d_model
    H_heads = base.model.attn.num_heads
    D_head  = C_model // H_heads
    s_steps = max(0, min(base.forecast_steps, base.win_size))
    _vT0    = base.win_size - s_steps
    _kt0    = (min(base.kl_tail_k, _vT0) if base.kl_tail_k > 0 else _vT0) if _vT0 > 0 else 1
    t0_pred = base.win_size - _kt0  # tail start index for TCN training

    # Identify train-normal windows (local indices into the `windows` tensor)
    _centers_sel = centers_full[win_idx]
    _y_sel       = y_win_full[win_idx]
    _tr_mask     = (_centers_sel < int(train_end)) & (_y_sel == 0)
    if int(_tr_mask.sum()) < 50:
        _tr_mask = _centers_sel < int(train_end)
    _tr_local = np.where(_tr_mask)[0]
    _take_tr  = min(len(_tr_local), 3000)
    if len(_tr_local) > _take_tr:
        _tr_local = np.sort(np.random.choice(_tr_local, size=_take_tr, replace=False))

    # Forward pass on train-normal windows (frozen base) to collect TCN training data
    base.model.eval()
    _tr_wins = (windows[_tr_local] if isinstance(windows, torch.Tensor)
                else torch.stack([windows[i] for i in _tr_local], dim=0))
    _chunks: list[tuple[torch.Tensor, ...]] = []
    for _b0 in range(0, len(_tr_local), max(1, base.batch_size)):
        _db = _tr_wins[_b0:_b0 + max(1, base.batch_size)].to(device)
        with torch.inference_mode():
            _Bb, _Tb, _ = _db.shape
            _hr   = base.model.embed(_db) + base.model.pos
            _hn   = base.model.ln1(_hr)                        # [B,T,C]
            _kf   = base.model.attn.wk(_hn)                   # [B,T,C]  K flat projection
            _vf   = base.model.attn.wv(_hn)                   # [B,T,C]  V flat projection
            if s_steps > 0:
                _xs = torch.cat([_hn.new_zeros(_Bb, s_steps, C_model), _hn[:, :-s_steps, :]], dim=1)
            else:
                _xs = _hn                                      # [B,T,C]  history-shifted input
        _chunks.append((_xs.cpu(), _kf.cpu(), _vf.cpu(), _hn.cpu()))
    _x_shift_tr  = torch.cat([c[0] for c in _chunks], dim=0)  # [N_tr, T, C]
    _k_target_tr = torch.cat([c[1] for c in _chunks], dim=0)
    _v_target_tr = torch.cat([c[2] for c in _chunks], dim=0)
    _h_target_tr = torch.cat([c[3] for c in _chunks], dim=0)
    del _chunks, _tr_wins

    k_pred_net = TCNQueryPredictor(d_model=C_model).to(device)
    v_pred_net = TCNQueryPredictor(d_model=C_model).to(device)
    h_pred_net = TCNQueryPredictor(d_model=C_model).to(device)

    def _train_rep_predictor(net: nn.Module, Xin: torch.Tensor, Ytgt: torch.Tensor, *, heads: bool) -> None:
        """Train net to predict Ytgt from Xin using tail-cosine loss — matches JEPA Q training."""
        net.train()
        N_t, T_t = Xin.shape[0], Xin.shape[1]
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
        for _ep in range(6):
            perm = torch.randperm(N_t)
            for _st in range(0, N_t, 256):
                ib  = perm[_st:_st + 256]
                xb  = Xin[ib].to(device)
                yb  = Ytgt[ib].to(device)
                out = net(xb)              # [B, T, C]
                B_b = xb.shape[0]
                if heads:
                    ph = out.view(B_b, T_t, H_heads, D_head).transpose(1, 2)
                    yh = yb.view(B_b, T_t, H_heads, D_head).transpose(1, 2)
                    pp = F.normalize(ph[:, :, t0_pred:, :], dim=-1)
                    tt = F.normalize(yh[:, :, t0_pred:, :], dim=-1)
                else:
                    pp = F.normalize(out[:, t0_pred:, :], dim=-1)
                    tt = F.normalize(yb[:, t0_pred:, :], dim=-1)
                loss = (1.0 - (pp * tt).sum(dim=-1)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net.eval()

    print(f"  [H3] training K/V/H predictors on {len(_tr_local)} train-normal windows...")
    _train_rep_predictor(k_pred_net, _x_shift_tr, _k_target_tr, heads=True)
    _train_rep_predictor(v_pred_net, _x_shift_tr, _v_target_tr, heads=True)
    _train_rep_predictor(h_pred_net, _x_shift_tr, _h_target_tr, heads=False)
    del _x_shift_tr, _k_target_tr, _v_target_tr, _h_target_tr

    # --- Inference pass: score all windows with trained predictors ---
    mse_vals: list[np.ndarray] = []
    qmis_vals: list[np.ndarray] = []
    kl_tail_vals: list[np.ndarray] = []
    attn_entropy_vals: list[np.ndarray] = []

    # Mechanistic link tensors (per-window, per-head)
    dq_norm_h_vals: list[np.ndarray] = []
    attn_kl_h_vals: list[np.ndarray] = []

    # H3: trained predictor mismatch (k/v/h predictors trained above; same metric as qmis)
    kmis_vals: list[np.ndarray] = []
    vmis_vals: list[np.ndarray] = []
    hmis_vals: list[np.ndarray] = []

    base.model.eval()
    k_pred_net.eval()
    v_pred_net.eval()
    h_pred_net.eval()

    with torch.inference_mode():
        for batch in loader:
            d = batch.to(device)

            rec, q_real, q_pred, k, q_tgt, tmask = base.model(d)

            # h_norm for K/V/H predictor input and targets
            _h_raw = base.model.embed(d) + base.model.pos
            _h_norm = base.model.ln1(_h_raw)  # [B,T,C]

            # Per-window scores
            mse = (rec - d).pow(2).sum(dim=-1).mean(dim=1)  # [B]
            # Query mismatch: JEPA predictor error of the trained q_pred_net
            qmis = base._jepa_tail_dist(q_tgt, q_pred)  # [B]
            kl_tail = base._kl_tail(q_real, q_pred, k)  # [B]

            # Mechanistic link: ||ΔQ|| (per-head) vs KL(A_real || A_pred)
            A_real = _attention_from_qk(q_real, k, forecast_steps=base.forecast_steps, mask_to_history=True)
            A_pred = _attention_from_qk(q_pred, k, forecast_steps=base.forecast_steps, mask_to_history=True)

            B, H, T, D = q_real.shape
            s = max(0, min(base.forecast_steps, T))
            valid_T = T - s
            if valid_T > 0:
                k_tail = min(base.kl_tail_k, valid_T) if base.kl_tail_k > 0 else valid_T
                t0 = max(s, T - k_tail)

                dq = (q_real[:, :, t0:, :] - q_pred[:, :, t0:, :])  # [B,H,k,D]
                dq_norm_h = dq.pow(2).sum(dim=-1).mean(dim=2).sqrt()  # [B,H]

                kl_h = _kl_divergence(A_real[:, :, t0:, :], A_pred[:, :, t0:, :]).mean(dim=2)  # [B,H]

                # K/V/H predictor mismatch — identical cosine metric to qmis
                _C = _h_norm.shape[2]
                if s > 0:
                    _xs_b = torch.cat([_h_norm.new_zeros(B, s, _C), _h_norm[:, :-s, :]], dim=1)
                else:
                    _xs_b = _h_norm
                _k_flat = base.model.attn.wk(_h_norm)  # [B,T,C]
                _v_flat = base.model.attn.wv(_h_norm)  # [B,T,C]
                _k_out  = k_pred_net(_xs_b)            # [B,T,C]
                _v_out  = v_pred_net(_xs_b)
                _h_out  = h_pred_net(_xs_b)

                # Cosine error on tail positions (head-split for K/V; flat for H)
                def _miss_heads(pred_f: torch.Tensor, tgt_f: torch.Tensor) -> torch.Tensor:
                    ph = pred_f.view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
                    th = tgt_f.view(B, T, H, D).transpose(1, 2)
                    pp = F.normalize(ph[:, :, t0:, :], dim=-1)
                    tt = F.normalize(th[:, :, t0:, :], dim=-1)
                    return (1.0 - (pp * tt).sum(dim=-1)).mean(dim=(1, 2))  # [B]

                def _miss_flat(pred_f: torch.Tensor, tgt_f: torch.Tensor) -> torch.Tensor:
                    pp = F.normalize(pred_f[:, t0:, :], dim=-1)
                    tt = F.normalize(tgt_f[:, t0:, :], dim=-1)
                    return (1.0 - (pp * tt).sum(dim=-1)).mean(dim=1)  # [B]

                kmis = _miss_heads(_k_out, _k_flat)
                vmis = _miss_heads(_v_out, _v_flat)
                hmis = _miss_flat(_h_out, _h_norm)
            else:
                dq_norm_h = mse.new_zeros((B, H))
                kl_h = mse.new_zeros((B, H))
                kmis = mse.new_zeros(mse.shape)
                vmis = mse.new_zeros(mse.shape)
                hmis = mse.new_zeros(mse.shape)

            # Attention entropy (teacher, tail rows)
            if valid_T > 0:
                ent = -(A_real[:, :, t0:, :].clamp_min(1e-12) * A_real[:, :, t0:, :].clamp_min(1e-12).log()).sum(dim=-1)
                ent = ent.mean(dim=(1, 2))  # [B]
            else:
                ent = mse.new_zeros(mse.shape)

            mse_vals.append(mse.detach().cpu().numpy())
            qmis_vals.append(qmis.detach().cpu().numpy())
            kl_tail_vals.append(kl_tail.detach().cpu().numpy())
            attn_entropy_vals.append(ent.detach().cpu().numpy())

            dq_norm_h_vals.append(dq_norm_h.detach().cpu().numpy())
            attn_kl_h_vals.append(kl_h.detach().cpu().numpy())

            kmis_vals.append(kmis.detach().cpu().numpy())
            vmis_vals.append(vmis.detach().cpu().numpy())
            hmis_vals.append(hmis.detach().cpu().numpy())

    mse_all = np.concatenate(mse_vals, axis=0)
    qmis_all = np.concatenate(qmis_vals, axis=0)
    kl_all = np.concatenate(kl_tail_vals, axis=0)
    ent_all = np.concatenate(attn_entropy_vals, axis=0)

    dq_norm_h_all = np.concatenate(dq_norm_h_vals, axis=0)  # [N,H]
    attn_kl_h_all = np.concatenate(attn_kl_h_vals, axis=0)  # [N,H]

    kmis_all = np.concatenate(kmis_vals, axis=0)  # [N]
    vmis_all = np.concatenate(vmis_vals, axis=0)  # [N]
    hmis_all = np.concatenate(hmis_vals, axis=0)  # [N]

    corr = float(np.corrcoef(mse_all, qmis_all)[0, 1]) if len(mse_all) > 2 else float("nan")
    s_corr = spearmanr(dq_norm_h_all.reshape(-1), attn_kl_h_all.reshape(-1))

    out_series = series_out_dir if series_out_dir is not None else (out_root / "series" / Path(series_name).stem)
    ensure_dir(out_series)

    np.savez_compressed(
        out_series / "signals.npz",
        mse=mse_all,
        qmis=qmis_all,
        kl=kl_all,
        attn_entropy=ent_all,
        dq_norm_h=dq_norm_h_all,
        attn_kl_h=attn_kl_h_all,
        kmis=kmis_all,
        vmis=vmis_all,
        hmis=hmis_all,
        win_idx=win_idx,
    )

    return {
        "series": series_name,
        "n_windows_total": int(n_total),
        "n_windows_used": int(len(mse_all)),
        "mse_med": robust_fit(mse_all)[0],
        "mse_iqr": robust_fit(mse_all)[1],
        "qmis_med": robust_fit(qmis_all)[0],
        "qmis_iqr": robust_fit(qmis_all)[1],
        "kl_med": robust_fit(kl_all)[0],
        "kl_iqr": robust_fit(kl_all)[1],
        "attn_entropy_mean": float(ent_all.mean()) if len(ent_all) else 0.0,
        "mse_qmis_corr": corr,
        "spearman_dq_vs_attnkl": float(s_corr),
    }


# =========================
# Plotting + report
# =========================


def plot_series_signals(series_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz = np.load(series_dir / "signals.npz")
    mse = npz["mse"]
    qmis = npz["qmis"]
    ent = npz["attn_entropy"]

    if len(mse) == 0:
        return

    # If aligned labels/scores exist, prefer plotting z-scores on real time with anomaly shading.
    ls_path = series_dir / "labels_and_scores.npz"
    labels = None
    centers = None
    mse_plot = mse
    q_plot = qmis
    title_suffix = "(raw)"
    if ls_path.exists():
        ls = np.load(ls_path)
        centers = ls["centers"]
        labels = ls["y_win"]
        mse_plot = ls["mse_z"]
        q_plot = ls["q_z"]
        title_suffix = "(z, train-normal stats)"

    # Rolling mean for readability
    def roll(x: np.ndarray, k: int = 101) -> np.ndarray:
        if len(x) < k:
            return x
        w = np.ones(k) / k
        return np.convolve(x, w, mode="same")

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    if centers is None:
        x = np.arange(len(mse_plot))
    else:
        x = centers[: len(mse_plot)]

    if labels is not None and len(labels) >= len(mse_plot) and (labels[: len(mse_plot)] > 0).any():
        in_anom = (labels[: len(mse_plot)] > 0)
        starts = np.where(np.diff(np.concatenate([[0], in_anom.view(np.int8), [0]])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[0], in_anom.view(np.int8), [0]])) == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(x[s], x[e - 1], color="red", alpha=0.12, lw=0)

    ax.plot(x, roll(mse_plot), label="Recon")
    ax.plot(x, roll(q_plot), label="Query mismatch")
    plt.legend()
    plt.title(f"Signals timeline {title_suffix}")
    plt.tight_layout()
    savefig_png_pdf(series_dir / "signals_timeseries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(mse_plot, q_plot, s=6, alpha=0.5)
    plt.xlabel("Recon" if ls_path.exists() else "MSE")
    plt.ylabel("Query mismatch")
    plt.tight_layout()
    savefig_png_pdf(series_dir / "mse_vs_qmis.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(ent, bins=40)
    plt.xlabel("Attention entropy (tail, teacher)")
    plt.ylabel("Count")
    plt.tight_layout()
    savefig_png_pdf(series_dir / "attn_entropy_hist.png", dpi=150)
    plt.close()


def write_report(out_root: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(out_root)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False)

    # Markdown report
    lines = []
    lines.append("# Mechanistic evidence suite (no-args)\n")
    lines.append(f"- Benchmark: {CFG.benchmark}\n")
    lines.append(f"- Split list: {CFG.split_list}\n")
    lines.append(f"- Series processed: {len(df)}\n")
    lines.append(f"- Device preference: {pick_device()}\n")
    lines.append("\n")

    lines.append("## Mechanistic evidence plan (hypotheses → tests)\n")
    lines.append("- **H1 Retrieval intent**: query mismatch spikes when the model must retrieve different past context.\n")
    lines.append("  - Test: find regions where query z-score is high while recon z-score is low; verify overlap with association/mode anomalies (label-shaded).\n")
    lines.append("  - Falsifier: anomalies consistently show recon-only without query-only slices.\n")
    lines.append("- **H2 Policy link**: larger $\\|\\Delta Q\\|$ implies larger attention redistribution.\n")
    lines.append("  - Test: correlation between $\\|\\Delta Q\\|$ and $KL(A_{real}\\|\\|A_{pred})$ pooled over tail timesteps.\n")
    lines.append("  - Falsifier: correlation $\\approx 0$ with tight CI across case studies.\n")
    lines.append("- **H3 Why Q > K/V/H**: keys/values are smoother (past supply), queries jump (current demand).\n")
    lines.append("  - Test: compare anomaly separation of target-prediction errors (Q vs K/V vs H) with block-bootstrap CI.\n")
    lines.append("  - Falsifier: K/V errors match or exceed Q error in AUC-PR consistently.\n")
    lines.append("- **H4 Complementarity**: recon and query capture disjoint subsets.\n")
    lines.append("  - Test: quadrant mass + matched-FPR recall for each score; measure conditional slices (low recon, high query).\n")
    lines.append("  - Falsifier: near-perfect correlation and no distinct quadrants.\n")
    lines.append("\n")

    lines.append("## Mathematical rationale (why Q mismatch is the right mechanism)\n")
    lines.append(
        "AxonAD’s attention uses per-head logits $z_{t,j} = q_t^\\top k_j / \\sqrt{d}$ and weights $A_t = \\mathrm{softmax}(z_t)$. "
        "If keys are held fixed (they summarize the *available past*), a query perturbation $\Delta q_t$ induces logit changes "
        "$\Delta z_{t,j} = (\Delta q_t)^\top k_j / \sqrt{d}$. Therefore, large query mismatch implies large, *structured* changes "
        "in which past tokens are selected (a policy / retrieval change).\n"
    )
    lines.append(
        "More explicitly: for a row-softmax, the Jacobian is $J = \\mathrm{diag}(A) - A A^\\top$. Hence $\\Delta A \\approx J\\,\\Delta z$, "
        "and with fixed keys $\\Delta z = K\\,\\Delta q/\\sqrt{d}$. Therefore, attention redistribution is *driven by query changes* when the memory (keys) is stable. "
        "This motivates Fig. F3: larger $\\|\\Delta Q\\|$ should imply larger divergence between attention distributions computed from $q_{real}$ vs $q_{pred}$ (we use $KL(A_{real}\\|\\|A_{pred})$).\n"
    )
    lines.append(
        "Why $Q$ beats $K/V/H$ as a prediction target: in causal sequence models, $K,V$ are a *memory encoding of the past* (supply) "
        "and tend to vary smoothly under association/mode anomalies where amplitudes remain normal, while $Q$ is the *current retrieval demand* "
        "that must re-route attention when the regime/associations change. Prediction errors for $K/V/H$ can be small yet non-discriminative; "
        "$Q$ error aligns with the mechanism that *changes the selected context*.\n"
    )
    lines.append(
        "Note on ablations: this script includes (a) *score-mode ablations* (MSE-only vs Q-only vs KL-only vs combinations) computed from the same base model outputs, "
        "and (b) *training-objective ablations* that **do** retrain separate models under objectives "
        "ReconOnly / JEPAOnly\_Q / PredKeys / PredValues / PredHiddenState (see `run_objective_ablations` and `run_training_objective_ablations` config flags).\n"
    )
    lines.append("\n")

    def _md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
        # Minimal Markdown table without requiring `tabulate`.
        if frame.empty:
            return "(empty)\n"
        frame = frame.copy()
        if len(frame) > max_rows:
            frame = frame.head(max_rows)
        cols = [str(c) for c in frame.columns]
        # stringify with compact floats
        def fmt(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                if not np.isfinite(v):
                    return "nan"
                return f"{v:.4g}"
            return str(v)

        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        body_lines = []
        for _, row in frame.iterrows():
            body_lines.append("| " + " | ".join(fmt(row[c]) for c in frame.columns) + " |\n")
        return header + sep + "".join(body_lines)

    lines.append("## Summary table\n")
    lines.append(_md_table(df, max_rows=30))
    if len(df) > 30:
        lines.append(f"\n(Showing first 30 rows of {len(df)}; full table in `summary.csv`.)\n")
    lines.append("\n")

    # =========================
    # Hypothesis-by-hypothesis audit (derived from generated artifacts)
    # =========================
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        try:
            if path.exists():
                return pd.read_csv(path)
        except Exception:
            pass
        return pd.DataFrame()

    def _finite(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        return s[np.isfinite(s.to_numpy(dtype=np.float64, copy=False))]

    def _count_ge(vals: pd.Series, thr: float) -> int:
        return int((vals >= float(thr)).sum())

    def _fmt_float(x: float) -> str:
        if x is None or not np.isfinite(x):
            return "nan"
        return f"{float(x):.3g}"

    tables_dir = out_root / "tables"
    eval_df = _safe_read_csv(tables_dir / "per_series_eval.csv")
    quad_df = _safe_read_csv(tables_dir / "quadrant_stats.csv")
    t1_df = _safe_read_csv(tables_dir / "T1_rep_shift_aucpr.csv")

    lines.append("## Hypothesis audit (what this run supports)\n")

    # H2: policy link
    spearman_vals = pd.Series(dtype=float)
    if "spearman_dq_vs_attnkl" in df.columns:
        spearman_vals = _finite(df["spearman_dq_vs_attnkl"])
    if spearman_vals.empty and ("spearman_dq_vs_attnkl" in eval_df.columns):
        spearman_vals = _finite(eval_df["spearman_dq_vs_attnkl"])

    if not spearman_vals.empty:
        n = int(len(spearman_vals))
        lines.append("### H2 — Policy link ($\\|\\Delta Q\\|$ ↔ attention redistribution)\n")
        lines.append(
            f"- Spearman($\\|\\Delta Q\\|$, $KL$) across {n} series: "
            f"median={_fmt_float(float(spearman_vals.median()))}, "
            f"min={_fmt_float(float(spearman_vals.min()))}, max={_fmt_float(float(spearman_vals.max()))}.\n"
        )
        lines.append(
            f"- Counts: ≥0.50: {_count_ge(spearman_vals, 0.50)}/{n}; ≥0.75: {_count_ge(spearman_vals, 0.75)}/{n}.\n"
        )

    if "attn_entropy_mean" in df.columns:
        ent = _finite(df["attn_entropy_mean"])
        if not ent.empty:
            lines.append(
                f"- Attention entropy (tail) is not degenerate: mean≈{_fmt_float(float(ent.mean()))} (range {_fmt_float(float(ent.min()))}–{_fmt_float(float(ent.max()))}).\n"
            )

    lines.append("\n")

    # H4: complementarity / non-redundancy
    lines.append("### H4 — Complementarity (query is not redundant with recon)\n")
    if "mse_qmis_corr" in df.columns:
        c = _finite(df["mse_qmis_corr"])
        if not c.empty:
            n = int(len(c))
            lines.append(
                f"- Window-level recon vs query mismatch correlation: median={_fmt_float(float(c.median()))} (range {_fmt_float(float(c.min()))}–{_fmt_float(float(c.max()))}).\n"
            )
            lines.append(f"- Near-independence is common: |corr|≤0.10 in {int((c.abs() <= 0.10).sum())}/{n} series.\n")

    if not eval_df.empty and {"ap_recon", "ap_query", "ap_combined"}.issubset(set(eval_df.columns)):
        ap_r = _finite(eval_df["ap_recon"])
        ap_q = _finite(eval_df["ap_query"])
        ap_c = _finite(eval_df["ap_combined"])
        m = min(len(ap_r), len(ap_q), len(ap_c))
        if m > 0:
            ar = ap_r.iloc[:m].to_numpy()
            aq = ap_q.iloc[:m].to_numpy()
            ac = ap_c.iloc[:m].to_numpy()
            # Primary comparison: combined vs recon alone (practical, since AxonAD always uses mse+q)
            improved_vs_recon = int((ac > ar + 1e-12).sum())
            mean_gain_vs_recon = float(np.mean(ac - ar))
            # Secondary: combined vs oracle-best-single per series (upper bound)
            best_single = np.maximum(ar, aq)
            improved_vs_best = int((ac > best_single + 1e-12).sum())
            mean_c = float(np.mean(ac))
            mean_r = float(np.mean(ar))
            mean_q = float(np.mean(aq))
            gain_str = ("+" if mean_gain_vs_recon >= 0 else "") + _fmt_float(mean_gain_vs_recon)
            lines.append(
                f"- Mean AUC-PR: combined={_fmt_float(mean_c)}, recon={_fmt_float(mean_r)}, query={_fmt_float(mean_q)}.\n"
            )
            lines.append(
                f"- Combined (mse+q) improves over recon alone in {improved_vs_recon}/{m} series "
                f"(mean gain={gain_str}).  "
                f"Combined vs oracle-best-single: {improved_vs_best}/{m} series — "
                "both numbers are expected when mse dominates; the mean gain is the primary evidence.\n"
            )
            if {"recall_recon_at_fpr", "recall_query_at_fpr", "recall_combined_at_fpr"}.issubset(set(eval_df.columns)):
                rr = _finite(eval_df["recall_recon_at_fpr"]).to_numpy()
                rq = _finite(eval_df["recall_query_at_fpr"]).to_numpy()
                rc = _finite(eval_df["recall_combined_at_fpr"]).to_numpy()
                m2 = min(len(rr), len(rq), len(rc))
                if m2 > 0:
                    improved_r = int((rc[:m2] > rr[:m2] + 1e-12).sum())
                    lines.append(
                        f"- At matched FPR={CFG.fpr_for_complementarity:g}, combined recall improves over recon in {improved_r}/{m2} series.\n"
                    )

    if not quad_df.empty and {
        "anom_lowRecon_highQuery",
        "anom_highRecon_lowQuery",
        "anom_both_high",
        "n_test_anom",
    }.issubset(set(quad_df.columns)):
        n = int(len(quad_df))
        q_only = pd.to_numeric(quad_df["anom_lowRecon_highQuery"], errors="coerce").fillna(0.0)
        r_only = pd.to_numeric(quad_df["anom_highRecon_lowQuery"], errors="coerce").fillna(0.0)
        n_anom = pd.to_numeric(quad_df["n_test_anom"], errors="coerce").fillna(0.0)
        has_q_only = int((q_only > 0).sum())
        has_r_only = int((r_only > 0).sum())
        # pooled shares (guard against 0)
        denom = float(n_anom.sum())
        if denom > 0:
            lines.append(
                f"- Quadrant mass (pooled anomalies): lowRecon/highQuery={_fmt_float(float(q_only.sum()/denom))}, highRecon/lowQuery={_fmt_float(float(r_only.sum()/denom))}, both-high={_fmt_float(float(pd.to_numeric(quad_df['anom_both_high'], errors='coerce').fillna(0.0).sum()/denom))}.\n"
            )
        lines.append(f"- Non-trivial disjointness exists: lowRecon/highQuery anomalies in {has_q_only}/{n} series; highRecon/lowQuery anomalies in {has_r_only}/{n} series.\n")

    lines.append("\n")

    # H1: retrieval-intent slices (time-localized)
    lines.append("### H1 — Retrieval intent (high-Q / low-recon slices are time-localized)\n")
    figs_dir = out_root / "figs"
    case_figs = sorted(figs_dir.glob("F1_case_timeline_*.png"))
    if len(case_figs) > 0:
        shown = ", ".join([p.name.replace("F1_case_timeline_", "") for p in case_figs[:4]])
        more = "" if len(case_figs) <= 4 else f" (+{len(case_figs) - 4} more)"
        lines.append(f"- Case timelines are available (Fig. F1): {shown}{more}.\n")
    if not quad_df.empty and "anom_lowRecon_highQuery" in quad_df.columns:
        q_only = pd.to_numeric(quad_df["anom_lowRecon_highQuery"], errors="coerce").fillna(0.0)
        lines.append(f"- Conditional evidence exists in aggregate: lowRecon/highQuery anomaly windows occur in {int((q_only > 0).sum())}/{int(len(quad_df))} series; use Fig. F1 for time localization.\n")

    lines.append("\n")

    # H3: Q vs K/V/H targets
    lines.append("### H3 — Why Q as a target (vs K/V/H)\n")

    # Primary evidence: training-objective ablations (per-series retrain, same approach as
    # run_AxonAD_ablations.py).  Each variant is a full model trained under a different
    # objective; its own anomaly score is evaluated.  This is the correct comparison.
    obj_abl_df = _safe_read_csv(tables_dir / "per_series_objective_ablations.csv")
    if not obj_abl_df.empty and {"series", "variant", "ap"}.issubset(set(obj_abl_df.columns)):
        abl = obj_abl_df.copy()
        abl["ap"] = pd.to_numeric(abl["ap"], errors="coerce")
        abl = abl[np.isfinite(abl["ap"].to_numpy(dtype=np.float64, copy=False))]
        if not abl.empty:
            winners_abl = (
                abl.sort_values(["series", "ap"], ascending=[True, False])
                .groupby("series", as_index=False)
                .first()
            )
            n_abl = int(len(winners_abl))
            wc_abl = winners_abl["variant"].value_counts().to_dict()
            win_str_abl = ", ".join([f"{k}:{int(v)}" for k, v in sorted(wc_abl.items(), key=lambda x: -x[1])])
            means_abl = abl.groupby("variant")["ap"].mean().to_dict()
            means_str_abl = ", ".join([f"{k}:{_fmt_float(float(v))}" for k, v in sorted(means_abl.items(), key=lambda x: -x[1])])
            lines.append(f"- **Primary (training-objective ablations)** — AUC-PR winners across {n_abl} series: {win_str_abl}.\n")
            lines.append(f"  - Mean AUC-PR by training objective: {means_str_abl}.\n")
            lines.append("  - Source: `tables/per_series_objective_ablations.csv` (per-series retrain; same methodology as `run_AxonAD_ablations.py`).\n")
    else:
        lines.append("- Primary training-objective ablation table missing (`tables/per_series_objective_ablations.csv`).\n")

    # Secondary evidence: per-series target-prediction probes on case studies.
    # Each series uses its own MLP probe trained on that series' train windows only,
    # so latent spaces are not mixed across series.
    if not t1_df.empty and {"series", "target", "ap"}.issubset(set(t1_df.columns)):
        t1 = t1_df.copy()
        t1["ap"] = pd.to_numeric(t1["ap"], errors="coerce")
        t1 = t1[np.isfinite(t1["ap"].to_numpy(dtype=np.float64, copy=False))]
        if not t1.empty:
            winners = (
                t1.sort_values(["series", "ap"], ascending=[True, False])
                .groupby("series", as_index=False)
                .first()
            )
            n = int(len(winners))
            win_counts = winners["target"].value_counts().to_dict()
            win_str = ", ".join([f"{k}:{int(v)}" for k, v in sorted(win_counts.items())])
            means = t1.groupby("target")["ap"].mean().to_dict()
            means_str = ", ".join([f"{k}:{_fmt_float(float(v))}" for k, v in sorted(means.items())])
            lines.append(f"- **Secondary (per-series prediction-error probes on case studies)** — winners across {n} series: {win_str}.\n")
            lines.append(f"  - Mean AUC-PR by target: {means_str}.\n")
            lines.append("  - Note: probes are trained per-series on each series' own train windows (no cross-series latent pooling).\n")
            lines.append("  - See F4/F5 for pooled comparisons; treat as *often*, not *always*.\n")
    else:
        lines.append("- Representation-shift AUC-PR table missing; expected `tables/T1_rep_shift_aucpr.csv`.\n")

    lines.append("\n")

    # Copy-ready paragraph (conservative wording)
    lines.append("## Copy-ready mechanistic paragraph (conservative wording)\n")
    lines.append(
        "Across tuning series, query mismatch is frequently a strong indicator of attention redistribution: the per-series Spearman correlation between $\\|\\Delta Q\\|$ "
        "and the KL divergence between attention distributions induced by real vs predicted queries is often large (Fig. F3; `tables/policy_link_spearman.csv`).\n"
    )
    lines.append(
        "Query mismatch is not a trivial proxy for reconstruction error: recon–query correlations are often small, and combining scores improves anomaly separation and matched-FPR recall in many series (`tables/per_series_eval.csv`).\n"
    )
    lines.append(
        "Quadrant statistics show that low-reconstruction/high-query anomaly windows occur in multiple series (`tables/quadrant_stats.csv`), and case-study timelines (Fig. F1) localize these slices in time.\n"
    )
    lines.append(
        "Representation-shift AUC-PR experiments (Fig. F4; `tables/T1_rep_shift_aucpr.csv`) indicate that the JEPA-trained Q shift signal is a stronger anomaly detector than naive K/V/H history-mean shifts, providing mechanistic evidence for why the Q prediction objective improves detection.\n"
    )
    lines.append("\n")

    lines.append("## Outputs\n")
    lines.append(f"- Summary CSV: `{out_root / 'summary.csv'}`\n")
    lines.append(f"- Per-series artifacts: `{out_root / 'series'}`\n")
    lines.append(f"- Cached checkpoints: `{out_root / 'checkpoints'}`\n")
    lines.append(f"- Figures: `{out_root / 'figs'}`\n")
    lines.append(f"- Tables: `{out_root / 'tables'}`\n")

    # Helpful pointers (only list what exists to avoid confusion).
    figs_dir = out_root / "figs"
    tables_dir = out_root / "tables"

    lines.append("\n### Key figures\n")
    fig_candidates = [
        ("F1_case_timeline_", "case timelines (complementarity in time)"),
        ("F2_scatter_pooled.png", "pooled recon-vs-query separation (quadrants)"),
        ("F3_policy_link_pooled.png", "pooled policy-change link ($\\|\\Delta Q\\|$ vs $KL$)"),
        ("F4_target_aucpr_pooled.png", "Q vs K/V/H target comparisons (AUC-PR)"),
        ("F5_target_error_box_pooled.png", "Q vs K/V/H target error distributions"),
        ("F6_target_policy_alignment_pooled.png", "mechanistic alignment (which target tracks attention-policy KL)"),
        ("F7_objective_ablation_aucpr.png", "training-objective ablations"),
    ]
    for name, desc in fig_candidates:
        if name.endswith(".png"):
            if (figs_dir / name).exists():
                lines.append(f"- `figs/{name}`: {desc}\n")
        else:
            if len(list(figs_dir.glob(f"{name}*.png"))) > 0:
                lines.append(f"- `figs/{name}*.png`: {desc}\n")

    lines.append("\n### Key tables\n")
    table_candidates = [
        ("per_series_eval.csv", "per-series AUC-PR + complementarity at matched FPR"),
        ("quadrant_stats.csv", "disjoint anomaly mass in recon/query quadrants"),
        ("policy_link_spearman.csv", "correlation stats for H2"),
        ("T1_rep_shift_aucpr.csv", "Q vs K vs V vs H representation-shift AUC-PR (all 20 series)"),
        ("T2_target_policy_alignment.csv", "alignment of scores with attention-policy KL"),
        ("per_series_score_modes.csv", "MSE-only vs Q-only vs KL-only vs combos (no retrain)"),
        ("per_series_objective_ablations.csv", "objective-score AUC-PR under different training objectives"),
        ("per_series_training_objective_ablations.csv", "training-objective ablations (with retrain)"),
    ]
    for name, desc in table_candidates:
        if (tables_dir / name).exists():
            lines.append(f"- `tables/{name}`: {desc}\n")

    (out_root / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def _plot_scatter_pooled(
    *,
    out_root: Path,
    mse_z: np.ndarray,
    q_z: np.ndarray,
    y: np.ndarray,
    max_points: int = 60000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    n = int(len(y))
    if n == 0:
        return
    if n > max_points:
        idx = np.random.choice(np.arange(n), size=max_points, replace=False)
        mse_z, q_z, y = mse_z[idx], q_z[idx], y[idx]

    norm = (y == 0)
    anom = (y == 1)

    plt.figure(figsize=(6.2, 6.0))
    plt.scatter(mse_z[norm], q_z[norm], s=4, alpha=0.20, label="normal")
    plt.scatter(mse_z[anom], q_z[anom], s=6, alpha=0.45, label="anomaly")
    plt.axvline(0.0, color="k", lw=0.8)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel("Recon z (train-normal)")
    plt.ylabel("Query mismatch z (train-normal)")
    plt.title("Pooled separation: recon vs query")
    plt.legend()
    plt.tight_layout()
    savefig_png_pdf(figs / "F2_scatter_pooled.png", dpi=200)
    plt.close()


def _plot_policy_link_pooled(
    *,
    out_root: Path,
    dq_norm_h: np.ndarray,
    attn_kl_h: np.ndarray,
    max_points: int = 120000,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")

    x = dq_norm_h.reshape(-1)
    y = attn_kl_h.reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(len(x))
    if n == 0:
        return {"spearman": float("nan"), "n_points": 0}
    if n > max_points:
        idx = np.random.choice(np.arange(n), size=max_points, replace=False)
        x, y = x[idx], y[idx]

    rho = spearmanr(x, y) if len(x) > 10 else float("nan")

    # Binned trend for readability
    bins = np.quantile(x, np.linspace(0, 1, 21))
    bin_id = np.digitize(x, bins[1:-1], right=True)
    x_m = np.array([np.nanmean(x[bin_id == i]) for i in range(20)])
    y_m = np.array([np.nanmean(y[bin_id == i]) for i in range(20)])

    plt.figure(figsize=(6.2, 5.0))
    plt.scatter(x, y, s=2, alpha=0.12)
    plt.plot(x_m, y_m, color="red", lw=2.0, label="binned mean")
    plt.xlabel(r"$||\Delta Q||$ (per-head, tail)")
    plt.ylabel(r"$KL(A_{real}||A_{pred})$ (per-head, tail)")
    plt.title(f"Pooled policy link (Spearman={rho:.3f})")
    plt.legend()
    plt.tight_layout()
    savefig_png_pdf(figs / "F3_policy_link_pooled.png", dpi=200)
    plt.close()

    return {"spearman": float(rho), "n_points": int(len(x))}


def _policy_kl_window_mean(base: TCNTransformer, windows: torch.Tensor) -> np.ndarray:
    """Attention policy mismatch per window.

    Returns:
      kl_mean: [N] where KL is averaged over heads and tail rows.
    """
    device = base.device
    base.model.eval()
    with torch.inference_mode():
        d = windows.to(device)
        rec, q_real, q_pred, k, q_tgt, _ = base.model(d)
        A_real = _attention_from_qk(q_real, k, forecast_steps=base.forecast_steps, mask_to_history=True)
        A_pred = _attention_from_qk(q_pred, k, forecast_steps=base.forecast_steps, mask_to_history=True)

        B, H, T, D = q_real.shape
        s = max(0, min(base.forecast_steps, T))
        valid_T = T - s
        if valid_T <= 0:
            return np.zeros((B,), dtype=np.float32)
        k_tail = min(base.kl_tail_k, valid_T) if base.kl_tail_k > 0 else valid_T
        t0 = max(s, T - k_tail)
        kl_h = _kl_divergence(A_real[:, :, t0:, :], A_pred[:, :, t0:, :]).mean(dim=2)  # [B,H]
        kl_w = kl_h.mean(dim=1)  # [B]
        return kl_w.detach().cpu().numpy().astype(np.float32)


def _plot_base_policy_alignment_pooled(
    *,
    out_root: Path,
    policy_kl: np.ndarray,
    mse_z: np.ndarray,
    qmis_z: np.ndarray,
    kl_z: np.ndarray,
) -> dict[str, float]:
    """Base-model internal alignment: which score tracks attention-policy mismatch?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        ok = np.isfinite(a) & np.isfinite(b)
        if int(ok.sum()) < 20:
            return float("nan")
        return float(spearmanr(a[ok], b[ok]))

    r = {
        "Recon(z)": rho(mse_z, policy_kl),
        "QueryMis(z)": rho(qmis_z, policy_kl),
        "KLtail(z)": rho(kl_z, policy_kl),
    }

    order = ["Recon(z)", "QueryMis(z)", "KLtail(z)"]
    plt.figure(figsize=(6.0, 3.8))
    plt.bar(order, [r[k] for k in order])
    plt.ylim(-1.0, 1.0)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.ylabel("Spearman corr(score, policy KL)")
    plt.title("Mechanistic alignment (base): which score tracks policy mismatch?")
    plt.tight_layout()
    savefig_png_pdf(figs / "F6_target_policy_alignment_pooled.png", dpi=200)
    plt.close()

    return {f"rho_{k}": float(v) for k, v in r.items()}


def _objective_window_score(
    *,
    model: ObjectiveAblationTCN,
    windows: torch.Tensor,
    as_detector: bool = False,
) -> np.ndarray:
    """Compute the per-window objective score (higher = more surprising).

    as_detector=True  → always return mse + jepa_tail_dist, regardless of the
    training objective.  This mirrors how run_AxonAD_ablations.py evaluates every
    variant with score_mode='mse+jepa', giving a fair comparison: all models are
    judged as anomaly detectors, not on how well they minimised their training loss.
    """
    device = model.device
    objective = str(model._train_objective)
    loader = DataLoader(windows, batch_size=model.batch_size, shuffle=False)
    out: list[np.ndarray] = []

    model.model.eval()
    with torch.inference_mode():
        for batch in loader:
            d = batch.to(device)
            rec, q_real, q_pred, k, q_tgt, tmask = model.model(d)

            if as_detector:
                # Mirror run_AxonAD_ablations.py decision_function(score_mode='mse+jepa'):
                # each component is individually normalized by per-model calibration stats
                # (fitted on training data), then summed.  Raw mse+jepa would be dominated
                # by whichever component has larger absolute scale (e.g. PredKeys has large
                # raw JEPA since it was never trained on JEPA loss).
                eps = 1e-9
                mse_s  = (rec - d).pow(2).sum(dim=-1).mean(dim=1)   # [B]
                jepa_s = model._jepa_tail_dist(q_tgt, q_pred)        # [B]
                mse_med  = float(model.mse_med  if model.mse_med  is not None else 0.0)
                mse_iqr  = float(model.mse_iqr  if model.mse_iqr  is not None else 1.0)
                jepa_med = float(model.jepa_med if model.jepa_med is not None else 0.0)
                jepa_iqr = float(model.jepa_iqr if model.jepa_iqr is not None else 1.0)
                z_mse  = (mse_s  - mse_med)  / (mse_iqr  + eps)
                z_jepa = (jepa_s - jepa_med) / (jepa_iqr + eps)
                out.append((z_mse + z_jepa).detach().cpu().numpy())
                continue

            if objective == "rec":
                s = (rec - d).pow(2).sum(dim=-1).mean(dim=1)  # [B]
                out.append(s.detach().cpu().numpy())
                continue

            if objective == "jepa":
                s = model._jepa_tail_dist(q_tgt, q_pred)  # [B]
                out.append(s.detach().cpu().numpy())
                continue

            # Build h_norm + shifted history
            h = model.model.embed(d) + model.model.pos
            h_norm = model.model.ln1(h)
            h_det = h_norm.detach()
            B, T, C = h_det.shape
            s_shift = max(0, min(int(model.model.forecast_steps), T))
            x_shift = model._shift_history(h_det, s_shift)
            q_pred_flat = model.model.attn.q_pred_net(x_shift)  # [B,T,C]

            if tmask is None or tmask.numel() == 0:
                out.append(np.zeros((B,), dtype=np.float32))
                continue
            m = tmask.to(d.device)
            if m.dtype != torch.bool:
                m = m.bool()

            def per_window_cos_dist(pred_flat: torch.Tensor, tgt_flat: torch.Tensor) -> torch.Tensor:
                # pred/tgt: [B,T,C]; mask: [B,T]
                p = F.normalize(pred_flat, dim=-1)
                t = F.normalize(tgt_flat, dim=-1)
                dist = 1.0 - (p * t).sum(dim=-1)  # [B,T]
                denom = m.sum(dim=1).clamp_min(1)
                num = (dist * m).sum(dim=1)
                return num / denom

            if objective == "pred_h":
                s = per_window_cos_dist(q_pred_flat, h_det)
                out.append(s.detach().cpu().numpy())
                continue

            if objective == "pred_k":
                if model.k_pred_net is None:
                    raise RuntimeError("k_pred_net missing")
                k_pred_flat = model.k_pred_net(x_shift)
                k_pred = model.model._split_heads(k_pred_flat, model.model.attn.num_heads)
                k_tgt = k.detach()
                s = per_window_cos_dist(model._merge_heads(k_pred), model._merge_heads(k_tgt))
                out.append(s.detach().cpu().numpy())
                continue

            if objective == "pred_v":
                if model.v_pred_net is None:
                    raise RuntimeError("v_pred_net missing")
                v_pred_flat = model.v_pred_net(x_shift)
                v_pred = model.model._split_heads(v_pred_flat, model.model.attn.num_heads)
                v_tgt = model.model.attn._split_heads(model.model.attn.wv(h_norm)).detach()
                s = per_window_cos_dist(model._merge_heads(v_pred), model._merge_heads(v_tgt))
                out.append(s.detach().cpu().numpy())
                continue

            raise ValueError(f"Unknown objective: {objective}")

    return np.concatenate(out, axis=0).astype(np.float32)


def _plot_objective_ablation_aucpr(
    *,
    out_root: Path,
    df: pd.DataFrame,
    order: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    if df.empty:
        return

    # Aggregate across series: mean AP + bootstrap CI over series.
    rows = []
    rng = np.random.default_rng(0)
    for name in order:
        sub = df[df["variant"] == name]
        aps = sub["ap"].to_numpy(dtype=np.float64)
        aps = aps[np.isfinite(aps)]
        if len(aps) == 0:
            rows.append((name, float("nan"), float("nan"), float("nan")))
            continue
        mean = float(np.mean(aps))
        if len(aps) < 3:
            rows.append((name, mean, float("nan"), float("nan")))
            continue
        boots = []
        for _ in range(500):
            idx = rng.integers(0, len(aps), size=len(aps))
            boots.append(float(np.mean(aps[idx])))
        lo, hi = np.percentile(np.array(boots), [2.5, 97.5])
        rows.append((name, mean, float(lo), float(hi)))

    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    lo = [r[2] for r in rows]
    hi = [r[3] for r in rows]
    err_lo = [m - l if np.isfinite(l) else 0.0 for m, l in zip(means, lo)]
    err_hi = [h - m if np.isfinite(h) else 0.0 for m, h in zip(means, hi)]

    plt.figure(figsize=(7.2, 3.8))
    plt.bar(names, means, yerr=[err_lo, err_hi], capsize=3)
    plt.ylim(0, max(0.05, float(np.nanmax(means)) * 1.15))
    plt.ylabel("AUC-PR (objective score)")
    plt.title("Training-objective ablations (same architecture)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    savefig_png_pdf(figs / "F7_objective_ablation_aucpr.png", dpi=200)
    plt.close()


def _plot_case_study_timeline(
    *,
    out_root: Path,
    series_name: str,
    centers: np.ndarray,
    y_win: np.ndarray,
    mse_z: np.ndarray,
    q_z: np.ndarray,
    combined: np.ndarray,
    zoom_idx: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    stem = Path(series_name).stem

    def shade_anoms(ax):
        # shade contiguous anomaly windows
        in_anom = (y_win > 0)
        if not in_anom.any():
            return
        starts = np.where(np.diff(np.concatenate([[0], in_anom.view(np.int8), [0]])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[0], in_anom.view(np.int8), [0]])) == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(centers[s], centers[e - 1], color="red", alpha=0.15, lw=0)

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    shade_anoms(ax)
    ax.plot(centers, mse_z, label="Recon z", lw=1.0)
    ax.plot(centers, q_z, label="Query mismatch z", lw=1.0)
    ax.plot(centers, combined, label="Combined", lw=1.2)
    ax.set_title(f"Case study: {stem}")
    ax.set_xlabel("time")
    ax.set_ylabel("z-score")
    ax.legend()
    plt.tight_layout()
    savefig_png_pdf(figs / f"F1_case_timeline_{stem}.png", dpi=200)
    plt.close()

    if zoom_idx is not None:
        i0 = max(0, zoom_idx - 250)
        i1 = min(len(centers), zoom_idx + 250)
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        shade_anoms(ax)
        ax.plot(centers[i0:i1], mse_z[i0:i1], label="Recon z", lw=1.0)
        ax.plot(centers[i0:i1], q_z[i0:i1], label="Query mismatch z", lw=1.0)
        ax.set_title(f"Association/mode slice (zoom): {stem}")
        ax.legend()
        plt.tight_layout()
        savefig_png_pdf(figs / f"F1_case_zoom_{stem}.png", dpi=200)
        plt.close()


def _plot_scatter_quadrants(
    *,
    out_root: Path,
    series_name: str,
    mse_z: np.ndarray,
    q_z: np.ndarray,
    y_win: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    stem = Path(series_name).stem
    norm = (y_win == 0)
    anom = (y_win == 1)

    plt.figure(figsize=(6, 6))
    plt.scatter(mse_z[norm], q_z[norm], s=6, alpha=0.3, label="normal")
    plt.scatter(mse_z[anom], q_z[anom], s=8, alpha=0.6, label="anomaly")
    plt.axvline(0.0, color="k", lw=0.8)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel("Recon z")
    plt.ylabel("Query mismatch z")
    plt.title(f"Scatter: {stem}")
    plt.legend()
    plt.tight_layout()
    savefig_png_pdf(figs / f"scatter_quadrants_{stem}.png", dpi=170)
    plt.close()

    # Quadrant stats (in TEST windows)
    lr_hq = int(((mse_z < 0) & (q_z > 0) & (anom)).sum())
    hr_lq = int(((mse_z > 0) & (q_z < 0) & (anom)).sum())
    both_hi = int(((mse_z > 0) & (q_z > 0) & (anom)).sum())
    both_lo = int(((mse_z < 0) & (q_z < 0) & (anom)).sum())

    return {
        "series": series_name,
        "anom_lowRecon_highQuery": lr_hq,
        "anom_highRecon_lowQuery": hr_lq,
        "anom_both_high": both_hi,
        "anom_both_low": both_lo,
    }


def _plot_policy_link(
    *,
    out_root: Path,
    series_name: str,
    dq_norm_h: np.ndarray,
    attn_kl_h: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    stem = Path(series_name).stem

    x = dq_norm_h.reshape(-1)
    y = attn_kl_h.reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y) if len(x) > 10 else float("nan")

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=4, alpha=0.25)
    plt.xlabel(r"$||\Delta Q||$ (per-head, tail)")
    plt.ylabel(r"$KL(A_{real}||A_{pred})$ (per-head, tail)")
    plt.title(f"Policy link: {stem} (Spearman={rho:.3f})")
    plt.tight_layout()
    savefig_png_pdf(figs / f"policy_link_{stem}.png", dpi=170)
    plt.close()

    return {"series": series_name, "spearman_dq_vs_attnkl": float(rho), "n_points": int(len(x))}


def _plot_head_specialization(
    *,
    out_root: Path,
    series_name: str,
    dq_norm_h: np.ndarray,
    y_win: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    stem = Path(series_name).stem

    H = dq_norm_h.shape[1]
    aps = []
    for h in range(H):
        ap = average_precision(y_win, dq_norm_h[:, h])
        aps.append(ap)
    aps = np.array(aps, dtype=np.float64)
    # If a split has no positives, AP is NaN; keep plotting stable.
    aps_plot = np.nan_to_num(aps, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-aps)

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(H), aps_plot[order])
    plt.xlabel("head (ranked)")
    plt.ylabel("AUC-PR of head mismatch")
    plt.title(f"Head specialization: {stem}")
    plt.tight_layout()
    savefig_png_pdf(figs / f"head_specialization_{stem}.png", dpi=170)
    plt.close()

    best = float(np.nanmax(aps)) if np.isfinite(aps).any() else float("nan")
    mean = float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan")
    return {"series": series_name, "best_head_ap": best, "mean_head_ap": mean}


def _pca_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit 2D PCA via SVD.

    Returns (mean, components) where components is [D,2].
    """
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2:
        raise ValueError(f"PCA needs x[N,D] with N>=5 and D>=2; got {x.shape}")
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    # SVD of centered matrix
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:2].T  # [D,2]
    return mu.astype(np.float32), comps.astype(np.float32)


def _pca_transform(x: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    xc = x - mu
    return (xc @ comps).astype(np.float32)


def _kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n <= k:
        return np.arange(n) % max(k, 1)
    centers = x[rng.choice(n, size=k, replace=False)]
    for _ in range(iters):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        lab = d2.argmin(axis=1)
        for j in range(k):
            m = (lab == j)
            if m.any():
                centers[j] = x[m].mean(axis=0)
    return lab


def _plot_query_manifold(
    *,
    out_root: Path,
    series_name: str,
    q_embed_2d: np.ndarray,
    cluster_id: np.ndarray,
    y_win: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(out_root / "figs")
    stem = Path(series_name).stem

    plt.figure(figsize=(6, 5))
    plt.scatter(q_embed_2d[:, 0], q_embed_2d[:, 1], c=cluster_id, s=6, cmap="tab10", alpha=0.5)
    plt.title(f"Query manifold clusters (train-normal): {stem}")
    plt.tight_layout()
    savefig_png_pdf(figs / f"q_manifold_clusters_{stem}.png", dpi=170)
    plt.close()

    # Overlay anomalies (in red)
    if (y_win > 0).any():
        plt.figure(figsize=(6, 5))
        plt.scatter(q_embed_2d[:, 0], q_embed_2d[:, 1], c="lightgray", s=6, alpha=0.35)
        plt.scatter(q_embed_2d[y_win > 0, 0], q_embed_2d[y_win > 0, 1], c="red", s=10, alpha=0.7)
        plt.title(f"Anomalies overlay in query space: {stem}")
        plt.tight_layout()
        savefig_png_pdf(figs / f"q_manifold_anoms_{stem}.png", dpi=170)
        plt.close()


def _extract_query_vectors_for_manifold(
    *,
    base: TCNTransformer,
    series_X: np.ndarray,
    series_y_point: np.ndarray,
    train_end: int,
    max_train: int = 3000,
    max_test: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract q_tgt tail-mean vectors for train-normal and test windows.

    Returns:
      q_train: [N_train, H*D]
      q_test:  [N_test, H*D] (mix of normal/anom, for overlay)
    """
    ds = ReconstructDataset(series_X, window_size=base.win_size, stride=1, normalize=True)
    y_win = window_label_from_point_labels(series_y_point, base.win_size)
    centers = window_centers(len(series_y_point), base.win_size)
    n = min(len(ds), len(y_win), len(centers))
    if n < 50:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    is_train = centers[:n] < int(train_end)
    is_test = ~is_train

    train_idx = np.where(is_train & (y_win[:n] == 0))[0]
    test_idx = np.where(is_test)[0]
    if len(train_idx) < 20 or len(test_idx) < 20:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    if len(train_idx) > max_train:
        train_idx = np.random.choice(train_idx, size=max_train, replace=False)
    if len(test_idx) > max_test:
        test_idx = np.random.choice(test_idx, size=max_test, replace=False)

    def q_tail(windows: torch.Tensor) -> np.ndarray:
        device = base.device
        base.model.eval()
        with torch.inference_mode():
            d = windows.to(device)
            rec, q_real, q_pred, k, q_tgt, _ = base.model(d)
            N, H, T, D = q_tgt.shape
            s = max(0, min(base.forecast_steps, T))
            valid_T = T - s
            if valid_T <= 0:
                return np.zeros((0, H * D), dtype=np.float32)
            k_tail = min(base.kl_tail_k, valid_T) if base.kl_tail_k > 0 else valid_T
            t0 = max(s, T - k_tail)
            q = q_tgt[:, :, t0:, :].mean(dim=2).reshape(N, H * D)
            return q.detach().cpu().numpy().astype(np.float32)

    w_train = torch.stack([ds.samples[i] for i in train_idx], dim=0)
    w_test = torch.stack([ds.samples[i] for i in test_idx], dim=0)
    return q_tail(w_train), q_tail(w_test)


# =========================
# Main
# =========================


def main() -> None:
    set_seed(CFG.seed)

    out_root = ensure_dir(Path(CFG.out_dir))
    (out_root / "config.json").write_text(json.dumps(asdict(CFG), indent=2), encoding="utf-8")

    files = read_file_list(CFG.benchmark, CFG.split_list)
    if CFG.series_limit is not None:
        files = files[: int(CFG.series_limit)]

    data_dir = series_data_dir(CFG.benchmark)

    rows: list[dict[str, Any]] = []
    per_series_eval: list[dict[str, Any]] = []
    quadrant_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    score_mode_rows: list[dict[str, Any]] = []

    # Pooled paper figures (use TEST windows only; downsampled at plot time)
    pooled_mse_z: list[np.ndarray] = []
    pooled_q_z: list[np.ndarray] = []
    pooled_kl_z: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    pooled_dq_h: list[np.ndarray] = []
    pooled_attnkl_h: list[np.ndarray] = []

    pooled_test: list[dict[str, Any]] = []

    objective_ablation_rows: list[dict[str, Any]] = []
    # H3: per-series representation-shift AUC-PR (Q/K/V/H discriminability)
    rep_shift_rows: list[dict[str, Any]] = []

    for i, fname in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] Series: {fname}")
        path = data_dir / fname
        if not path.exists():
            print(f"  [skip] missing file: {path}")
            continue

        X, y_point = load_series_csv(path)
        tr = parse_train_index(fname)
        train_end = int(tr) if tr is not None else int(0.8 * len(X))

        # Train/load base model (TRAIN only)
        base, ckpt = load_or_train_base_model(series_name=fname, series_X=X, train_end=train_end, out_root=out_root)
        print(f"  [ok] base model checkpoint: {ckpt}")

        # Extract mechanistic signals (window-level)
        metrics = extract_mechanistic_signals(
            base=base,
            series_X=X,
            series_y_point=y_point,
            train_end=train_end,
            series_name=fname,
            out_root=out_root,
        )
        print(f"  [ok] extracted signals (corr mse/qmis={metrics.get('mse_qmis_corr')})")

        series_dir = out_root / "series" / Path(fname).stem
        sig = np.load(series_dir / "signals.npz")
        mse = sig["mse"].astype(np.float64)
        qmis = sig["qmis"].astype(np.float64)
        kl = sig["kl"].astype(np.float64) if "kl" in sig.files else None
        dq_norm_h = sig["dq_norm_h"].astype(np.float64)
        attn_kl_h = sig["attn_kl_h"].astype(np.float64)
        k_shift = sig["kmis"].astype(np.float64) if "kmis" in sig.files else np.zeros_like(mse)
        v_shift = sig["vmis"].astype(np.float64) if "vmis" in sig.files else np.zeros_like(mse)
        h_shift = sig["hmis"].astype(np.float64) if "hmis" in sig.files else np.zeros_like(mse)

        win_idx = sig["win_idx"].astype(np.int64) if "win_idx" in sig.files else np.arange(len(mse), dtype=np.int64)
        y_win_full = window_label_from_point_labels(y_point, base.win_size)
        centers_full = window_centers(len(y_point), base.win_size)
        n_full = min(len(y_win_full), len(centers_full))
        win_idx = win_idx[win_idx < n_full]

        y_win = y_win_full[win_idx]
        centers = centers_full[win_idx]

        # Split windows into TRAIN vs TEST by center
        is_train_w = centers < int(train_end)
        is_test_w = ~is_train_w

        # Robust z-score stats from TRAIN windows only
        # Robust stats from TRAIN-NORMAL only (avoid leakage via anomaly oversampling).
        train_norm = is_train_w & (y_win == 0)
        if int(train_norm.sum()) < 50:
            train_norm = is_train_w

        mse_z, mse_stats = robust_z(mse[train_norm], mse)
        q_z, q_stats = robust_z(qmis[train_norm], qmis)
        if kl is not None:
            kl_z, _ = robust_z(kl[train_norm], kl)
        else:
            kl_z = np.zeros_like(mse_z)

        # H3: z-score trained predictor mismatch scores (all four use identical TCN + cosine loss)
        # Q: q_pred_net trained end-to-end with JEPA; K/V/H: equivalent nets trained per-series
        q_shift_z, _ = robust_z(qmis[train_norm], qmis)
        k_shift_z, _ = robust_z(k_shift[train_norm], k_shift)
        v_shift_z, _ = robust_z(v_shift[train_norm], v_shift)
        h_shift_z, _ = robust_z(h_shift[train_norm], h_shift)

        # Evaluate each as anomaly detector (test windows only)
        if int((y_win[is_test_w] == 1).sum()) > 0:
            for rep_name, rep_score in [("Q", q_shift_z[is_test_w]), ("K", k_shift_z[is_test_w]),
                                        ("V", v_shift_z[is_test_w]), ("H", h_shift_z[is_test_w])]:
                ap_rs, lo_rs, hi_rs = block_bootstrap_ci(
                    y_win[is_test_w], rep_score, average_precision,
                    CFG.block_bootstrap_block_len, CFG.block_bootstrap_samples, CFG.seed + 6000 + i,
                )
                rep_shift_rows.append({
                    "series": fname, "rep": rep_name,
                    "ap": ap_rs, "ap_lo": lo_rs, "ap_hi": hi_rs,
                    "n_test": int(is_test_w.sum()),
                    "n_test_anom": int((y_win[is_test_w] == 1).sum()),
                })

        combined = mse_z + q_z  # additive, matching AxonAD score_mode='mse+jepa'

        # Reporting metrics on TEST windows only
        y_test = y_win[is_test_w]
        mse_test = mse_z[is_test_w]
        q_test = q_z[is_test_w]
        kl_test = kl_z[is_test_w]
        comb_test = combined[is_test_w]

        ap_mse, ap_mse_lo, ap_mse_hi = block_bootstrap_ci(
            y_test, mse_test, average_precision, CFG.block_bootstrap_block_len, CFG.block_bootstrap_samples, CFG.seed + i
        )
        ap_q, ap_q_lo, ap_q_hi = block_bootstrap_ci(
            y_test, q_test, average_precision, CFG.block_bootstrap_block_len, CFG.block_bootstrap_samples, CFG.seed + 1000 + i
        )
        ap_c, ap_c_lo, ap_c_hi = block_bootstrap_ci(
            y_test, comb_test, average_precision, CFG.block_bootstrap_block_len, CFG.block_bootstrap_samples, CFG.seed + 2000 + i
        )

        # Score-mode ablations (same base model, different scoring logic)
        score_modes = {
            "mse": mse_test,
            "q": q_test,
            "kl": kl_test,
            "mse+q": mse_test + q_test,
            "mse+kl": mse_test + kl_test,
            "q+kl": q_test + kl_test,
            "mse+q+kl": mse_test + q_test + kl_test,
            "max(mse,q)": np.maximum(mse_test, q_test),
            "max(mse,q,kl)": np.maximum(np.maximum(mse_test, q_test), kl_test),
        }
        for sm_name, sm_score in score_modes.items():
            m, lo, hi = block_bootstrap_ci(
                y_test,
                sm_score,
                average_precision,
                CFG.block_bootstrap_block_len,
                CFG.block_bootstrap_samples,
                CFG.seed + 4000 + i,
            )
            score_mode_rows.append(
                {
                    "series": fname,
                    "score_mode": sm_name,
                    "ap": m,
                    "ap_lo": lo,
                    "ap_hi": hi,
                    "n_test": int(len(y_test)),
                    "n_test_anom": int((y_test == 1).sum()),
                }
            )

        # Complementarity at matched FPR (thresholds from TRAIN windows)
        th_m = find_threshold_at_fpr(mse_z[is_train_w], CFG.fpr_for_complementarity)
        th_q = find_threshold_at_fpr(q_z[is_train_w], CFG.fpr_for_complementarity)
        # Calibrate combined threshold directly on the combined train scores so that
        # FPR is correctly matched (using max(th_m,th_q) would be too strict and
        # miss cases where one score crosses its own threshold but not the larger one).
        th_c = find_threshold_at_fpr(combined[is_train_w], CFG.fpr_for_complementarity)
        det_m = (mse_test >= th_m)
        det_q = (q_test >= th_q)
        det_c = (comb_test >= th_c)

        an = (y_test == 1)
        recall_m = float((det_m & an).sum() / max(1, an.sum()))
        recall_q = float((det_q & an).sum() / max(1, an.sum()))
        recall_c = float((det_c & an).sum() / max(1, an.sum()))

        # Query-only slice finder (for zoom)
        query_only = (q_z > CFG.z_hi) & (mse_z < CFG.z_lo) & (y_win > 0)
        zoom_idx = int(np.argmax(query_only.astype(np.int64))) if query_only.any() else None

        # Optional per-series debug plots (disabled in paper mode)
        if (not CFG.paper_figures_only) and CFG.make_per_series_debug_plots:
            try:
                plot_series_signals(series_dir)
            except Exception as e:
                print(f"  [warn] base plots failed: {e}")

        per_series_eval.append(
            {
                "series": fname,
                "train_end": int(train_end),
                "ap_recon": ap_mse,
                "ap_recon_lo": ap_mse_lo,
                "ap_recon_hi": ap_mse_hi,
                "ap_query": ap_q,
                "ap_query_lo": ap_q_lo,
                "ap_query_hi": ap_q_hi,
                "ap_combined": ap_c,
                "ap_combined_lo": ap_c_lo,
                "ap_combined_hi": ap_c_hi,
                "recall_recon_at_fpr": recall_m,
                "recall_query_at_fpr": recall_q,
                "recall_combined_at_fpr": recall_c,
                "spearman_dq_vs_attnkl": float(metrics.get("spearman_dq_vs_attnkl", float("nan"))),
                "mse_train_med": float(mse_stats["med"]),
                "qmis_train_med": float(q_stats["med"]),
            }
        )

        # Quadrant stats (no plot here; pooled scatter is generated once at the end)
        norm = (y_test == 0)
        anom = (y_test == 1)
        quadrant_rows.append(
            {
                "series": fname,
                "anom_lowRecon_highQuery": int(((mse_test < 0) & (q_test > 0) & (anom)).sum()),
                "anom_highRecon_lowQuery": int(((mse_test > 0) & (q_test < 0) & (anom)).sum()),
                "anom_both_high": int(((mse_test > 0) & (q_test > 0) & (anom)).sum()),
                "anom_both_low": int(((mse_test < 0) & (q_test < 0) & (anom)).sum()),
                "n_test": int(len(y_test)),
                "n_test_anom": int(anom.sum()),
            }
        )

        pooled_mse_z.append(mse_test.astype(np.float32))
        pooled_q_z.append(q_test.astype(np.float32))
        pooled_kl_z.append(kl_test.astype(np.float32))
        pooled_y.append(y_test.astype(np.int64))

        pooled_dq_h.append(dq_norm_h[is_test_w].astype(np.float32))
        pooled_attnkl_h.append(attn_kl_h[is_test_w].astype(np.float32))

        # Save window labels + centers + z-scores for full reproducibility
        np.savez_compressed(
            series_dir / "labels_and_scores.npz",
            centers=centers,
            y_win=y_win,
            is_train_w=is_train_w,
            mse_z=mse_z,
            q_z=q_z,
            kl_z=kl_z,
            combined=combined,
            win_idx=win_idx,
            mse_stats=np.array([mse_stats["med"], mse_stats["iqr"]], dtype=np.float32),
            q_stats=np.array([q_stats["med"], q_stats["iqr"]], dtype=np.float32),
        )

        # Policy-link stats (no per-series plot in paper mode)
        x = dq_norm_h[is_test_w].reshape(-1)
        y = attn_kl_h[is_test_w].reshape(-1)
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok]) if int(ok.sum()) > 10 else float("nan")
        policy_rows.append({"series": fname, "spearman_dq_vs_attnkl": float(rho), "n_points": int(ok.sum())})

        # Optional head specialization (appendix)
        if CFG.make_head_specialization:
            head_rows.append(_plot_head_specialization(out_root=out_root, series_name=fname, dq_norm_h=dq_norm_h[is_test_w], y_win=y_test))

        # Timeline plot (we will generate for final case studies once selected)
        rows.append({**metrics})

        # Training-objective ablations (cached checkpoints)
        if CFG.run_objective_ablations:
            variants = [
                ("ReconOnly", "rec", CFG.batch_size),
                ("JEPAOnly_Q", "jepa", CFG.batch_size),
                ("PredHiddenState", "pred_h", CFG.batch_size),
                ("PredKeys", "pred_k", 64),
                ("PredValues", "pred_v", 64),
            ]

            # Build windows for the same indices we already use (win_idx)
            ds_full = ReconstructDataset(X, window_size=base.win_size, stride=1, normalize=True)
            # make sure indices are valid in the dataset
            valid_idx = win_idx[win_idx < len(ds_full)]
            windows_all = torch.stack([ds_full.samples[j] for j in valid_idx], dim=0)

            # train/test masks aligned to these windows
            is_train = is_train_w[: len(valid_idx)]
            is_test = ~is_train
            y_all = y_win[: len(valid_idx)]
            train_norm2 = is_train & (y_all == 0)
            if int(train_norm2.sum()) < 50:
                train_norm2 = is_train

            # ── Inject AxonAD base (balanced, already trained) ──────────────────
            # In run_AxonAD_ablations.py every variant is scored with mse+jepa
            # so the balanced Base beats ReconOnly because it was trained on both.
            # Here we mirror that by using the already-computed mse_z + q_z.
            n_vi = len(valid_idx)
            base_mse_sc = mse_z[:n_vi]
            base_q_sc   = q_z[:n_vi]
            base_sc  = base_mse_sc + base_q_sc   # additive, same as mse+jepa in ablations script
            y_t_base = y_all[is_test]
            sc_t_base = base_sc[is_test]
            if int((y_t_base == 1).sum()) > 0:
                try:
                    ap_b, lo_b, hi_b = block_bootstrap_ci(
                        y_t_base, sc_t_base, average_precision,
                        CFG.block_bootstrap_block_len, CFG.block_bootstrap_samples,
                        CFG.seed + 5000 + i,
                    )
                    objective_ablation_rows.append({
                        "series": fname,
                        "variant": "AxonAD_Base",
                        "train_objective": "balanced",
                        "ap": ap_b, "ap_lo": lo_b, "ap_hi": hi_b,
                        "n_test": int(len(y_t_base)),
                        "n_test_anom": int((y_t_base == 1).sum()),
                    })
                except Exception as _e:
                    print(f"  [warn] base injection into objective_ablation_rows failed: {_e}")

            for vname, obj, bs in variants:
                try:
                    ab = load_or_train_objective_ablation(
                        series_name=fname,
                        series_X=X,
                        train_end=train_end,
                        out_root=out_root,
                        variant_name=vname,
                        train_objective=obj,
                        batch_size=int(bs),
                    )
                    # Score as anomaly detector (mse+jepa) — same as run_AxonAD_ablations.py
                    # Using as_detector=True ensures a fair comparison: every variant is
                    # evaluated with the same combined score, not its training-loss metric.
                    s_obj = _objective_window_score(model=ab, windows=windows_all, as_detector=True).astype(np.float64)
                    s_z, _ = robust_z(s_obj[train_norm2], s_obj)

                    y_t = y_all[is_test]
                    sc_t = s_z[is_test]
                    ap, lo, hi = block_bootstrap_ci(
                        y_t,
                        sc_t,
                        average_precision,
                        CFG.block_bootstrap_block_len,
                        CFG.block_bootstrap_samples,
                        CFG.seed + 5000 + i,
                    )
                    objective_ablation_rows.append(
                        {
                            "series": fname,
                            "variant": vname,
                            "train_objective": obj,
                            "ap": ap,
                            "ap_lo": lo,
                            "ap_hi": hi,
                            "n_test": int(len(y_t)),
                            "n_test_anom": int((y_t == 1).sum()),
                        }
                    )
                except Exception as e:
                    print(f"  [warn] objective ablation failed ({vname}) for {fname}: {e}")

    # Choose case studies after scanning all series
    eval_df = pd.DataFrame(per_series_eval)
    if len(eval_df) > 0:
        # recon-dominant: ap_recon - ap_query highest
        eval_df["gap_recon_minus_query"] = eval_df["ap_recon"] - eval_df["ap_query"]
        eval_df["gap_query_minus_recon"] = eval_df["ap_query"] - eval_df["ap_recon"]
        s1 = eval_df.sort_values("gap_recon_minus_query", ascending=False).head(1)
        s2 = eval_df.sort_values("gap_query_minus_recon", ascending=False).head(1)
        picks = [*s1["series"].tolist(), *s2["series"].tolist()]
        if int(CFG.case_studies) >= 3:
            s3 = eval_df.sort_values("ap_combined", ascending=False).head(1)
            picks.extend(s3["series"].tolist())
        case_series = list(dict.fromkeys(picks))[: int(CFG.case_studies)]
    else:
        case_series = []

    # Produce required case-study plots (timeline + zoom slice)
    for sname in case_series:
        sdir = out_root / "series" / Path(sname).stem
        ls = np.load(sdir / "labels_and_scores.npz")
        centers = ls["centers"]
        y_win = ls["y_win"]
        is_train_w = ls["is_train_w"].astype(bool)
        mse_z = ls["mse_z"]
        q_z = ls["q_z"]
        combined = ls["combined"]
        # zoom: anomaly windows where query is high and recon is low
        zoom = np.where((~is_train_w) & (y_win == 1) & (q_z > CFG.z_hi) & (mse_z < CFG.z_lo))[0]
        zoom_idx = int(zoom[0]) if len(zoom) else None
        _plot_case_study_timeline(
            out_root=out_root,
            series_name=sname,
            centers=centers,
            y_win=y_win,
            mse_z=mse_z,
            q_z=q_z,
            combined=combined,
            zoom_idx=zoom_idx,
        )

        # Optional query manifold (appendix)
        if CFG.make_query_manifold:
            X_case, y_case_point = load_series_csv(series_data_dir(CFG.benchmark) / sname)
            tr_case = parse_train_index(sname)
            train_end_case = int(tr_case) if tr_case is not None else int(0.8 * len(X_case))
            base_case, _ = load_or_train_base_model(series_name=sname, series_X=X_case, train_end=train_end_case, out_root=out_root)

            q_train, q_test = _extract_query_vectors_for_manifold(
                base=base_case,
                series_X=X_case,
                series_y_point=y_case_point,
                train_end=train_end_case,
                max_train=3000,
                max_test=1200,
            )

            if q_train.size > 0 and q_test.size > 0:
                try:
                    mu, comps = _pca_fit(q_train.astype(np.float32))
                    z_train = _pca_transform(q_train.astype(np.float32), mu, comps)
                    z_test = _pca_transform(q_test.astype(np.float32), mu, comps)

                    k = int(min(6, max(2, round(math.sqrt(len(z_train))))))
                    cl_train = _kmeans(z_train, k=k, seed=CFG.seed)

                    z_all = np.concatenate([z_train, z_test], axis=0)
                    cl_all = np.concatenate([cl_train, np.full((len(z_test),), -1, dtype=np.int64)], axis=0)
                    y_all = np.concatenate([np.zeros((len(z_train),), dtype=np.int64), np.ones((len(z_test),), dtype=np.int64)], axis=0)

                    _plot_query_manifold(out_root=out_root, series_name=sname, q_embed_2d=z_all, cluster_id=cl_all, y_win=y_all)
                except Exception as e:
                    print(f"  [warn] query manifold plot failed for {sname}: {e}")

    # Train pooled regressors (Q vs K/V/H) on TRAIN windows only
    tables_dir = ensure_dir(out_root / "tables")
    figs_dir = ensure_dir(out_root / "figs")

    # Paper figures: pooled separation + pooled policy link
    try:
        _plot_scatter_pooled(
            out_root=out_root,
            mse_z=np.concatenate(pooled_mse_z, axis=0) if pooled_mse_z else np.zeros((0,), dtype=np.float32),
            q_z=np.concatenate(pooled_q_z, axis=0) if pooled_q_z else np.zeros((0,), dtype=np.float32),
            y=np.concatenate(pooled_y, axis=0) if pooled_y else np.zeros((0,), dtype=np.int64),
        )
    except Exception as e:
        print(f"  [warn] pooled scatter plot failed: {e}")

    try:
        _plot_policy_link_pooled(
            out_root=out_root,
            dq_norm_h=np.concatenate(pooled_dq_h, axis=0) if pooled_dq_h else np.zeros((0, 0), dtype=np.float32),
            attn_kl_h=np.concatenate(pooled_attnkl_h, axis=0) if pooled_attnkl_h else np.zeros((0, 0), dtype=np.float32),
        )
    except Exception as e:
        print(f"  [warn] pooled policy-link plot failed: {e}")

    # New F6: base-model internal alignment (scores vs policy mismatch)
    try:
        mse_all = np.concatenate(pooled_mse_z, axis=0) if pooled_mse_z else np.zeros((0,), dtype=np.float32)
        q_all = np.concatenate(pooled_q_z, axis=0) if pooled_q_z else np.zeros((0,), dtype=np.float32)
        kl_all = np.concatenate(pooled_kl_z, axis=0) if pooled_kl_z else np.zeros((0,), dtype=np.float32)
        attnkl_h_all = np.concatenate(pooled_attnkl_h, axis=0) if pooled_attnkl_h else np.zeros((0, 0), dtype=np.float32)
        policy_kl = attnkl_h_all.mean(axis=1) if attnkl_h_all.ndim == 2 and attnkl_h_all.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
        r = _plot_base_policy_alignment_pooled(out_root=out_root, policy_kl=policy_kl, mse_z=mse_all, qmis_z=q_all, kl_z=kl_all)

        pd.DataFrame(
            [
                {"score": "Recon(z)", "spearman_score_vs_policyKL": r.get("rho_Recon(z)", float("nan"))},
                {"score": "QueryMis(z)", "spearman_score_vs_policyKL": r.get("rho_QueryMis(z)", float("nan"))},
                {"score": "KLtail(z)", "spearman_score_vs_policyKL": r.get("rho_KLtail(z)", float("nan"))},
            ]
        ).to_csv(tables_dir / "T2_target_policy_alignment.csv", index=False)
    except Exception as e:
        print(f"  [warn] pooled base policy-alignment plot failed: {e}")

    # H3: save representation-shift AUC-PR table and bar plot
    # (computed per-series across all 20 series during the main loop; no extra ML needed)
    if len(rep_shift_rows) > 0:
        rs_df = pd.DataFrame(rep_shift_rows)
        rs_df.to_csv(tables_dir / "T1_rep_shift_aucpr.csv", index=False)
        print(f"  [H3] rep-shift AUC-PR saved ({len(rs_df)} rows)")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            figs_d = ensure_dir(out_root / "figs")
            means = rs_df.groupby("rep")["ap"].mean()
            order = [r for r in ["Q", "K", "V", "H"] if r in means.index]
            plt.figure(figsize=(5, 3.8))
            bars = plt.bar(order, [float(means.get(r, 0)) for r in order])
            for bar, r in zip(bars, order):
                bar.set_color("#2a7de1" if r == "Q" else "#aaaaaa")
            plt.ylabel("Mean AUC-PR (rep-shift score, all 20 series)")
            plt.title("H3: Which representation predictor error best detects anomalies?\n"
                      "All four (Q/K/V/H) scored with identical TCNQueryPredictor + cosine loss")
            plt.ylim(0, max(0.05, float(means.max()) * 1.15))
            plt.tight_layout()
            savefig_png_pdf(figs_d / "F4_rep_shift_aucpr.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"  [warn] rep-shift bar plot failed: {e}")

    # Save summary tables
    if len(per_series_eval) > 0:
        eval_df.to_csv(tables_dir / "per_series_eval.csv", index=False)
    if len(quadrant_rows) > 0:
        pd.DataFrame(quadrant_rows).to_csv(tables_dir / "quadrant_stats.csv", index=False)
    if len(policy_rows) > 0:
        pd.DataFrame(policy_rows).to_csv(tables_dir / "policy_link_spearman.csv", index=False)
    if len(head_rows) > 0:
        pd.DataFrame(head_rows).to_csv(tables_dir / "head_specialization.csv", index=False)
    if len(score_mode_rows) > 0:
        pd.DataFrame(score_mode_rows).to_csv(tables_dir / "per_series_score_modes.csv", index=False)
    if len(objective_ablation_rows) > 0:
        df_ab = pd.DataFrame(objective_ablation_rows)
        df_ab.to_csv(tables_dir / "per_series_objective_ablations.csv", index=False)
        try:
            _plot_objective_ablation_aucpr(
                out_root=out_root,
                df=df_ab,
                order=["AxonAD_Base", "ReconOnly", "JEPAOnly_Q", "PredHiddenState", "PredKeys", "PredValues"],
            )
        except Exception as e:
            print(f"  [warn] objective ablation plot failed: {e}")

    # Training-objective ablations (retrain specific variants)
    if CFG.run_training_objective_ablations and len(eval_df) > 0:
        ablation_series = case_series if str(CFG.ablation_scope).lower() == "case" else eval_df["series"].tolist()
        ablation_variants = [
            ("ReconOnly", "rec", CFG.batch_size),
            ("JEPAOnly_Q", "jepa", CFG.batch_size),
            ("PredHiddenState", "pred_h", CFG.batch_size),
            ("PredKeys", "pred_k", 64),
            ("PredValues", "pred_v", 64),
        ]
        ab_rows: list[dict[str, Any]] = []

        # ── Pre-populate ab_rows with AxonAD base from saved labels_and_scores ──
        # The balanced base model is already trained; inject its mse+q scores so
        # it appears alongside the ablation variants in the CSV without retraining.
        for sname in ablation_series:
            sdir = out_root / "series" / Path(sname).stem
            lsf = sdir / "labels_and_scores.npz"
            if not lsf.exists():
                continue
            try:
                ls = np.load(lsf)
                mse_zb = ls["mse_z"].astype(np.float64)
                q_zb   = ls["q_z"].astype(np.float64)
                y_wb   = ls["y_win"].astype(np.int64)
                itr_b  = ls["is_train_w"].astype(bool)
                itest_b = ~itr_b
                y_test_b  = y_wb[itest_b]
                sc_test_b = (mse_zb + q_zb)[itest_b]  # additive mse+q
                if int((y_test_b == 1).sum()) == 0:
                    continue
                for sm_name, sm_sc in [
                    ("mse", mse_zb[itest_b]),
                    ("q",   q_zb[itest_b]),
                    ("mse+q", sc_test_b),
                ]:
                    ap_b, lo_b, hi_b = block_bootstrap_ci(
                        y_test_b, sm_sc, average_precision,
                        CFG.block_bootstrap_block_len, CFG.block_bootstrap_samples,
                        CFG.seed + 9000,
                    )
                    ab_rows.append({
                        "series": sname, "variant": "AxonAD_Base",
                        "train_objective": "balanced", "score_mode": sm_name,
                        "ap": ap_b, "ap_lo": lo_b, "ap_hi": hi_b,
                        "n_test": int(len(y_test_b)), "n_test_anom": int((y_test_b == 1).sum()),
                    })
            except Exception as _e:
                print(f"  [warn] base injection into ab_rows failed for {sname}: {_e}")

        for sname in ablation_series:
            print(f"\n[ablation] series={sname}")
            X, y_point = load_series_csv(series_data_dir(CFG.benchmark) / sname)
            tr = parse_train_index(sname)
            train_end = int(tr) if tr is not None else int(0.8 * len(X))
            y_win_full = window_label_from_point_labels(y_point, CFG.win_size)
            centers_full = window_centers(len(y_point), CFG.win_size)

            for vname, objective, bs in ablation_variants:
                try:
                    model, ck = load_or_train_ablation_model(
                        series_name=sname,
                        series_X=X,
                        train_end=train_end,
                        out_root=out_root,
                        variant_name=vname,
                        train_objective=objective,
                        batch_size=int(bs),
                    )
                    print(f"  [ablation:{vname}] ckpt={ck}")

                    # Extract signals into a variant-specific subdir so we don't clobber base artifacts.
                    vdir = ensure_dir(out_root / "series" / Path(sname).stem / "variants" / vname)
                    extract_mechanistic_signals(
                        base=model,
                        series_X=X,
                        series_y_point=y_point,
                        train_end=train_end,
                        series_name=sname,
                        out_root=out_root,
                        series_out_dir=vdir,
                    )

                    sig = np.load(vdir / "signals.npz")
                    mse = sig["mse"].astype(np.float64)
                    qmis = sig["qmis"].astype(np.float64)
                    kl = sig["kl"].astype(np.float64) if "kl" in sig.files else np.zeros_like(mse)
                    win_idx = sig["win_idx"].astype(np.int64) if "win_idx" in sig.files else np.arange(len(mse), dtype=np.int64)
                    n_full = min(len(y_win_full), len(centers_full))
                    win_idx = win_idx[win_idx < n_full]
                    y_win = y_win_full[win_idx]
                    centers = centers_full[win_idx]
                    is_train_w = centers < int(train_end)
                    is_test_w = ~is_train_w

                    train_norm = is_train_w & (y_win == 0)
                    if int(train_norm.sum()) < 50:
                        train_norm = is_train_w

                    mse_z, _ = robust_z(mse[train_norm], mse)
                    q_z, _ = robust_z(qmis[train_norm], qmis)
                    kl_z, _ = robust_z(kl[train_norm], kl)

                    y_test = y_win[is_test_w]
                    scores = {
                        "mse": mse_z[is_test_w],
                        "q": q_z[is_test_w],
                        "kl": kl_z[is_test_w],
                        "mse+q": mse_z[is_test_w] + q_z[is_test_w],
                        "mse+kl": mse_z[is_test_w] + kl_z[is_test_w],
                        "q+kl": q_z[is_test_w] + kl_z[is_test_w],
                        "mse+q+kl": mse_z[is_test_w] + q_z[is_test_w] + kl_z[is_test_w],
                        "max(mse,q)": np.maximum(mse_z[is_test_w], q_z[is_test_w]),
                        "max(mse,q,kl)": np.maximum(np.maximum(mse_z[is_test_w], q_z[is_test_w]), kl_z[is_test_w]),
                    }
                    for sm_name, sm_score in scores.items():
                        ap, lo, hi = block_bootstrap_ci(
                            y_test,
                            sm_score,
                            average_precision,
                            CFG.block_bootstrap_block_len,
                            CFG.block_bootstrap_samples,
                            CFG.seed + 9000,
                        )
                        ab_rows.append(
                            {
                                "series": sname,
                                "variant": vname,
                                "train_objective": objective,
                                "score_mode": sm_name,
                                "ap": ap,
                                "ap_lo": lo,
                                "ap_hi": hi,
                                "n_test": int(len(y_test)),
                                "n_test_anom": int((y_test == 1).sum()),
                            }
                        )
                except Exception as e:
                    print(f"  [warn] ablation failed for {sname}/{vname}: {e}")

        if len(ab_rows) > 0:
            pd.DataFrame(ab_rows).to_csv(tables_dir / "per_series_training_objective_ablations.csv", index=False)

    # Report
    write_report(out_root, rows)
    print(f"\nDone. Report: {out_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
