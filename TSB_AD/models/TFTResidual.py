"""
TFT-Residual++ (paper-faithful motifs + pipeline-friendly)

Adds the two requested upgrades:
1) Decoder horizon > 1 (pred_len): predict the last `pred_len` steps of each window
   from the first `win_size - pred_len` steps.
2) Quantile head + quantile (pinball) loss:
   predicts quantiles (e.g., [0.1, 0.5, 0.9]) for each predicted step and feature.
   For anomaly scoring, uses the median (0.5) prediction.

Pipeline compatibility:
- Uses ReconstructDataset windows (B, L, F)
- BaseDetector wrapper with fit() and decision_function()
- Produces 1 score per window; center pads to length T like your other detectors.
"""

from __future__ import division, print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import get_gpu


# ---------------------------
# Loss: Quantile (Pinball)
# ---------------------------

class QuantileLoss(nn.Module):
    """Pinball loss for multiple quantiles.
    y_true: (..., F)
    y_pred: (..., Q, F)  (quantiles on dim=-2)
    """
    def __init__(self, quantiles):
        super().__init__()
        qs = torch.tensor(quantiles, dtype=torch.float32)
        self.register_buffer("qs", qs, persistent=False)

    def forward(self, y_pred, y_true):
        # y_pred: (..., Q, F), y_true: (..., F)
        # broadcast y_true -> (..., Q, F)
        y_true_q = y_true.unsqueeze(-2)
        e = y_true_q - y_pred
        q = self.qs.view(*([1] * (e.ndim - 2)), -1, 1)  # (..., Q, 1)
        loss = torch.maximum(q * e, (q - 1.0) * e)
        return loss.mean()


# ---------------------------
# TFT building blocks
# ---------------------------

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.dropout(sublayer_out))


class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """
    Close to Google TFT GRN:
      hidden = Linear(x) + Linear(context, no bias) [optional]
      hidden = ELU
      hidden = Linear(hidden)
      gating = GLU(hidden)
      out = LayerNorm(skip(x) + Dropout(gating))
    """
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.0, context_dim=None):
        super().__init__()
        self.context_dim = context_dim
        self.d_out = d_in if d_out is None else d_out

        self.skip = nn.Identity() if d_in == self.d_out else nn.Linear(d_in, self.d_out)
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.ctx = nn.Linear(context_dim, d_hidden, bias=False) if context_dim is not None else None
        self.fc2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.glu = GLU(d_hidden, self.d_out)
        self.addnorm = AddNorm(self.d_out, dropout=dropout)

    def forward(self, x, context=None):
        h = self.fc1(x)
        if self.ctx is not None:
            if context is None:
                raise ValueError("GRN expected context but got None")
            h = h + self.ctx(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.addnorm(self.skip(x), h)


class VariableSelectionNetwork(nn.Module):
    """
    Temporal VSN over variables:
      var_emb: (B,T,N,D)
      context: (B,T,C) optional
    """
    def __init__(self, num_vars, d_model, d_hidden, dropout=0.0, context_dim=None):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model

        self.var_grns = nn.ModuleList([
            GRN(d_in=d_model, d_hidden=d_hidden, d_out=d_model, dropout=dropout, context_dim=None)
            for _ in range(num_vars)
        ])

        self.weight_grn = GRN(
            d_in=num_vars * d_model,
            d_hidden=d_hidden,
            d_out=num_vars,
            dropout=dropout,
            context_dim=context_dim
        )

    def forward(self, var_emb, context=None):
        B, T, N, D = var_emb.shape
        assert N == self.num_vars and D == self.d_model

        transformed = []
        for i in range(N):
            transformed.append(self.var_grns[i](var_emb[:, :, i, :]))  # (B,T,D)
        transformed = torch.stack(transformed, dim=2)                  # (B,T,N,D)

        flat = transformed.reshape(B, T, N * D)  
        logits = self.weight_grn(flat, context=context)                # (B,T,N)
        weights = torch.softmax(logits, dim=-1)

        selected = (weights.unsqueeze(-1) * transformed).sum(dim=2)     # (B,T,D)
        return selected, weights


class StaticEncoder(nn.Module):
    """
    Produces instance-specific static context from history.
    Input:  x_hist (B, T_enc, F)
    Output: s      (B, d_model)
    """
    def __init__(self, num_features, d_model, d_hidden, dropout=0.0):
        super().__init__()
        self.num_features = num_features
        self.proj = nn.Linear(num_features, d_model)
        self.grn = GRN(d_in=d_model, d_hidden=d_hidden, d_out=d_model, dropout=dropout)

    def forward(self, x_hist):
        # pool over time -> (B,F)
        pooled = x_hist.mean(dim=1)
        h = self.proj(pooled)          # (B, d_model)
        s = self.grn(h)                # (B, d_model)
        return s


def make_time_features(B, T_all, device, dtype, num_freqs=2):
    """
    Returns time features (B, T_all, F_time).
    Features:
      - normalized position t/T
      - sin/cos at a few harmonics
    """
    t = torch.linspace(0, 1, T_all, device=device, dtype=dtype).view(1, T_all, 1)  # (1,T,1)
    feats = [t]

    # simple harmonics: 1..num_freqs
    for k in range(1, num_freqs + 1):
        feats.append(torch.sin(2 * math.pi * k * t))
        feats.append(torch.cos(2 * math.pi * k * t))

    out = torch.cat(feats, dim=-1)                # (1,T,F_time)
    return out.expand(B, T_all, out.size(-1))     # (B,T,F_time)


def causal_mask(T, device):
    # True = disallow
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable attention (Google TFT-like):
    - separate Q/K per head
    - shared V projection across heads
    - average heads
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v_shared = nn.Linear(d_model, self.d_head, bias=False)

        self.w_o = nn.Linear(self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        B, T, D = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,d)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,d)

        v_shared = self.w_v_shared(x)                                           # (B,T,d)
        v = v_shared.unsqueeze(1).expand(B, self.n_heads, T, self.d_head)       # (B,H,T,d)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,T)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)     # (B,H,T,d)
        out = out.mean(dim=1)           # (B,T,d)
        out = self.w_o(out)             # (B,T,D)
        out = self.dropout(out)

        if return_attn:
            return out, attn
        return out


class TFTTemporalBlock(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.0):
        super().__init__()
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.attn_glu = GLU(d_model, d_model)
        self.attn_addnorm = AddNorm(d_model, dropout=dropout)
        self.ff_grn = GRN(d_in=d_model, d_hidden=d_hidden, d_out=d_model, dropout=dropout)

    def forward(self, x, mask=None):
        a = self.attn(x, mask=mask)
        a = self.attn_glu(a)
        x = self.attn_addnorm(x, a)
        x = self.ff_grn(x)
        return x


# ---------------------------
# TFT model with multi-step decoder + quantiles
# ---------------------------

class TFTResidualModel(nn.Module):
    def __init__(self,
                 num_features,
                 pred_len,
                 quantiles=(0.1, 0.5, 0.9),
                 d_model=64,
                 d_hidden=128,
                 n_heads=4,
                 num_attn_layers=1,
                 dropout=0.1,
                 # --- NEW (safe defaults) ---
                 use_time_covariates=True,
                 time_num_freqs=2):
        super().__init__()
        self.num_features = num_features
        self.pred_len = int(pred_len)
        if self.pred_len < 1:
            raise ValueError("pred_len must be >= 1")
        self.quantiles = list(quantiles)
        self.q = len(self.quantiles)
        self.d_model = d_model

        self.use_time_covariates = bool(use_time_covariates)
        self.time_num_freqs = int(time_num_freqs)

        # Per-variable embedding (continuous only): each variable -> Linear(1->d_model)
        self.var_embed = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])

        # --- Instance-specific static encoder instead of learnable parameter ---
        self.static_encoder = StaticEncoder(
            num_features=self.num_features,
            d_model=d_model,
            d_hidden=d_hidden,
            dropout=dropout
        )

        # --- Time covariates embeddings (each time feature gets its own embedding) ---
        # time features dim = 1 + 2*time_num_freqs
        self.time_dim = 1 + 2 * self.time_num_freqs
        if self.use_time_covariates:
            self.time_embed = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.time_dim)])

        # --- Separate VSNs: past observed vs known future (TFT-like split) ---
        self.vsn_past = VariableSelectionNetwork(
            num_vars=self.num_features,
            d_model=d_model,
            d_hidden=d_hidden,
            dropout=dropout,
            context_dim=d_model
        )

        self.vsn_future = VariableSelectionNetwork(
            num_vars=(self.time_dim if self.use_time_covariates else 1),
            d_model=d_model,
            d_hidden=d_hidden,
            dropout=dropout,
            context_dim=d_model
        )

        # --- Encoder/decoder LSTMs instead of one over concatenation ---
        self.lstm_enc = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

        self.lstm_glu = GLU(d_model, d_model)
        self.lstm_addnorm = AddNorm(d_model, dropout=dropout)

        # Static enrichment (now uses real per-window static context)
        self.enrich_grn = GRN(d_in=d_model, d_hidden=d_hidden, d_out=d_model, dropout=dropout, context_dim=d_model)

        self.blocks = nn.ModuleList([
            TFTTemporalBlock(d_model, d_hidden, n_heads, dropout=dropout)
            for _ in range(num_attn_layers)
        ])

        self.out_grn = GRN(d_in=d_model, d_hidden=d_hidden, d_out=d_model, dropout=dropout)
        self.out_glu = GLU(d_model, d_model)
        self.out_addnorm = AddNorm(d_model, dropout=dropout)

        # Quantile head: project last pred_len states to (pred_len, Q, F)
        self.proj_quantiles = nn.Linear(d_model, num_features * self.q)

    def _embed_vars(self, x):
        # x: (B,T,F) -> (B,T,N,D)
        B, T, F_ = x.shape
        assert F_ == self.num_features
        emb = []
        for i in range(self.num_features):
            ei = self.var_embed[i](x[:, :, i:i+1])  # (B,T,D)
            emb.append(ei)
        return torch.stack(emb, dim=2)              # (B,T,N,D)

    def _embed_time(self, tfeat):
        """
        tfeat: (B,T,F_time) -> (B,T,N_time,D) with N_time=F_time
        Each time feature is treated as a separate continuous variable.
        """
        B, T, Ft = tfeat.shape
        assert Ft == self.time_dim
        emb = []
        for i in range(Ft):
            ei = self.time_embed[i](tfeat[:, :, i:i+1])  # (B,T,D)
            emb.append(ei)
        return torch.stack(emb, dim=2)                   # (B,T,Ft,D)

    def forward(self, x_hist, decoder_len=None):
        """
        x_hist: (B, T_enc, F)
        decoder_len: int, default self.pred_len
        returns:
          yq_hat: (B, T_dec, Q, F)
        """
        if decoder_len is None:
            decoder_len = self.pred_len
        T_dec = int(decoder_len)
        if T_dec < 1:
            raise ValueError("decoder_len must be >= 1")

        B, T_enc, F_ = x_hist.shape
        device = x_hist.device
        dtype = x_hist.dtype

        # ---- 1) Instance-specific static context from history ----
        s = self.static_encoder(x_hist)                     # (B, d_model)
        ctx_enc = s[:, None, :].expand(B, T_enc, self.d_model)
        ctx_dec = s[:, None, :].expand(B, T_dec, self.d_model)

        # ---- 2) Past observed embeddings + VSN (past) ----
        var_emb_enc = self._embed_vars(x_hist)              # (B,T_enc,F,D)
        enc_selected, _ = self.vsn_past(var_emb_enc, context=ctx_enc)  # (B,T_enc,D)

        # ---- 3) Known future inputs (time covariates) + VSN (future) ----
        if self.use_time_covariates:
            # Build time features for FULL length, then slice future part
            tfeat_all = make_time_features(B, T_enc + T_dec, device, dtype, num_freqs=self.time_num_freqs)
            tfeat_dec = tfeat_all[:, -T_dec:, :]            # (B,T_dec,F_time)
            var_emb_dec = self._embed_time(tfeat_dec)       # (B,T_dec,N_time,D)
            dec_selected, _ = self.vsn_future(var_emb_dec, context=ctx_dec)  # (B,T_dec,D)
        else:
            # fallback: if you insist on no known-future, use zeros (less TFT-faithful)
            dec_selected = torch.zeros(B, T_dec, self.d_model, device=device, dtype=dtype)

        # ---- 4) LSTM encoder -> initialize decoder ----
        enc_out, (h, c) = self.lstm_enc(enc_selected)       # (B,T_enc,D)
        dec_out, _ = self.lstm_dec(dec_selected, (h, c))    # (B,T_dec,D)

        # ---- 5) Gate + residual on LSTM outputs (TFT gating motif) ----
        enc_temporal = self.lstm_addnorm(enc_selected, self.lstm_glu(enc_out))  # (B,T_enc,D)
        dec_temporal = self.lstm_addnorm(dec_selected, self.lstm_glu(dec_out))  # (B,T_dec,D)

        temporal = torch.cat([enc_temporal, dec_temporal], dim=1)               # (B,T_all,D)
        T_all = temporal.size(1)

        # ---- 6) Static enrichment (context) ----
        ctx_all = s[:, None, :].expand(B, T_all, self.d_model)
        enriched = self.enrich_grn(temporal, context=ctx_all)

        # ---- 7) Causal self-attention blocks over full sequence ----
        mask = causal_mask(T_all, device=enriched.device)
        x = enriched
        for blk in self.blocks:
            x = blk(x, mask=mask)

        # ---- 8) Output processing + skip ----
        dec = self.out_grn(x)
        dec = self.out_glu(dec)
        out = self.out_addnorm(dec, temporal)               # (B,T_all,D)

        # ---- 9) Take decoder positions, project quantiles ----
        out_dec = out[:, -T_dec:, :]                        # (B,T_dec,D)
        qflat = self.proj_quantiles(out_dec)                # (B,T_dec,F*Q)
        qflat = qflat.view(B, T_dec, self.q, self.num_features)  # (B,T_dec,Q,F)
        return qflat


# ---------------------------
# Detector wrapper
# ---------------------------

class TFTResidual(BaseDetector):
    def __init__(self,
                 win_size=100,
                 feats=1,
                 pred_len=10,
                 quantiles=(0.1, 0.5, 0.9),
                 d_model=64,
                 d_hidden=128,
                 n_heads=4,
                 num_attn_layers=1,
                 dropout=0.1,
                 batch_size=128,
                 epochs=30,
                 lr=1e-3,
                 validation_size=0.2,
                 loss_type="quantile",            # "quantile" or "mse"/"huber"
                 score_mode="residual_last",      # "residual_last" or "residual_all"
                 use_time_covariates=True,
                 time_num_freqs=2
                 ):
        super().__init__()

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = int(win_size)
        self.feats = int(feats)
        self.pred_len = int(pred_len)
        if self.pred_len < 1 or self.pred_len >= self.win_size:
            raise ValueError("pred_len must satisfy 1 <= pred_len < win_size")

        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        self.score_mode = score_mode
        self.quantiles = list(quantiles)

        self.model = TFTResidualModel(
            num_features=self.feats,
            pred_len=self.pred_len,
            quantiles=self.quantiles,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            num_attn_layers=num_attn_layers,
            dropout=dropout,
            use_time_covariates=use_time_covariates,
            time_num_freqs=time_num_freqs
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        lt = (loss_type or "quantile").lower()
        if lt == "quantile":
            self.criterion = QuantileLoss(self.quantiles).to(self.device)
            self._use_quantile_loss = True
        elif lt == "huber":
            self.criterion = nn.SmoothL1Loss().to(self.device)
            self._use_quantile_loss = False
        else:
            self.criterion = nn.MSELoss().to(self.device)
            self._use_quantile_loss = False

        # median quantile index for scoring
        if 0.5 in self.quantiles:
            self.q50_idx = self.quantiles.index(0.5)
        else:
            # fallback: closest to 0.5
            self.q50_idx = int(np.argmin([abs(q - 0.5) for q in self.quantiles]))

        self.__anomaly_score = None

    def fit(self, data):
        split = int((1 - self.validation_size) * len(data))
        tsTrain = data[:split]
        tsValid = data[split:]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        T_enc = self.win_size - self.pred_len

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device)  # (B,L,F)

                x_hist = d[:, :T_enc, :]                   # (B,T_enc,F)
                y_true = d[:, T_enc:, :]                   # (B,T_dec,F) where T_dec=pred_len

                yq_hat = self.model(x_hist, decoder_len=self.pred_len)  # (B,T_dec,Q,F)

                if self._use_quantile_loss:
                    loss = self.criterion(yq_hat, y_true)               # quantile loss
                else:
                    # compare against median head for regression losses
                    y_hat = yq_hat[:, :, self.q50_idx, :]               # (B,T_dec,F)
                    loss = self.criterion(y_hat, y_true)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=float(loss.item()))

            # Optional validation print (no early stopping by request)
            if len(valid_loader) > 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for d, _ in valid_loader:
                        d = d.to(self.device)
                        x_hist = d[:, :T_enc, :]
                        y_true = d[:, T_enc:, :]
                        yq_hat = self.model(x_hist, decoder_len=self.pred_len)
                        if self._use_quantile_loss:
                            val_loss += float(self.criterion(yq_hat, y_true).item())
                        else:
                            y_hat = yq_hat[:, :, self.q50_idx, :]
                            val_loss += float(self.criterion(y_hat, y_true).item())
                val_loss /= max(1, len(valid_loader))
                print(f"   val_loss={val_loss:.6f}")

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        scores = []

        T_enc = self.win_size - self.pred_len

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.inference_mode():
            for idx, (d, _) in loop:
                d = d.to(self.device)  # (B,L,F)

                x_hist = d[:, :T_enc, :]
                y_true = d[:, T_enc:, :]                         # (B,T_dec,F)
                yq_hat = self.model(x_hist, decoder_len=self.pred_len)
                y_med = yq_hat[:, :, self.q50_idx, :]             # (B,T_dec,F)

                residual = torch.abs(y_true - y_med)              # (B,T_dec,F)

                if self.score_mode == "residual_all":
                    # mean over time and features
                    score_win = residual.mean(dim=(1, 2))         # (B,)
                else:
                    # residual_last: only last predicted step
                    score_win = residual[:, -1, :].mean(dim=-1)   # (B,)

                scores.append(score_win.detach().cpu())

        scores = torch.cat(scores, dim=0).numpy()
        self.__anomaly_score = scores

        # Center padding to length T (your pipeline convention)
        if self.__anomaly_score.shape[0] < len(data):
            pad_l = math.ceil((self.win_size - 1) / 2)
            pad_r = (self.win_size - 1) // 2
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]] * pad_l
                + list(self.__anomaly_score)
                + [self.__anomaly_score[-1]] * pad_r
            )

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
