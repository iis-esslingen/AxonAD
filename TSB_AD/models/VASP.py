"""
This function is adapted from [VASP]
Refactored to match the OmniAnomaly/BaseDetector structure.

VASP-style upgrade (minimal, leakage-safe):
- Train VAE (reconstruction + KL) as before
- Fit per-feature repair thresholds tau[f] from PAST-ONLY errors (t=0..T-2)
- Train an LSTM predictor on repaired PAST-ONLY windows to predict the LAST step
- decision_function scoring remains single-path reconstruction MSE
- Causal timestamp alignment: left-padding (score aligns to window end, no center padding)

IMPORTANT: Keeps your stability mapping:
  std = sqrt(softplus(logvar) + eps)
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import Normal, kl_divergence
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu


class VASPModel(nn.Module):
    def __init__(self,
                 feats,
                 seq_len,
                 latent_dim=8,
                 enc_hidden_1=80,
                 enc_hidden_2=65,
                 dec_hidden_1=65,
                 dec_hidden_2=80,
                 dropout=0.2):
        super(VASPModel, self).__init__()
        self.name = 'VASP'
        self.n_feats = feats
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.enc_fc1 = nn.Linear(feats, enc_hidden_1)

        # NOTE: In PyTorch, LSTM(dropout=...) only applies if num_layers > 1.
        self.enc_lstm = nn.LSTM(enc_hidden_1, enc_hidden_2, batch_first=True, dropout=dropout)
        self.enc_lstm_dropout = nn.Dropout(0.1)  # not true recurrent dropout, but kept as your intent

        # Dense after LSTM (Last timestep only)
        self.enc_fc2 = nn.Linear(enc_hidden_2, feats)

        # Latent heads
        self.z_mean = nn.Linear(feats, latent_dim)
        self.z_logvar = nn.Linear(feats, latent_dim)

        # --- Decoder ---
        self.dec_fc1 = nn.Linear(latent_dim, feats)
        self.dec_fc2 = nn.Linear(feats, dec_hidden_1)

        self.dec_lstm = nn.LSTM(dec_hidden_1, dec_hidden_1, batch_first=True, dropout=dropout)
        self.dec_lstm_dropout = nn.Dropout(0.1)

        self.dec_fc3 = nn.Linear(dec_hidden_1, dec_hidden_2)
        self.dec_out = nn.Linear(dec_hidden_2, feats)

    def reparameterize(self, mean, logvar, training):
        # KEEP: your stability mapping (strictly positive)
        std = torch.sqrt(F.softplus(logvar) + 1e-8)
        if training:
            eps = torch.randn_like(mean)
            return mean + std * eps, std
        else:
            return mean, std

    def forward(self, x, training=True):
        # x: [B, T, F]
        bs, seq, _ = x.shape

        # --- Encoder ---
        x_reshaped = x.reshape(-1, self.n_feats)
        h = self.enc_fc1(x_reshaped)
        h = h.reshape(bs, seq, -1)

        lstm_out, _ = self.enc_lstm(h)
        last_h = lstm_out[:, -1, :]

        if training:
            last_h = self.enc_lstm_dropout(last_h)

        h = self.enc_fc2(last_h)

        z_mu = self.z_mean(h)
        z_lv = self.z_logvar(h)
        z, z_std = self.reparameterize(z_mu, z_lv, training)

        # --- Decoder ---
        h = self.dec_fc1(z)
        h = self.dec_fc2(h)  # [B, dec_h1]

        # IMPORTANT: repeat to actual seq length (enables past-only calls with T-1)
        h_rep = h.unsqueeze(1).repeat(1, seq, 1)  # [B, seq, dec_h1]

        dec_lstm_out, _ = self.dec_lstm(h_rep)
        if training:
            dec_lstm_out = self.dec_lstm_dropout(dec_lstm_out)

        dec_flat = dec_lstm_out.contiguous().view(-1, dec_lstm_out.shape[-1])
        h = self.dec_fc3(dec_flat)
        x_hat_flat = self.dec_out(h)

        x_hat = x_hat_flat.reshape(bs, seq, self.n_feats)
        return x_hat, z_mu, z_lv, z_std


class VASPPredictor(nn.Module):
    """
    Causal predictor: repaired past -> predicts last step
      input : [B, T-1, F]
      output: [B, F]
    """
    def __init__(self, feats, hidden=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(feats, hidden, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden, feats)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1, :])


class VASP(BaseDetector):
    def __init__(self,
                 win_size=32,
                 feats=1,
                 latent_dim=8,
                 batch_size=512,
                 epochs=100,
                 patience=15,
                 lr=1e-3,
                 kl_weight=0.5,
                 validation_size=0.2):
        super().__init__()

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = int(win_size)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.feats = int(feats)
        self.validation_size = float(validation_size)
        self.kl_weight = float(kl_weight)

        # Internal constants (NOT exposed as hyperparameters; avoids changing your API)
        self._tau_quantile = 0.995
        self._tau_max_samples = 200000
        self._pred_hidden = 128

        self.model = VASPModel(
            feats=self.feats,
            seq_len=self.win_size,
            latent_dim=latent_dim
        ).to(self.device)

        self.predictor = VASPPredictor(
            feats=self.feats,
            hidden=self._pred_hidden
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.pred_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr, amsgrad=True)

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)
        self.pred_early_stopping = EarlyStoppingTorch(None, patience=patience)

        self.tau_ = None  # torch [F] on device
        self.__anomaly_score = None

    # ---------------------------
    # Threshold fitting (PAST-ONLY)
    # ---------------------------

    @torch.no_grad()
    def _fit_tau_past_only(self, train_loader):
        """
        Fit per-feature tau[f] from reconstruction error of past-only windows:
          d_past = d[:, :-1, :]
        This guarantees the target (last step) never influences tau.
        """
        self.model.eval()

        samples = []
        seen = 0

        for (d, _) in tqdm.tqdm(train_loader, desc="Fitting tau (past-only)", leave=True):
            d = d.to(self.device)
            if d.shape[1] < 2:
                continue

            d_past = d[:, :-1, :]  # [B, T-1, F]
            x_hat_past, _, _, _ = self.model(d_past, training=False)
            err_past = (d_past - x_hat_past) ** 2
            err_flat = err_past.reshape(-1, self.feats)  # [B*(T-1), F]

            err_np = err_flat.detach().to("cpu").numpy()
            n = err_np.shape[0]

            remaining = self._tau_max_samples - seen
            if remaining <= 0:
                break

            if n <= remaining:
                samples.append(err_np)
                seen += n
            else:
                idx = np.random.choice(n, size=remaining, replace=False)
                samples.append(err_np[idx])
                seen += remaining
                break

        if len(samples) == 0:
            tau = np.zeros((self.feats,), dtype=np.float32)
        else:
            E = np.concatenate(samples, axis=0)  # [N, F]
            tau = np.quantile(E, self._tau_quantile, axis=0).astype(np.float32)

        self.tau_ = torch.tensor(tau, device=self.device)  # [F]

    # ---------------------------
    # Repair (PAST-ONLY)
    # ---------------------------

    @torch.no_grad()
    def _repair_past_only(self, d):
        """
        d: [B, T, F]
        returns:
          d_past_rep: [B, T-1, F] repaired past
          d_target  : [B, F]      target last step (untouched)
        """
        d_past = d[:, :-1, :]
        d_target = d[:, -1, :]

        x_hat_past, _, _, _ = self.model(d_past, training=False)
        err_past = (d_past - x_hat_past) ** 2

        if self.tau_ is None:
            return d_past, d_target

        tau = self.tau_.view(1, 1, -1)
        mask_past = err_past > tau
        d_past_rep = torch.where(mask_past, x_hat_past, d_past)
        return d_past_rep, d_target

    # ---------------------------
    # Fit
    # ---------------------------

    def fit(self, data):
        data = np.asarray(data)
        if len(data) < self.win_size:
            self.tau_ = torch.zeros((self.feats,), device=self.device)
            return

        cut = int((1 - self.validation_size) * len(data))
        tsTrain = data[:cut]
        tsValid = data[cut:] if cut < len(data) else data[:0]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_loader = None
        if len(tsValid) >= self.win_size:
            valid_loader = DataLoader(
                dataset=ReconstructDataset(tsValid, window_size=self.win_size),
                batch_size=self.batch_size,
                shuffle=False
            )

        # ---- Stage 1: VAE ----
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            avg_loss = 0.0

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for _, (d, _) in loop:
                d = d.to(self.device)

                x_hat, z_mu, z_lv, z_std = self.model(d, training=True)

                # Reconstruction loss (MSE sum over time+features, mean over batch)
                rec_loss = ((d - x_hat) ** 2).sum(dim=(1, 2)).mean()

                # KL divergence to N(0, I)
                p = Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                q = Normal(z_mu, z_std)
                kl_loss = kl_divergence(q, p).sum(dim=1).mean()

                loss = rec_loss + self.kl_weight * kl_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                avg_loss += float(loss.item())
                loop.set_description(f"VAE Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=float(loss.item()), rec=float(rec_loss.item()), kl=float(kl_loss.item()))

            # Validation metric for early stopping
            if valid_loader is not None and len(valid_loader) > 0:
                self.model.eval()
                avg_val = 0.0
                with torch.no_grad():
                    for (d, _) in valid_loader:
                        d = d.to(self.device)
                        x_hat, z_mu, z_lv, z_std = self.model(d, training=False)
                        rec_loss = ((d - x_hat) ** 2).sum(dim=(1, 2)).mean()
                        kl_loss = kl_divergence(
                            Normal(z_mu, z_std),
                            Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                        ).sum(dim=1).mean()
                        avg_val += float((rec_loss + self.kl_weight * kl_loss).item())
                metric = avg_val / max(1, len(valid_loader))
            else:
                metric = avg_loss / max(1, len(train_loader))

            self.early_stopping(metric, self.model)
            if self.early_stopping.early_stop:
                print("   VAE Early stopping<<<")
                break

        # ---- Fit tau (past-only) ----
        self._fit_tau_past_only(train_loader)

        # ---- Stage 2: Predictor (causal; repaired past only; target never used in repair) ----
        for epoch in range(1, self.epochs + 1):
            self.predictor.train()
            self.model.eval()

            avg_loss = 0.0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for _, (d, _) in loop:
                d = d.to(self.device)
                if d.shape[1] < 2:
                    continue

                d_past_rep, d_target = self._repair_past_only(d)
                y_hat = self.predictor(d_past_rep)

                pred_loss = ((y_hat - d_target) ** 2).sum(dim=1).mean()

                self.pred_optimizer.zero_grad(set_to_none=True)
                pred_loss.backward()
                self.pred_optimizer.step()

                avg_loss += float(pred_loss.item())
                loop.set_description(f"PRED Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(pred=float(pred_loss.item()))

            # Predictor validation early stopping
            if valid_loader is not None and len(valid_loader) > 0:
                self.predictor.eval()
                avg_val = 0.0
                with torch.no_grad():
                    for (d, _) in valid_loader:
                        d = d.to(self.device)
                        if d.shape[1] < 2:
                            continue
                        d_past_rep, d_target = self._repair_past_only(d)
                        y_hat = self.predictor(d_past_rep)
                        pred_loss = ((y_hat - d_target) ** 2).sum(dim=1).mean()
                        avg_val += float(pred_loss.item())
                metric = avg_val / max(1, len(valid_loader))
            else:
                metric = avg_loss / max(1, len(train_loader))

            self.pred_early_stopping(metric, self.predictor)
            if self.pred_early_stopping.early_stop:
                print("   PRED Early stopping<<<")
                break

    # ---------------------------
    # Scoring (single path; no future leakage; causal alignment)
    # ---------------------------

    def decision_function(self, data):
        data = np.asarray(data)
        T = len(data)

        if T < self.win_size:
            self.__anomaly_score = np.zeros((T,), dtype=np.float32)
            return self.__anomaly_score

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        scores = []

        with torch.no_grad():
            loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
            for _, (d, _) in loop:
                d = d.to(self.device)
                x_hat, _, _, _ = self.model(d, training=False)

                # Single scoring path: reconstruction MSE
                err = (d - x_hat) ** 2
                score_win = err.sum(dim=-1).mean(dim=1)  # [B]

                scores.append(score_win.detach().to("cpu"))

        scores = torch.cat(scores, dim=0).numpy()  # length ~ T - win_size + 1
        self.__anomaly_score = scores

        # Causal alignment to the window end (no center padding; avoids future dependence)
        pad_left = self.win_size - 1
        if self.__anomaly_score.shape[0] < T:
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]] * pad_left + list(self.__anomaly_score),
                dtype=np.float32
            )
            self.__anomaly_score = self.__anomaly_score[:T]

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
