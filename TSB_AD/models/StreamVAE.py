"""
This function is adapted for [StreamVAE]
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

# --- Helper Modules ---

class MultiheadGQA(nn.Module):
    """Grouped Query Attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Simplified for 1:1 mapping in minimal ver
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        B, T, _ = query.shape
        # Projections and reshaping for multi-head
        q = self.w_q(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(out)

class LearnableSoftThreshold(nn.Module):
    def __init__(self, n_feats, init=1.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.expm1(torch.tensor(init))) * torch.ones(n_feats))

    def forward(self, x):
        tau = F.softplus(self.log_tau).view(1, 1, -1)
        return torch.sign(x) * F.relu(torch.abs(x) - tau)

class LearnableEMA(nn.Module):
    """
    Per-feature learnable EMA over the time dimension.
    y_t = alpha * y_{t-1} + (1 - alpha) * x_t,  with y_0 = x_0
    alpha is in (0,1), parameterized via a logit.
    """
    def __init__(self, n_feats, init_alpha=0.9):
        super().__init__()
        alpha = float(init_alpha)
        logit_alpha = math.log(alpha / (1.0 - alpha))
        self.logit_alpha = nn.Parameter(torch.full((n_feats,), logit_alpha))

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D] EMA along T
        """
        B, T, D = x.shape
        alpha = torch.sigmoid(self.logit_alpha).view(1, 1, D)  # [1,1,D]
        one_minus_alpha = 1.0 - alpha

        # Vectorized EMA computation (gradient-safe, no in-place ops)
        y_list = []
        y_prev = x[:, 0, :]  # [B, D]
        y_list.append(y_prev)
        
        for t in range(1, T):
            y_t = alpha.squeeze(1) * y_prev + one_minus_alpha.squeeze(1) * x[:, t, :]
            y_list.append(y_t)
            y_prev = y_t

        return torch.stack(y_list, dim=1)  # [B, T, D]

# --- Main Architecture ---

class StreamVAEModel(nn.Module):
    def __init__(self, feats, latent_dim, hidden_dim, device):
        super(StreamVAEModel, self).__init__()
        self.name = 'StreamVAE'
        self.device = device
        self.n_feats = feats
        self.n_hidden = hidden_dim
        self.n_latent = latent_dim
        self.beta = 1e-4 # Initial beta

        # --- Encoder ---
        self.enc_l1 = nn.LSTM(feats, self.n_hidden, 1, bidirectional=True, batch_first=True)
        self.enc_l2 = nn.LSTM(self.n_hidden*2, self.n_hidden//2, 1, bidirectional=True, batch_first=True)
        
        enc_out_dim = self.n_hidden # (hidden//2 * 2)
        
        # Gated Injection
        self.inj = nn.Linear(feats, self.n_latent, bias=False)
        self.inj_gate = nn.Parameter(torch.tensor(-2.0))
        
        self.to_mean = nn.Linear(enc_out_dim + self.n_latent, self.n_latent)
        self.to_logvar = nn.Linear(enc_out_dim + self.n_latent, self.n_latent)

        # --- Core/Attention ---
        self.proj_z = nn.Linear(self.n_latent, self.n_latent)
        self.proj_enc = nn.Linear(enc_out_dim, self.n_latent)
        
        # Learnable EMA baselines for drift/spike routing
        self.ema_z = LearnableEMA(self.n_latent, init_alpha=0.9)
        self.ema_enc = LearnableEMA(self.n_latent, init_alpha=0.9)
        
        # Dual Path Attention
        heads = max(2, self.n_latent // 8)
        self.attn_a = MultiheadGQA(self.n_latent, heads) # Drift
        self.attn_b = MultiheadGQA(self.n_latent, heads) # Spike
        
        self.attn_gain = nn.Parameter(torch.zeros(1))
        self.gate_merge = nn.Linear(2, 1)
        
        self.ln1 = nn.LayerNorm(self.n_latent)
        self.ffn = nn.Sequential(nn.Linear(self.n_latent, self.n_latent*3), nn.GELU(), nn.Linear(self.n_latent*3, self.n_latent))
        self.ln2 = nn.LayerNorm(self.n_latent)
        self.ffn_scale = nn.Parameter(torch.zeros(1))

        # --- Decoder ---
        self.dec_l1 = nn.LSTM(self.n_latent, self.n_hidden//2, 1, bidirectional=True, batch_first=True)
        self.dec_l2 = nn.LSTM(self.n_hidden, self.n_hidden, 1, bidirectional=True, batch_first=True)
        
        dec_out_dim = self.n_hidden * 2
        self.K = 4 # MoE experts
        
        self.dec_mean_k = nn.Linear(dec_out_dim, feats * self.K)
        self.dec_logvar = nn.Parameter(torch.full((feats,), -1.0))
        self.dec_gate = nn.Linear(self.n_latent, feats * self.K)
        
        # Residual Event Path
        self.event_head = nn.Linear(self.n_latent, feats, bias=False)
        self.event_shrink = LearnableSoftThreshold(feats)
        self.event_gate = nn.Parameter(torch.tensor(-3.0))
        self.ev_gamma = nn.Parameter(torch.zeros(feats))

    def _rms(self, x, eps=1e-8):
        return (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()

    def _sma(self, x, k=7):
        if k <= 1: return x
        xT = x.transpose(1, 2)
        pad = (k - 1) // 2
        xT = F.pad(xT, (pad, pad), mode='replicate')
        xT = F.avg_pool1d(xT, kernel_size=k, stride=1)
        return xT.transpose(1, 2)

    def _first_diff(self, x):
        d = x[:, 1:] - x[:, :-1]
        z0 = torch.zeros_like(x[:, :1])
        return torch.cat([z0, d], dim=1)

    def forward(self, x):
        bs, win, _ = x.shape
        
        # 1. Encode
        h, _ = self.enc_l1(x)
        h, _ = self.enc_l2(h)
        
        s = self.inj(x)
        alpha = torch.sigmoid(self.inj_gate)
        h_cat = torch.cat([h, alpha * s], dim=-1)
        
        mu = self.to_mean(h_cat)
        lv = self.to_logvar(h_cat)
        
        std = torch.sqrt(F.softplus(lv) + 1e-8)
        z = mu + std * torch.randn_like(mu) if self.training else mu
        
        # 2. Attention / Core
        z_proj = self.proj_z(z)
        enc_proj = self.proj_enc(h)
        
        # Path A (Drift / gradient) & Path B (Spike / deviations) using learnable EMAs
        enc_smooth = self.ema_enc(enc_proj)    # [B,T,D] smooth encoder baseline
        z_smooth = self.ema_z(z_proj)          # [B,T,D] smooth latent baseline

        # Drift path: first differences of smooth baseline
        enc_grad = self._first_diff(enc_smooth)
        Q_a, K_a = F.normalize(enc_grad, dim=-1), F.normalize(enc_grad, dim=-1)
        out_a = self.attn_a(Q_a, K_a, z_smooth)

        # Spike path: high-pass deviations from baseline
        enc_dev = enc_proj - enc_smooth
        z_dev = z_proj - z_smooth
        Q_b, K_b = F.normalize(enc_dev, dim=-1), F.normalize(enc_dev, dim=-1)
        out_b = self.attn_b(Q_b, K_b, z_dev)
        
        # Merge
        wa = out_a.pow(2).mean(dim=-1, keepdim=True)
        wb = out_b.pow(2).mean(dim=-1, keepdim=True)
        g = torch.sigmoid(self.gate_merge(torch.cat([wa, wb], dim=-1)))
        attn_out = self.attn_gain * (g * out_a + (1-g) * out_b)
        
        # Residual & FFN
        scale = self._rms(attn_out).detach() / (self._rms(z_proj).detach() + 1e-8)
        out1 = self.ln1(z_proj * scale + attn_out)
        out2 = self.ln2(out1 + self.ffn_scale * self.ffn(out1))
        
        # 3. Decode
        h_dec, _ = self.dec_l1(out2)
        h_dec, _ = self.dec_l2(h_dec)
        
        # MoE Reconstruction
        mu_k = self.dec_mean_k(h_dec).view(bs, win, self.n_feats, self.K)
        
        gate_logits = self.dec_gate(z_proj).view(bs, win, self.n_feats, self.K)
        gate_norm = (gate_logits - gate_logits.mean(-1, keepdim=True)) / (gate_logits.std(-1, keepdim=True) + 1e-5)
        gate_probs = F.softmax(gate_norm, dim=-1)
        
        mu_base = (mu_k * gate_probs).sum(dim=-1)
        
        # Event Residual
        z_spike = self._first_diff(z_proj)
        ev_raw = self.event_head(z_spike)
        ev_norm = ev_raw / (self._rms(ev_raw.detach()) + 1e-8)  # Normalize per feature
        ev_shrunk = self.event_shrink(ev_norm)

        # Rescale
        ev_final = ev_shrunk * self._rms(mu_base.detach()) * torch.exp(self.ev_gamma).view(1, 1, -1)

        # Final event residual in data space (what we will L1-penalize)
        ev_res = torch.sigmoid(self.event_gate) * ev_final  # [B,T,F]

        rec_mu = mu_base + ev_res
        rec_std = torch.sqrt(F.softplus(self.dec_logvar.view(1, 1, -1)) + 1e-4)

        return rec_mu, rec_std, mu, std, gate_probs, ev_res


class StreamVAE(BaseDetector):
    def __init__(self,
                 win_size=100,
                 feats=1,
                 latent_dim=64,
                 batch_size=128,
                 epochs=30,
                 patience=10,
                 lr=0.001,
                 validation_size=0.2,
                 target_kl=100.0,
                 event_l1_weight=1e-3):
        super().__init__()
        
        self.cuda = True
        self.device = get_gpu(self.cuda)
        
        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        self.target_kl = target_kl
        self.event_l1_weight = event_l1_weight

        self.model = StreamVAEModel(feats, latent_dim, 256, self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss(reduction='none') # Fallback, actual loss is Manual LogProb
        
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)
        
        # KL Control state
        self.kl_ema = None
        
    def _update_beta(self, current_kl):
        # Target KL Controller Logic
        if self.kl_ema is None:
            self.kl_ema = current_kl
        else:
            self.kl_ema = 0.95 * self.kl_ema + 0.05 * current_kl
            
        err = (self.kl_ema - self.target_kl) / (self.target_kl + 1e-8)
        new_beta = self.model.beta * np.exp(0.01 * err)
        self.model.beta = float(np.clip(new_beta, 1e-6, 10.0))

    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

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

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            avg_loss = 0
            
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device) # (B, Win, F)

                rec_mu, rec_std, z_mu, z_std, gates, ev_res = self.model(d)
                
                # 1. Reconstruction Loss (Log Prob)
                dist = Normal(rec_mu, rec_std)
                log_probs = dist.log_prob(d).sum(dim=-1).sum(dim=1).mean() # Sum F, Sum T, Mean B
                nll = -log_probs
                
                # 2. KL Divergence
                q = Normal(z_mu, z_std)
                p = Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                kl = kl_divergence(q, p).sum(dim=(1,2)).mean()
                
                # 3. Regularization (Entropy for MoE + L1 for sparse events)
                entropy = -(gates * (gates + 1e-8).log()).sum(-1).mean()
                reg_entropy = 0.001 * (math.log(4) * 0.5 - entropy).relu()

                # L1 penalty on event residual in data space
                l1_events = ev_res.abs().mean()  # [B,T,F] -> scalar

                loss = (
                    nll
                    + self.model.beta * kl
                    + reg_entropy
                    + self.event_l1_weight * l1_events
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                
                # Update Beta
                self._update_beta(kl.item())
                
                avg_loss += loss.item()
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), beta=self.model.beta, kl=kl.item())

            # Validation
            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                with torch.no_grad():
                    for idx, (d, _) in enumerate(valid_loader):
                        d = d.to(self.device)
                        rec_mu, rec_std, z_mu, z_std, gates, ev_res = self.model(d)

                        dist = Normal(rec_mu, rec_std)
                        nll = -dist.log_prob(d).sum(dim=(1, 2)).mean()
                        kl = kl_divergence(
                            Normal(z_mu, z_std),
                            Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                        ).sum(dim=(1, 2)).mean()

                        entropy = -(gates * (gates + 1e-8).log()).sum(-1).mean()
                        reg_entropy = 0.001 * (math.log(4) * 0.5 - entropy).relu()
                        l1_events = ev_res.abs().mean()

                        loss = (
                            nll
                            + self.model.beta * kl
                            + reg_entropy
                            + self.event_l1_weight * l1_events
                        )
                        avg_loss_val += loss.item()
                
                avg_loss = avg_loss_val / len(valid_loader)
            else:
                avg_loss = avg_loss / len(train_loader)
                
            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.inference_mode():
            for idx, (d, _) in loop:
                d = d.to(self.device)
                
                rec_mu, rec_std, z_mu, z_std, _, _ = self.model(d)
                
                # Anomaly Score = Negative Log Likelihood over the window
                dist = Normal(rec_mu, rec_std)
                # (B, T)
                log_prob = dist.log_prob(d).sum(dim=-1) 
                
                # Mean NLL per window
                score_win = -log_prob.mean(dim=1) 
                scores.append(score_win.detach().to("cpu"))

        scores = torch.cat(scores, dim=0).numpy()
        self.__anomaly_score = scores
        
        # Padding for window-based score mapping back to timestamp
        if self.__anomaly_score.shape[0] < len(data):
            pad_l = math.ceil((self.win_size-1)/2)
            pad_r = (self.win_size-1)//2
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]]*pad_l + 
                list(self.__anomaly_score) + 
                [self.__anomaly_score[-1]]*pad_r
            )
            
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score