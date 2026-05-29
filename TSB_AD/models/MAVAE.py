"""
This function is adapted from [MA-VAE]
Refactored to match the OmniAnomaly/BaseDetector structure.
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

class MAVAEModel(nn.Module):
    def __init__(self, 
                 feats, 
                 seq_len, 
                 latent_dim=64,
                 key_dim=64,
                 enc_h1=256, 
                 enc_h2=128, 
                 dec_h1=128, 
                 dec_h2=256,
                 n_heads=8,
                 noise_std=0.01):
        super(MAVAEModel, self).__init__()
        self.name = 'MAVAE'
        self.n_feats = feats
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.key_dim = key_dim  # per-head dimension (TF semantics)
        self.noise_std = noise_std
        
        # Calculate total attention dimension (TF: n_heads * key_dim)
        self.attn_dim = key_dim
        
        # --- Encoder (BiLSTM) ---
        self.enc_bilstm1 = nn.LSTM(feats, enc_h1, batch_first=True, bidirectional=True)
        self.enc_bilstm2 = nn.LSTM(enc_h1 * 2, enc_h2, batch_first=True, bidirectional=True)
        
        # Heads (Input = enc_h2 * 2 due to bidirectional)
        self.z_mean = nn.Linear(enc_h2 * 2, latent_dim)
        self.z_logvar = nn.Linear(enc_h2 * 2, latent_dim)
        
        # --- Multi-Head Attention (MA-VAE parameterisation, TF-aligned) ---
        # Q,K from X -> attention space (n_heads * key_dim per-head)
        self.q_proj = nn.Linear(feats, self.attn_dim)
        self.k_proj = nn.Linear(feats, self.attn_dim)
        
        # V from Z -> attention space (n_heads * key_dim per-head)
        self.v_proj = nn.Linear(latent_dim, self.attn_dim)
        
        # Attention runs in attention space (total width = n_heads * key_dim)
        # PyTorch embed_dim is total width; TF key_dim is per-head
        self.mha = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Map attention output back to latent space (matches TF output_shape=latent_dim)
        self.attn_to_latent = nn.Linear(self.attn_dim, latent_dim)

        # --- Decoder (BiLSTM) ---
        # Input is the output of Attention (latent_dim)
        self.dec_bilstm1 = nn.LSTM(latent_dim, dec_h1, batch_first=True, bidirectional=True)
        self.dec_bilstm2 = nn.LSTM(dec_h1 * 2, dec_h2, batch_first=True, bidirectional=True)
        
        # Heads (Input = dec_h2 * 2)
        self.x_mean = nn.Linear(dec_h2 * 2, feats)
        self.x_logvar = nn.Linear(dec_h2 * 2, feats)

    def reparameterize(self, mean, logvar, training):
        # Softplus for strictly positive variance
        std = torch.sqrt(F.softplus(logvar) + 1e-8)
        if training:
            eps = torch.randn_like(mean)
            return mean + std * eps, std
        else:
            return mean, std

    def forward(self, x, training=True):
        # x: [B, T, F]
        
        # 1. Input Noise
        if training and self.noise_std > 0:
            x_noisy = x + torch.randn_like(x) * self.noise_std
        else:
            x_noisy = x

        # 2. Encoder
        h, _ = self.enc_bilstm1(x_noisy)
        h, _ = self.enc_bilstm2(h)
        
        z_mu = self.z_mean(h)
        z_lv = self.z_logvar(h)
        z, z_std = self.reparameterize(z_mu, z_lv, training)
        
        # 3. Multi-Head Attention
        # Q, K come from clean Input X (TF behavior: noise only in encoder)
        # V comes from Latent Z (projected)
        # This allows the model to query the latent space based on input features
        q = self.q_proj(x)           # [B, T, attn_dim] - clean X
        k = self.k_proj(x)           # [B, T, attn_dim] - clean X
        v = self.v_proj(z)           # [B, T, attn_dim] - V from Z
        
        attn_out, _ = self.mha(query=q, key=k, value=v)   # [B, T, attn_dim]
        attn_lat = self.attn_to_latent(attn_out)          # [B, T, latent_dim]
        
        # 4. Decoder
        # Input is attention output
        h, _ = self.dec_bilstm1(attn_lat)
        h, _ = self.dec_bilstm2(h)
        
        x_mu = self.x_mean(h)
        x_lv = self.x_logvar(h)
        
        x_std = torch.sqrt(F.softplus(x_lv) + 1e-8)
        
        return x_mu, x_std, z_mu, z_std


class MAVAE(BaseDetector):
    def __init__(self,
                 win_size=32,
                 feats=1,
                 latent_dim=64,
                 key_dim=64,
                 batch_size=512,
                 epochs=100,
                 patience=15,
                 lr=1e-3,
                 beta_start=1e-8,
                 beta_end=1e-2,
                 annealing_epochs=25,
                 grace_period=25,
                 n_heads=8,
                 noise_std=0.01,
                 validation_size=0.2):
        super().__init__()
        
        self.cuda = True
        self.device = get_gpu(self.cuda)
        
        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.key_dim = key_dim
        self.validation_size = validation_size
        
        # Annealing Params
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.annealing_epochs = annealing_epochs
        self.grace_period = grace_period
        self.current_beta = beta_start

        # Dynamic Head Selection (if n_heads not valid, adjust)
        # Ensure attn_dim = n_heads * key_dim is valid
        # Check if key_dim allows requested n_heads
        h = n_heads
        while h > 1 and (h * key_dim) % h != 0:  # always true, so just check key_dim sensibility
            h -= 1
        final_heads = max(1, h)

        self.model = MAVAEModel(
            feats=feats, 
            seq_len=win_size, 
            latent_dim=latent_dim,
            key_dim=key_dim,
            n_heads=final_heads,
            noise_std=noise_std
        ).to(self.device)

        # Adam with amsgrad=True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)
        
        # Pre-calculate beta schedule (Cyclical)
        self.beta_values = np.linspace(beta_start, beta_end, annealing_epochs)

    def _update_beta(self, epoch):
        # Cyclical Annealing Logic
        # Grace period is 0-indexed in calculation
        grace_idx = max(0, self.grace_period - 1)
        shifted_epochs = max(0, epoch - 1 - grace_idx)
        
        if (epoch - 1) < grace_idx:
            # Linear warm-up during grace period? 
            # Original code: step_size * (epoch % grace)
            # We stick to source logic:
            step_size = (self.beta_start / self.grace_period)
            self.current_beta = step_size * ((epoch-1) % self.grace_period)
        else:
            # Cyclical
            cycle_pos = int(shifted_epochs % self.annealing_epochs)
            self.current_beta = self.beta_values[cycle_pos]

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
            self._update_beta(epoch)
            
            self.model.train()
            avg_loss = 0
            
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device)

                x_mu, x_std, z_mu, z_std = self.model(d, training=True)
                
                # 1. Reconstruction Loss (NLL)
                dist = Normal(x_mu, x_std)
                # Sum over features, Sum over time, Mean over batch (matching source logic)
                # Source: sum(dim=1).mean() -> sum time, mean batch. Features implicitly summed in evaluate_log_prob
                log_probs = dist.log_prob(d).sum(dim=1).sum(dim=-1).mean()
                nll_loss = -log_probs
                
                # 2. KL Divergence
                p_z = Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                q_z = Normal(z_mu, z_std)
                # Sum time & latent, Mean batch
                kl_loss = kl_divergence(q_z, p_z).sum(dim=(1, 2)).mean()
                
                loss = nll_loss + self.current_beta * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.item()
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), beta=self.current_beta)

            # Validation
            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                with torch.no_grad():
                    for idx, (d, _) in enumerate(valid_loader):
                        d = d.to(self.device)
                        x_mu, x_std, _, _ = self.model(d, training=False)
                        
                        dist = Normal(x_mu, x_std)
                        nll_loss = -dist.log_prob(d).sum(dim=(1,2)).mean()
                        avg_loss_val += nll_loss.item()
                
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
                
                x_mu, x_std, _, _ = self.model(d, training=False)
                
                # Anomaly Score = Negative Log Likelihood
                dist = Normal(x_mu, x_std)
                log_prob = dist.log_prob(d)
                
                # Sum over features (dim=-1), Mean over time (dim=1)
                # High NLL = Anomaly
                score_win = -log_prob.sum(dim=-1).mean(dim=1)
                
                scores.append(score_win.detach().to("cpu"))

        scores = torch.cat(scores, dim=0).numpy()
        self.__anomaly_score = scores
        
        # Padding
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