"""
This function is adapted from [VS-VAE]
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
from torch.distributions import Normal, Laplace, kl_divergence
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu

class VS_VAE_Model(nn.Module):
    def __init__(self, 
                 feats, 
                 seq_len, 
                 latent_dim=3, 
                 attn_vec_size=3,
                 enc_hidden=128, 
                 dec_hidden=64):
        super(VS_VAE_Model, self).__init__()
        self.name = 'VS_VAE'
        self.n_feats = feats
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.attn_vec_size = attn_vec_size
        
        # --- Encoder (BiLSTM) ---
        self.enc_bilstm = nn.LSTM(feats, enc_hidden, batch_first=True, bidirectional=True)
        # Input to latent heads is 2 * enc_hidden (bidirectional)
        self.z_mean = nn.Linear(2 * enc_hidden, latent_dim)
        self.z_logvar = nn.Linear(2 * enc_hidden, latent_dim)
        
        # --- Variational Self-Attention (VS) ---
        # Takes encoder states (size 2 * enc_hidden) as input
        self.attn_mean = nn.Linear(2 * enc_hidden, attn_vec_size)
        self.attn_logvar = nn.Linear(2 * enc_hidden, attn_vec_size)
        
        # --- Decoder (BiLSTM) ---
        # Input: Latent + Attention Vector
        self.dec_bilstm = nn.LSTM(latent_dim + attn_vec_size, dec_hidden, batch_first=True, bidirectional=True)
        # Output heads (Laplace params)
        self.x_mean = nn.Linear(2 * dec_hidden, feats)
        self.x_logvar = nn.Linear(2 * dec_hidden, feats)

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
        bs, seq, _ = x.shape
        
        # 1. Encoder
        # Add slight noise during training for robustness
        if training:
            x = x + torch.randn_like(x) * 0.1
            
        enc_out, _ = self.enc_bilstm(x) # [B, T, 2*enc_hidden]
        
        # Latent z comes from last hidden state
        last_state = enc_out[:, -1, :]
        z_mu = self.z_mean(last_state)
        z_lv = self.z_logvar(last_state)
        z, z_std = self.reparameterize(z_mu, z_lv, training)
        
        # 2. Variational Self-Attention
        # Scaled Dot-Product Attention on Encoder Outputs
        d_k = enc_out.size(-1)
        scores = torch.matmul(enc_out, enc_out.transpose(-2, -1)) / math.sqrt(d_k)
        attn_probs = F.softmax(scores, dim=-1) # [B, T, T]
        
        # Apply attention to values
        attn_val = torch.matmul(attn_probs, enc_out) # [B, T, 2*enc_hidden]
        
        # Get variational attention params
        a_mu = self.attn_mean(attn_val)
        a_lv = self.attn_logvar(attn_val)
        a, a_std = self.reparameterize(a_mu, a_lv, training) # [B, T, attn_vec_size]
        
        # 3. Decoder
        # Expand z to sequence length
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1) # [B, T, latent_dim]
        
        # Concatenate Z and Attention
        dec_in = torch.cat([z_rep, a], dim=-1) # [B, T, latent + attn]
        
        dec_out, _ = self.dec_bilstm(dec_in)
        
        x_mu = self.x_mean(dec_out)
        x_lv = self.x_logvar(dec_out)
        
        # For reconstruction, we just return parameters (Laplace distribution)
        # Using softplus for scale
        x_scale = torch.sqrt(F.softplus(x_lv) + 1e-8)
        
        return x_mu, x_scale, z_mu, z_std, a_mu, a_std


class VS_VAE(BaseDetector):
    def __init__(self,
                 win_size=32,
                 feats=1,
                 latent_dim=3,
                 attn_vec_size=3,
                 batch_size=512,
                 epochs=100,
                 patience=15,
                 lr=1e-3,
                 beta_start=1e-8,
                 beta_end=1e-2,
                 att_beta=1e-2,
                 validation_size=0.2):
        super().__init__()
        
        self.cuda = True
        self.device = get_gpu(self.cuda)
        
        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        
        # KL Annealing Params
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.att_beta = att_beta
        self.current_beta = beta_start

        self.model = VS_VAE_Model(
            feats=feats, 
            seq_len=win_size, 
            latent_dim=latent_dim,
            attn_vec_size=attn_vec_size
        ).to(self.device)

        # Adam with amsgrad=True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

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
        
        # Annealing Schedule
        anneal_epochs = 25
        step_size = (self.beta_end - self.beta_start) / anneal_epochs

        for epoch in range(1, self.epochs + 1):
            # Update Beta
            if epoch <= anneal_epochs:
                self.current_beta = self.beta_start + step_size * (epoch - 1)
            else:
                self.current_beta = self.beta_end
                
            self.model.train()
            avg_loss = 0
            
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device)

                x_mu, x_scale, z_mu, z_std, a_mu, a_std = self.model(d, training=True)
                
                # 1. Reconstruction Loss (Laplace Log Prob)
                dist = Laplace(x_mu, x_scale)
                # Mean over all dimensions (batch, seq, feat)
                log_prob = dist.log_prob(d).mean() 
                nll_loss = -log_prob
                
                # 2. Latent KL Divergence (Standard VAE KL)
                p_z = Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                q_z = Normal(z_mu, z_std)
                # Sum latent, mean batch
                kl_z = kl_divergence(q_z, p_z).sum(dim=1).mean()
                
                # 3. Attention KL Divergence
                p_a = Normal(torch.zeros_like(a_mu), torch.ones_like(a_std))
                q_a = Normal(a_mu, a_std)
                # Sum attention dim, mean batch & seq
                kl_a = kl_divergence(q_a, p_a).sum(dim=-1).mean()
                
                # 4. L1 Regularization (Manually added)
                l1_reg = 0.0
                for param in self.model.parameters():
                    l1_reg += torch.norm(param, p=1)
                l1_loss = 1e-8 * l1_reg
                
                # Total Loss
                loss = nll_loss + self.current_beta * (kl_z + self.att_beta * kl_a) + l1_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
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
                        x_mu, x_scale, _, _, _, _ = self.model(d, training=False)
                        
                        dist = Laplace(x_mu, x_scale)
                        nll_loss = -dist.log_prob(d).mean()
                        
                        # Use NLL for validation / Early Stopping
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
                
                x_mu, x_scale, _, _, _, _ = self.model(d, training=False)
                
                # Anomaly Score = Negative Log Likelihood (Laplace)
                dist = Laplace(x_mu, x_scale)
                log_prob = dist.log_prob(d)
                
                # Sum over features (dim=-1), Mean over time (dim=1)
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