"""
This function is adapted from [W-VAE]
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

class WVAEModel(nn.Module):
    def __init__(self, 
                 feats, 
                 seq_len, 
                 latent_dim=5, 
                 enc_hidden=128, 
                 dec_hidden=128,
                 noise_std=0.8):
        super(WVAEModel, self).__init__()
        self.name = 'WVAE'
        self.n_feats = feats
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        
        # --- Encoder (BiLSTM) ---
        self.enc_bilstm = nn.LSTM(feats, enc_hidden, batch_first=True, bidirectional=True)
        
        # Latent heads (Input = 2 * enc_hidden)
        self.z_mean = nn.Linear(enc_hidden * 2, latent_dim)
        self.z_logvar = nn.Linear(enc_hidden * 2, latent_dim)
        
        # --- Decoder (BiLSTM) ---
        # Input to decoder is latent dimension (repeated)
        self.dec_bilstm = nn.LSTM(latent_dim, dec_hidden, batch_first=True, bidirectional=True)
        
        # Reconstruction heads (Input = 2 * dec_hidden)
        self.x_mean = nn.Linear(dec_hidden * 2, feats)
        self.x_logvar = nn.Linear(dec_hidden * 2, feats)

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
        
        # 1. Noise Injection (Specific to W-VAE robustness)
        if training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
            
        # 2. Encoder
        enc_out, _ = self.enc_bilstm(x)
        # Take last hidden state
        last_state = enc_out[:, -1, :] 
        
        z_mu = self.z_mean(last_state)
        z_lv = self.z_logvar(last_state)
        z, z_std = self.reparameterize(z_mu, z_lv, training)
        
        # 3. Decoder
        # Repeat latent vector for sequence length
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1) # [B, T, latent_dim]
        
        dec_out, _ = self.dec_bilstm(z_rep)
        
        x_mu = self.x_mean(dec_out)
        x_lv = self.x_logvar(dec_out)
        
        # Prepare std for output distribution
        x_std = torch.sqrt(F.softplus(x_lv) + 1e-8)
        
        return x_mu, x_std, z_mu, z_std


class WVAE(BaseDetector):
    def __init__(self,
                 win_size=32,
                 feats=1,
                 latent_dim=5,
                 batch_size=512,
                 epochs=100,
                 patience=15,
                 lr=1e-3,
                 clipnorm=5.0,
                 beta_start=1e-8,
                 beta_end=1e-2,
                 anneal_epochs=25,
                 noise_std=0.8,
                 l1_reg=1e-7,
                 validation_size=0.2):
        super().__init__()
        
        self.cuda = True
        self.device = get_gpu(self.cuda)
        
        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        
        # Training specifics
        self.clipnorm = clipnorm
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_epochs = anneal_epochs
        self.current_beta = beta_start
        self.l1_reg = l1_reg

        self.model = WVAEModel(
            feats=feats, 
            seq_len=win_size, 
            latent_dim=latent_dim,
            noise_std=noise_std
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
        
        # Annealing calculation helper
        step_size = (self.beta_end - self.beta_start) / self.anneal_epochs

        for epoch in range(1, self.epochs + 1):
            # Beta Annealing Update
            if epoch <= self.anneal_epochs:
                self.current_beta = self.beta_start + step_size * (epoch - 1)
            else:
                self.current_beta = self.beta_end

            self.model.train()
            avg_loss = 0
            
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device)

                x_mu, x_std, z_mu, z_std = self.model(d, training=True)
                
                # 1. Reconstruction Loss (NLL)
                dist = Normal(x_mu, x_std)
                # Sum over time, Mean over batch (features assumed independent)
                nll_loss = -dist.log_prob(d).sum(dim=1).mean()
                
                # 2. KL Divergence
                p_z = Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
                q_z = Normal(z_mu, z_std)
                # Sum latent, Mean batch
                kl_loss = kl_divergence(q_z, p_z).sum(dim=1).mean()
                
                # 3. L1 Regularization
                l1_loss = 0.0
                for param in self.model.parameters():
                    l1_loss += torch.norm(param, p=1)
                l1_loss = self.l1_reg * l1_loss
                
                # Total Loss
                loss = nll_loss + self.current_beta * kl_loss + l1_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping (Clipnorm 5.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)
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
                        # No noise in validation
                        x_mu, x_std, _, _ = self.model(d, training=False)
                        
                        dist = Normal(x_mu, x_std)
                        nll_loss = -dist.log_prob(d).sum(dim=1).mean()
                        
                        # Standard metric for early stopping is usually just reconstruction/NLL
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