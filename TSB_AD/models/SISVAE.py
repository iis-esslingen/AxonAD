"""
Fast, paper-close SISVAE (VRNN/SISVAE-family) for your BaseDetector pipeline.

Keeps:
- q(z_t | x_t, h_{t-1})   (posterior)
- p(z_t | h_{t-1})        (conditional prior)
- p(x_t | z_t, h_{t-1})   (decoder)
- Smoothness: KL( p(x_t|.) || p(x_{t+1}|.) ) on decoder output distributions

Speed changes:
- No Python loop over time
- Use fused GRU to get h_{t} from phi_x(x_t) (causal)
- Analytic Gaussian NLL/KL (no torch.distributions / kl_divergence)
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu


_LOG_2PI = math.log(2.0 * math.pi)
_EPS = 1e-8


# -----------------------------
#   Analytic Gaussian helpers
# -----------------------------
def _softplus_var(logvar: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    # variance is strictly positive
    return F.softplus(logvar) + eps

def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Elementwise NLL of diagonal Normal(mu, std). Returns same shape as x.
    NLL = 0.5 * [ ((x-mu)^2 / var) + log(var) + log(2pi) ]
    """
    var = std * std + _EPS
    return 0.5 * (((x - mu) ** 2) / var + torch.log(var) + _LOG_2PI)

def gaussian_kl(mu_q: torch.Tensor, std_q: torch.Tensor, mu_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    """
    Elementwise KL( N(mu_q, std_q) || N(mu_p, std_p) ) for diagonal Gaussians.
    Returns same shape as mu_q.
    KL = log(std_p/std_q) + (var_q + (mu_q-mu_p)^2)/(2*var_p) - 1/2
    """
    var_q = std_q * std_q + _EPS
    var_p = std_p * std_p + _EPS
    return torch.log(std_p + _EPS) - torch.log(std_q + _EPS) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5


class SISVAEModel(nn.Module):
    """
    Fast SISVAE-family model:
      h_t from causal GRU over phi_x(x_t)
      prior/post/dec match the SISVAE conditional structure, vectorized over time
    """
    def __init__(self, feats, latent_dim, hidden_dim=200, phi_dim=None):
        super().__init__()
        self.name = "SISVAE"
        self.n_feats = feats
        self.n_hidden = hidden_dim
        self.n_latent = latent_dim

        if phi_dim is None:
            phi_dim = hidden_dim
        self.phi_dim = phi_dim

        # phi_x(x_t)
        self.phi_x = nn.Sequential(
            nn.Linear(feats, phi_dim),
            nn.ReLU(),
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU(),
        )

        # Causal backbone to produce h_t efficiently (fused kernel)
        self.backbone_gru = nn.GRU(
            input_size=phi_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Posterior q(z_t | x_t, h_{t-1})
        self.enc_net = nn.Sequential(
            nn.Linear(phi_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        # Prior p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # phi_z(z_t)
        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, phi_dim),
            nn.ReLU(),
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU(),
        )

        # Decoder p(x_t | z_t, h_{t-1})
        self.dec_net = nn.Sequential(
            nn.Linear(phi_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.x_mean = nn.Linear(hidden_dim, feats)
        self.x_logvar = nn.Linear(hidden_dim, feats)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, training: bool):
        var = _softplus_var(logvar)
        std = torch.sqrt(var)
        if training:
            eps = torch.randn_like(mu)
            return mu + std * eps, std
        return mu, std

    def forward(self, x, training=True):
        # x: [B, T, F]
        B, T, Fdim = x.shape
        device = x.device

        # phi_x for all timesteps (vectorized)
        phix = self.phi_x(x.reshape(B * T, Fdim)).reshape(B, T, self.phi_dim)

        # backbone GRU gives h_t; we need h_{t-1}
        h_all, _ = self.backbone_gru(phix)  # [B, T, H]
        h0 = torch.zeros(B, 1, self.n_hidden, device=device)
        h_prev = torch.cat([h0, h_all[:, :-1, :]], dim=1)  # [B, T, H]

        # prior p(z_t|h_{t-1})
        p_h = self.prior_net(h_prev.reshape(B * T, self.n_hidden)).reshape(B, T, self.n_hidden)
        p_mu = self.prior_mean(p_h)
        p_lv = self.prior_logvar(p_h)
        p_std = torch.sqrt(_softplus_var(p_lv))

        # enforce base prior at t=0: N(0, I) - non-in-place for MPS compatibility
        p_mu = p_mu.clone()
        p_mu[:, 0, :] = 0.0
        p_std = p_std.clone()
        p_std[:, 0, :] = 1.0

        # posterior q(z_t|x_t,h_{t-1})
        enc_in = torch.cat([phix, h_prev], dim=-1)  # [B,T,P+H]
        e_h = self.enc_net(enc_in.reshape(B * T, self.phi_dim + self.n_hidden)).reshape(B, T, self.n_hidden)
        q_mu = self.z_mean(e_h)
        q_lv = self.z_logvar(e_h)
        z, q_std = self._reparameterize(q_mu, q_lv, training)

        # decoder p(x_t|z_t,h_{t-1})
        phiz = self.phi_z(z.reshape(B * T, self.n_latent)).reshape(B, T, self.phi_dim)
        dec_in = torch.cat([phiz, h_prev], dim=-1)
        d_h = self.dec_net(dec_in.reshape(B * T, self.phi_dim + self.n_hidden)).reshape(B, T, self.n_hidden)

        x_mu = self.x_mean(d_h)
        x_lv = self.x_logvar(d_h)
        x_std = torch.sqrt(_softplus_var(x_lv))

        return x_mu, x_std, q_mu, q_std, p_mu, p_std


class SISVAE(BaseDetector):
    def __init__(self,
                 win_size=100,
                 feats=1,
                 latent_dim=40,
                 hidden_dim=200,
                 batch_size=128,
                 epochs=100,
                 patience=10,
                 lr=1e-3,
                 smooth_weight=0.5,
                 validation_size=0.2):
        super().__init__()

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        self.smooth_weight = smooth_weight

        self.model = SISVAEModel(
            feats=feats,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def loss_function(self, x, x_mu, x_std, z_mu, z_std, prior_mu, prior_std):
        """
        Reconstruction (NLL) + KL(q||p) + Smoothness on decoder distributions
        """
        # 1) Recon NLL: sum over (T,F), mean over batch
        nll = gaussian_nll(x, x_mu, x_std).sum(dim=(1, 2)).mean()

        # 2) KL: sum over (T,latent), mean over batch
        kl = gaussian_kl(z_mu, z_std, prior_mu, prior_std).sum(dim=(1, 2)).mean()

        # 3) Smoothness: KL( p(x_{t-1}|.) || p(x_t|.) ), sum over (T-1,F), mean over batch
        mu_prev, std_prev = x_mu[:, :-1, :], x_std[:, :-1, :]
        mu_curr, std_curr = x_mu[:,  1:, :], x_std[:,  1:, :]
        smooth = gaussian_kl(mu_prev, std_prev, mu_curr, std_curr).sum(dim=(1, 2)).mean()

        return nll, kl, smooth

    def fit(self, data):
        tsTrain = data[:int((1 - self.validation_size) * len(data))]
        tsValid = data[int((1 - self.validation_size) * len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            avg_loss = 0.0

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device)

                x_mu, x_std, z_mu, z_std, p_mu, p_std = self.model(d, training=True)
                nll, kl, smooth = self.loss_function(d, x_mu, x_std, z_mu, z_std, p_mu, p_std)

                loss = nll + kl + self.smooth_weight * smooth

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), nll=nll.item(), kl=kl.item(), sm=smooth.item())

            # Validation
            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0.0
                with torch.no_grad():
                    for idx, (d, _) in enumerate(valid_loader):
                        d = d.to(self.device)
                        x_mu, x_std, z_mu, z_std, p_mu, p_std = self.model(d, training=False)
                        nll, kl, smooth = self.loss_function(d, x_mu, x_std, z_mu, z_std, p_mu, p_std)
                        loss = nll + kl + self.smooth_weight * smooth
                        avg_loss_val += loss.item()
                avg_loss = avg_loss_val / max(1, len(valid_loader))
            else:
                avg_loss = avg_loss / max(1, len(train_loader))

            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        self.model.eval()
        scores = []

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.inference_mode():
            for idx, (d, _) in loop:
                d = d.to(self.device)

                x_mu, x_std, _, _, _, _ = self.model(d, training=False)

                # Window score: mean over time of per-step NLL summed over features
                nll_tf = gaussian_nll(d, x_mu, x_std).sum(dim=-1)  # [B,T]
                score_win = nll_tf.mean(dim=1)                    # [B]
                scores.append(score_win.detach().to("cpu"))

        scores = torch.cat(scores, dim=0).numpy()
        self.__anomaly_score = scores

        # Window padding (center padding to length T, matching your current convention)
        if self.__anomaly_score.shape[0] < len(data):
            pad_l = math.ceil((self.win_size - 1) / 2)
            pad_r = (self.win_size - 1) // 2
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]] * pad_l +
                list(self.__anomaly_score) +
                [self.__anomaly_score[-1]] * pad_r
            )

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
