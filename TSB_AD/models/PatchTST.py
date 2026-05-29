from types import SimpleNamespace
from typing import Dict, Optional

import math
import os
import shutil

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
import torchinfo
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import (
    EarlyStoppingTorch,
    adjust_learning_rate,
    get_gpu,
)


class TriangularCausalMask:
    def __init__(self, B: int, L: int, device: torch.device = torch.device("cpu")):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self.mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
            )


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, n_vars, seq_len]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, n_vars, patch_num, patch_len] -> [B*n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                cur_delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=cur_delta)
                x = conv_layer(x)
                attns.append(attn)

            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool = True, factor: int = 5, scale=None, attention_dropout: float = 0.1, output_attention: bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or (1.0 / math.sqrt(E))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, float("-inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model: int, n_heads: int, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [B, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST-style reconstruction model for anomaly detection.
    """

    def __init__(self, configs, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, "pred_len", configs.seq_len)
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            configs.input_c,
            self.head_nf,
            configs.seq_len,
            head_dropout=configs.dropout,
        )

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B, L, D]
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)  # [B, D, L]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*D, patch_num, d_model]

        enc_out, _ = self.encoder(enc_out)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )  # [B, D, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, D, d_model, patch_num]

        dec_out = self.head(enc_out)  # [B, D, L]
        dec_out = dec_out.permute(0, 2, 1)  # [B, L, D]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.anomaly_detection(x_enc)


class PatchTST:
    def __init__(
        self,
        win_size: int = 96,
        input_c: int = 1,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-4,
        patience: int = 3,
        features: str = "M",
        lradj: str = "type1",
        validation_size: float = 0.2,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        factor: int = 5,
        activation: str = "gelu",
        e_layers: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        use_cuda: bool = True,
    ):
        super().__init__()

        self.win_size = win_size
        self.input_c = input_c
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.features = features
        self.lradj = lradj
        self.validation_size = validation_size

        self.__anomaly_score = None
        self.y_hats = None
        self.save_path = None

        self.cuda = use_cuda and torch.cuda.is_available()
        self.device = get_gpu(self.cuda)

        configs = SimpleNamespace(
            seq_len=self.win_size,
            pred_len=self.win_size,
            input_c=self.input_c,
            c_out=self.input_c,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            factor=factor,
            activation=activation,
            e_layers=e_layers,
        )

        self.model = Model(configs=configs, patch_len=patch_len, stride=stride).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.anomaly_criterion = nn.MSELoss(reduction="none")

        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)
        self.input_shape = (self.batch_size, self.win_size, self.input_c)

    def fit(self, data):
        split_idx = int((1 - self.validation_size) * len(data))
        tsTrain = data[:split_idx]
        tsValid = data[split_idx:]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for i, (batch_x, _) in loop:
                self.model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)

                loss.backward()
                self.model_optim.step()

                train_loss += loss.item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss / (i + 1))

            self.model.eval()
            total_loss = []

            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with torch.no_grad():
                for i, (batch_x, _) in loop:
                    batch_x = batch_x.float().to(self.device)
                    outputs = self.model(batch_x)

                    if self.features == "MS":
                        outputs_eval = outputs[:, :, -1:]
                        true_eval = batch_x[:, :, -1:]
                    else:
                        outputs_eval = outputs
                        true_eval = batch_x

                    loss = self.criterion(outputs_eval, true_eval)
                    total_loss.append(loss.item())

                    loop.set_description(f"Valid Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(batch_loss=loss.item())

            valid_loss = float(np.mean(total_loss)) if total_loss else 0.0
            self.early_stopping(valid_loss, self.model)

            print(f"Epoch {epoch}: train_loss={train_loss / max(len(train_loader), 1):.6f}, valid_loss={valid_loss:.6f}")

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.model_optim, epoch + 1, self.lradj, self.lr)

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        attens_energy = []
        y_hats = []

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for i, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)

                # [B, L, D] -> mean over features => [B, L]
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)

                # keep last timestep reconstruction
                last_score = score[:, -1].detach().cpu().numpy()  # [B]
                last_pred = outputs[:, -1, :].detach().cpu().numpy()  # [B, D]

                attens_energy.append(last_score)
                y_hats.append(last_pred)

                loop.set_description("Testing Phase")

        scores = np.concatenate(attens_energy, axis=0).reshape(-1)
        y_hats = np.concatenate(y_hats, axis=0)  # [num_windows, D]

        self.__anomaly_score = scores
        self.y_hats = y_hats

        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

        # pad window-based scores back to point-wise length
        if self.__anomaly_score.shape[0] < len(data):
            left_pad = math.ceil((self.win_size - 1) / 2)
            right_pad = (self.win_size - 1) // 2
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]] * left_pad
                + list(self.__anomaly_score)
                + [self.__anomaly_score[-1]] * right_pad
            )

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def get_y_hat(self) -> np.ndarray:
        return self.y_hats

    def param_statistic(self, save_file: str):
        model_stats = torchinfo.summary(self.model, input_size=self.input_shape, verbose=0)
        with open(save_file, "w") as f:
            f.write(str(model_stats))