"""
This function is adapted from [Time-RCD] by [thu-sail-lab]
Original source: [https://github.com/thu-sail-lab/Time-RCD]
"""

import math
import os
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import BaseDetector


HF_REPO_ID = "thu-sail-lab/Time-RCD"
UNI_CHECKPOINT = "best_model/pretrain_checkpoint_best_uni.pth"
MULTI_CHECKPOINT = "best_model/pretrain_checkpoint_best_multi.pth"
EPS = 1e-8


def _to_2d_array(data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError("Time_RCD expects input with shape (n_samples, n_features).")
    if data.shape[0] == 0:
        raise ValueError("Time_RCD received an empty time series.")
    return data


def _build_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")

    device = str(device).strip().lower()
    if device == "cpu":
        return torch.device("cpu")
    if device.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")
    return torch.device(device)


class TimeRCDWindowDataset(Dataset):
    def __init__(self, data, window_size, stride=None, normalize=True):
        super().__init__()
        self.data = _to_2d_array(data)
        self.window_size = int(window_size)
        self.stride = int(stride or window_size)
        self.original_length = self.data.shape[0]

        if normalize:
            mean = np.mean(self.data, axis=0, keepdims=True)
            std = np.std(self.data, axis=0, keepdims=True)
            std = np.where(std < EPS, 1.0, std)
            self.data = (self.data - mean) / std

        remainder = self.data.shape[0] % self.window_size
        if remainder > 0:
            pad_length = self.window_size - remainder
            pad_values = np.repeat(self.data[-1:, :], pad_length, axis=0)
            self.data = np.concatenate([self.data, pad_values], axis=0)
            self.valid_mask = np.concatenate(
                [np.ones(self.original_length, dtype=bool), np.zeros(pad_length, dtype=bool)]
            )
        else:
            self.valid_mask = np.ones(self.data.shape[0], dtype=bool)

        self.num_windows = max(0, (self.data.shape[0] - self.window_size) // self.stride + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.window_size
        window = torch.tensor(self.data[start:end], dtype=torch.float32)
        mask = torch.tensor(self.valid_mask[start:end], dtype=torch.bool)
        return window, mask


@dataclass
class TimeSeriesConfig:
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 16
    num_layers: int = 8
    num_heads: int = 8
    d_ff_dropout: float = 0.1
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1


@dataclass
class TimeRCDConfig:
    ts_config: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    batch_size: int = 64
    dropout: float = 0.1


default_config = TimeRCDConfig()


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        norm_x = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x.type_as(self.scale)).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        time_idx = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        return torch.einsum("i,j->ij", time_idx, self.inv_freq)


class BinaryAttentionBias(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(2, num_heads)

    def forward(self, query_id, kv_id):
        same_feature = query_id.unsqueeze(-1).eq(kv_id.unsqueeze(-2)).unsqueeze(1)
        diff_bias = self.embedding.weight[0].view(1, -1, 1, 1)
        same_bias = self.embedding.weight[1].view(1, -1, 1, 1)
        return torch.where(same_feature, same_bias, diff_bias)


class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, num_features):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.binary_attention_bias = BinaryAttentionBias(num_heads) if num_features > 1 else None

    def apply_rope(self, x, freqs):
        batch_size, seq_len, embed_dim = x.shape
        x = x.view(batch_size, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)
        sin = freqs.sin().unsqueeze(0)
        rotated = torch.stack(
            [
                x[..., 0] * cos - x[..., 1] * sin,
                x[..., 0] * sin + x[..., 1] * cos,
            ],
            dim=-1,
        )
        return rotated.view(batch_size, seq_len, embed_dim)

    def forward(self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        batch_size, seq_len, _ = query.shape

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self.apply_rope(query, freqs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.apply_rope(key, freqs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.binary_attention_bias is not None and query_id is not None and kv_id is not None:
            scores = scores + self.binary_attention_bias(query_id, kv_id)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)


class LlamaMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = d_model * 4
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=True)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=True)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout, activation, num_features):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, num_features)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = LlamaMLP(d_model)
        self.activation = activation

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, freqs, src_id, src_id, attn_mask=attn_mask)
        src = residual + self.dropout(src)

        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        return residual + self.dropout(src)


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout, activation, num_layers, num_features):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithRoPE(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    activation=activation,
                    num_features=num_features,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, freqs, src_id, attn_mask=attn_mask)
        return src


class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_proj=256,
        patch_size=16,
        num_layers=8,
        num_heads=8,
        d_ff_dropout=0.1,
        use_rope=True,
        num_features=1,
        activation="gelu",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_proj = d_proj
        self.num_features = num_features
        self.use_rope = use_rope
        self.embedding_layer = nn.Linear(patch_size, d_model)
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        self.rope_embedder = RotaryEmbedding(d_model)
        self.transformer_encoder = CustomTransformerEncoder(
            d_model=d_model,
            nhead=num_heads,
            dropout=d_ff_dropout,
            activation=activation,
            num_layers=num_layers,
            num_features=num_features,
        )

    def forward(self, time_series, mask):
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)

        batch_size, seq_len, num_features = time_series.size()
        if num_features != self.num_features:
            raise ValueError(f"Expected {self.num_features} input features, got {num_features}.")

        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_len = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_len), value=0.0)
            mask = F.pad(mask, (0, pad_len), value=False)

        num_patches = padded_length // self.patch_size
        patches = time_series.view(batch_size, num_patches, self.patch_size, num_features)
        patches = patches.permute(0, 3, 1, 2).contiguous().view(batch_size, num_features * num_patches, self.patch_size)

        embedded_patches = self.embedding_layer(patches)

        patch_mask = mask.view(batch_size, num_patches, self.patch_size).sum(dim=-1) > 0
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1).reshape(batch_size, num_features * num_patches)
        feature_id = torch.arange(num_features, device=time_series.device).repeat_interleave(num_patches)
        feature_id = feature_id.unsqueeze(0).expand(batch_size, -1)
        freqs = self.rope_embedder(num_features * num_patches).to(time_series.device)

        encoded = self.transformer_encoder(embedded_patches, freqs, src_id=feature_id, attn_mask=full_mask)
        projected = self.projection_layer(encoded)
        local_embeddings = projected.view(batch_size, num_features, num_patches, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
        local_embeddings = local_embeddings.view(batch_size, padded_length, num_features, self.d_proj)
        return local_embeddings[:, :seq_len]


class TimeSeriesPretrainModel(nn.Module):
    def __init__(self, config: TimeRCDConfig):
        super().__init__()
        ts_config = config.ts_config
        self.ts_encoder = TimeSeriesEncoder(
            d_model=ts_config.d_model,
            d_proj=ts_config.d_proj,
            patch_size=ts_config.patch_size,
            num_layers=ts_config.num_layers,
            num_heads=ts_config.num_heads,
            d_ff_dropout=ts_config.d_ff_dropout,
            use_rope=ts_config.use_rope,
            num_features=ts_config.num_features,
            activation=ts_config.activation,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(ts_config.d_proj, ts_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ts_config.d_proj * 4, ts_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ts_config.d_proj * 4, 1),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(ts_config.d_proj, ts_config.d_proj // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ts_config.d_proj // 2, 2),
        )

    def forward(self, time_series, mask=None):
        return self.ts_encoder(time_series, mask)


class Time_RCD(BaseDetector):
    def __init__(
        self,
        win_size=15000,
        input_c=1,
        batch_size=64,
        device=None,
        checkpoint=None,
        model_id=HF_REPO_ID,
        cache_dir=None,
    ):
        self.model_name = "Time_RCD"
        self.win_size = int(win_size)
        self.input_c = int(input_c)
        self.batch_size = int(batch_size)
        self.device = _build_device(device)
        self.checkpoint = checkpoint
        self.model_id = model_id
        self.cache_dir = cache_dir

        self.config = deepcopy(default_config)
        self.config.batch_size = self.batch_size
        self.config.ts_config.num_features = self.input_c

        self.model = TimeSeriesPretrainModel(self.config).to(self.device)
        checkpoint_path = self._resolve_checkpoint()
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _checkpoint_name(self):
        return MULTI_CHECKPOINT if self.input_c > 1 else UNI_CHECKPOINT

    def _resolve_checkpoint(self):
        checkpoint_name = self._checkpoint_name()

        if self.checkpoint is not None:
            checkpoint_path = os.path.expanduser(str(self.checkpoint))
            if os.path.isdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, os.path.basename(checkpoint_name))
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Time_RCD checkpoint not found: {checkpoint_path}")
            return checkpoint_path

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Time_RCD requires huggingface_hub to download checkpoints automatically."
            ) from exc

        try:
            return hf_hub_download(
                repo_id=self.model_id,
                filename=self._checkpoint_name(),
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Time_RCD checkpoint '{checkpoint_name}' from '{self.model_id}'. "
                "Please check network access or pass a local checkpoint path."
            ) from exc

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load Time_RCD checkpoint. This usually means the selected uni/multi "
                "checkpoint does not match the input channel setting."
            ) from exc

    def fit(self, X, y=None):
        self.decision_scores_ = self.decision_function(X)
        return self

    def zero_shot(self, X):
        self.decision_scores_ = self.decision_function(X)
        return self.decision_scores_

    def decision_function(self, X):
        data = _to_2d_array(X)
        window_size = min(self.win_size, len(data))
        dataset = TimeRCDWindowDataset(
            data=data,
            window_size=window_size,
            stride=window_size,
            normalize=True,
        )
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)

        scores = []
        self.model.eval()
        with torch.inference_mode():
            for batch_x, batch_mask in loader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                local_embeddings = self.model(time_series=batch_x, mask=batch_mask)
                anomaly_logits = self.model.anomaly_head(local_embeddings).mean(dim=-2)
                anomaly_probs = torch.softmax(anomaly_logits, dim=-1)[..., 1]

                batch_scores = anomaly_probs.detach().cpu().numpy()
                batch_mask = batch_mask.detach().cpu().numpy()
                for sample_scores, sample_mask in zip(batch_scores, batch_mask):
                    scores.append(sample_scores[sample_mask])

        if not scores:
            return np.zeros(len(data), dtype=np.float32)

        anomaly_score = np.concatenate(scores, axis=0).astype(np.float32)
        if anomaly_score.shape[0] != len(data):
            anomaly_score = anomaly_score[: len(data)]
        return anomaly_score
