"""
GDN (torch_geometric-free) re-implementation compatible with your pipeline + MPS:
- Uses ReconstructDataset
- Matches the pasted GDN structure: node embeddings -> top-k graph -> attention message passing -> OutLayer -> (B, F)
- No torch_geometric dependency; uses pure PyTorch edge_index + index_add segment-softmax
- Training target: predict LAST STEP of each window (no leakage if input uses history-only)
- Scoring: last-step per-sensor error aggregated to scalar (TSB-AD style window score)
- Center padding to length T
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
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu


# -----------------------------
#   Edge utilities (pure torch)
# -----------------------------
def remove_self_loops(edge_index: torch.Tensor):
    # edge_index: (2, E)
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    return edge_index[:, mask]


def add_self_loops(edge_index: torch.Tensor, num_nodes: int):
    # ensure every node has i->i
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    self_loops = torch.stack([self_loops, self_loops], dim=0)  # (2, N)
    return torch.cat([edge_index, self_loops], dim=1)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    # Vectorized version: no Python loops, pure tensor ops
    device = org_edge_index.device
    E = org_edge_index.size(1)
    offsets = (torch.arange(batch_num, device=device, dtype=org_edge_index.dtype) * node_num)  # (B,)
    edge = org_edge_index[:, :, None] + offsets[None, None, :]  # (2, E, B)
    return edge.permute(0, 2, 1).reshape(2, batch_num * E).long().contiguous()


def segment_softmax(scores: torch.Tensor, dst_index: torch.Tensor, num_dst: int, eps: float = 1e-12):
    """
    scores: (E, H) or (E,)  attention logits per edge
    dst_index: (E,)         destination node id per edge (0..num_dst-1)
    returns: same shape as scores: softmax over edges grouped by dst_index
    
    Fully vectorized: no Python loops, no device-to-CPU syncs.
    """
    if scores.ndim == 1:
        scores = scores[:, None]
        squeeze_back = True
    else:
        squeeze_back = False

    E, H = scores.shape
    idx = dst_index[:, None].expand(E, H)  # (E, H)

    # max per dst using scatter_reduce_
    max_per_dst = torch.full((num_dst, H), -1e30, device=scores.device, dtype=scores.dtype)
    max_per_dst.scatter_reduce_(0, idx, scores, reduce="amax", include_self=True)

    # stable exp
    exp_scores = torch.exp(scores - max_per_dst[dst_index])
    
    # sum per dst using index_add_
    sum_per_dst = torch.zeros((num_dst, H), device=scores.device, dtype=scores.dtype)
    sum_per_dst.index_add_(0, dst_index, exp_scores)

    out = exp_scores / (sum_per_dst[dst_index] + eps)
    return out.squeeze(1) if squeeze_back else out


# -----------------------------
#   Official-code building blocks
# -----------------------------
class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out


class GraphLayer(nn.Module):
    """
    Pure-torch stand-in for the torch_geometric MessagePassing layer in your pasted code.

    It matches the key operations:
      - linear projection: lin(x) -> heads*out_channels
      - attention logits: (key_i * cat_att_i).sum + (key_j * cat_att_j).sum
      - leaky_relu + softmax over incoming edges per destination
      - message: x_j * alpha
      - aggregate: sum messages per destination
      - concat or mean heads + bias
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
        inter_dim=-1,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)

        self.lin = nn.Linear(self.in_channels, self.heads * self.out_channels, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_j = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_em_i = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_em_j = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * self.out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.__alpha__ = None  # for optional inspection

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_i)
        nn.init.xavier_uniform_(self.att_j)
        nn.init.zeros_(self.att_em_i)
        nn.init.zeros_(self.att_em_j)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """
        x: (N, in_channels) where N = batch_num * node_num
        edge_index: (2, E) with src=edge_index[0], dst=edge_index[1]
        embedding: (N, emb_dim) repeated per batch (as in original code)
        """
        # project
        x_proj = self.lin(x)  # (N, H*out)
        x_proj = x_proj.view(-1, self.heads, self.out_channels)  # (N, H, out)

        # remove/add self loops like original
        edge_index = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x_proj.size(0))

        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = x_proj.size(0)

        x_i = x_proj[dst]  # destination features
        x_j = x_proj[src]  # source features

        if embedding is not None:
            emb_i = embedding[dst].unsqueeze(1).repeat(1, self.heads, 1)
            emb_j = embedding[src].unsqueeze(1).repeat(1, self.heads, 1)
            key_i = torch.cat([x_i, emb_i], dim=-1)
            key_j = torch.cat([x_j, emb_j], dim=-1)
            cat_att_i = torch.cat([self.att_i, self.att_em_i], dim=-1)
            cat_att_j = torch.cat([self.att_j, self.att_em_j], dim=-1)
        else:
            key_i = x_i
            key_j = x_j
            cat_att_i = self.att_i
            cat_att_j = self.att_j

        # attention logits per edge per head: (E, H)
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)

        # softmax across incoming edges per destination node
        alpha = segment_softmax(alpha, dst, num_dst=num_nodes)  # (E,H)

        if return_attention_weights:
            self.__alpha__ = alpha.detach()

        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # message: x_j * alpha
        msg = x_j * alpha.unsqueeze(-1)  # (E,H,out)

        # aggregate sum over dst
        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)  # sum messages into destination nodes

        # concat heads or mean
        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, self.__alpha__)
        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)
        return self.relu(out)


# -----------------------------
#   Core model (matches your pasted GDN nn.Module)
# -----------------------------
class GDNModel(nn.Module):
    """
    Same interface/shape behavior as your pasted torch_geometric model:
    Input:  data (B, node_num, input_dim)
    Output: out  (B, node_num)
    """
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
        super(GDNModel, self).__init__()

        self.edge_index_sets = edge_index_sets
        self.topk = int(topk)
        self.learned_graph = None

        embed_dim = int(dim)
        self.embedding = nn.Embedding(int(node_num), embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(int(input_dim), int(dim), inter_dim=int(dim) + embed_dim, heads=1) for _ in range(edge_set_num)]
        )

        self.out_layer = OutLayer(int(dim) * edge_set_num, int(node_num), int(out_layer_num), inter_num=int(out_layer_inter_dim))
        self.cache_edge_index_sets = [None] * edge_set_num
        self.dp = nn.Dropout(0.2)

        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index=None):
        x = data
        device = x.device
        edge_index_sets = self.edge_index_sets

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()  # (B*node_num, input_dim)

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            _ = self.cache_edge_index_sets[i]  # computed but not used in original (kept for fidelity)

            all_embeddings = self.embedding(torch.arange(node_num, device=device))  # (node_num, dim)

            weights = all_embeddings.detach()
            all_embeddings = all_embeddings.repeat(batch_num, 1)  # (B*node_num, dim)

            weights = weights.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / (normed_mat + 1e-12)

            topk_num = int(min(self.topk, node_num))
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num, device=device).unsqueeze(1).repeat(1, topk_num).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)  # (2, node_num*topk)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)

            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)  # (B*node_num, dim*edge_set_num)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num, device=device)
        out = torch.mul(x, self.embedding(indexes))  # (B,node,dim*)

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)  # (B,node_num)

        return out


# -----------------------------
#   Detector wrapper (keeps class name GDN)
# -----------------------------
class GDN(BaseDetector):
    def __init__(
        self,
        win_size=100,
        feats=1,
        dim=64,
        out_layer_inter_dim=256,
        out_layer_num=1,
        topk=20,
        batch_size=128,
        epochs=30,
        patience=10,
        lr=0.001,
        validation_size=0.2,
        score_mode="mse_last",  # "mse_last" or "mae_last"
    ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = int(win_size)
        self.feats = int(feats)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.validation_size = float(validation_size)
        self.score_mode = str(score_mode)

        # satisfy edge_index_sets loop (original model uses learned gated graph anyway)
        dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_index_sets = [dummy_edge_index]

        # input_dim = window history length (L-1)
        input_dim = max(1, self.win_size - 1)

        self.model = GDNModel(
            edge_index_sets=edge_index_sets,
            node_num=self.feats,
            dim=dim,
            out_layer_inter_dim=out_layer_inter_dim,
            input_dim=input_dim,
            out_layer_num=out_layer_num,
            topk=topk,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def _window_to_graph_input(self, d: torch.Tensor):
        """
        d: (B, L, F) from ReconstructDataset
        Build node features as history-only (no leakage):
          X_hist = d[:, :-1, :]  -> (B, L-1, F)
          Xg     = (B, F, L-1)
        Target is last step:
          y = d[:, -1, :]        -> (B, F)
        """
        X_hist = d[:, :-1, :]
        y = d[:, -1, :]
        Xg = X_hist.permute(0, 2, 1).contiguous()
        return Xg, y

    def fit(self, data):
        tsTrain = data[: int((1 - self.validation_size) * len(data))]
        tsValid = data[int((1 - self.validation_size) * len(data)) :]

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
            avg_loss = 0.0

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.float().to(self.device)
                Xg, y = self._window_to_graph_input(d)

                yhat = self.model(Xg, None)
                loss = self.criterion(yhat, y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                avg_loss += float(loss.item())
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=float(loss.item()))

            if len(valid_loader) > 0:
                self.model.eval()
                val_loss = 0.0
                with torch.inference_mode():
                    for d, _ in valid_loader:
                        d = d.float().to(self.device)
                        Xg, y = self._window_to_graph_input(d)
                        yhat = self.model(Xg, None)
                        val_loss += float(self.criterion(yhat, y).item())
                avg = val_loss / max(1, len(valid_loader))
            else:
                avg = avg_loss / max(1, len(train_loader))

            self.early_stopping(avg, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        scores = []

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.inference_mode():
            for idx, (d, _) in loop:
                d = d.float().to(self.device)
                Xg, y = self._window_to_graph_input(d)

                yhat = self.model(Xg, None)

                if self.score_mode == "mse_last":
                    e = (yhat - y) ** 2
                    s = e.mean(dim=-1)
                elif self.score_mode == "mae_last":
                    e = torch.abs(yhat - y)
                    s = e.mean(dim=-1)
                else:
                    raise ValueError(f"Unknown score_mode={self.score_mode}")

                scores.append(s.detach().cpu())

        scores = torch.cat(scores, dim=0).numpy()
        self.__anomaly_score = scores

        # center padding to match length T
        if self.__anomaly_score.shape[0] < len(data):
            pad_l = math.ceil((self.win_size - 1) / 2)
            pad_r = (self.win_size - 1) // 2
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]] * pad_l
                + list(self.__anomaly_score)
                + [self.__anomaly_score[-1]] * pad_r,
                dtype=np.float32,
            )
            self.__anomaly_score = self.__anomaly_score[: len(data)]

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
