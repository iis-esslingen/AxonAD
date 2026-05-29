# -*- coding: utf-8 -*-
# Author: Gerardo Pastrana
# License: Apache-2.0 License
#
# CHARM (CHannel Aware Representation Model) anomaly detector for TSB-AD.
#
# Uses the hosted CHARM API (c3-charm SDK) to embed sliding windows of
# a time series, then scores each test window by its k-nearest-neighbor
# cosine distance to clean training windows.  Overlapping window scores
# are aggregated to per-timestep anomaly scores.
#
# Requires:  pip install TSB-AD c3-charm

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.slidingWindows import find_length_rank

from charm import CharmClient


def _effective_window(
    series_length: int, window_size: int, stride: int, k: int, min_window: int = 64
) -> tuple:
    """Pick the largest usable (window_size, stride) for a given series.

    Gracefully handles short series by reducing window size or stride.

    Returns (eff_ws, eff_stride) or None if the series is too short.
    """
    if series_length < min_window:
        return None
    min_windows = max(2 * k, 10)

    ws = min(window_size, series_length - (min_windows - 1) * stride)
    if ws >= min_window:
        return ws, stride

    ws = min(window_size, series_length - (min_windows - 1))
    if ws >= min_window:
        return ws, 1

    ws = min(window_size, series_length)
    available = series_length - ws + 1
    if available >= min_windows:
        eff_stride = max(1, (series_length - ws) // (min_windows - 1))
        return ws, eff_stride

    return ws, 1


def _create_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Slide a (T, C) array into (N_windows, window_size, C) windows."""
    T, C = data.shape
    starts = np.arange(0, max(T - window_size + 1, 1), stride)
    return np.stack([data[s : min(s + window_size, T)] for s in starts])


def _window_scores_to_pointwise(
    window_scores: np.ndarray,
    window_size: int,
    stride: int,
    total_length: int,
    mode: str = "mean",
) -> np.ndarray:
    """Aggregate overlapping window-level scores to per-timestep scores."""
    n_windows = len(window_scores)

    if mode == "mean":
        if stride == 1 and n_windows > 0:
            cs = np.empty(n_windows + 1, dtype=np.float64)
            cs[0] = 0.0
            np.cumsum(window_scores, out=cs[1:])
            t = np.arange(total_length)
            w_start = np.clip(t - window_size + 1, 0, n_windows)
            w_end = np.clip(t + 1, 0, n_windows)
            accum = cs[w_end] - cs[w_start]
            count = np.maximum(w_end - w_start, 1).astype(np.float64)
            return (accum / count).astype(np.float32)

        accum = np.zeros(total_length, dtype=np.float64)
        count = np.zeros(total_length, dtype=np.float64)
        for i, score in enumerate(window_scores):
            start = i * stride
            end = min(start + window_size, total_length)
            accum[start:end] += score
            count[start:end] += 1
        count = np.maximum(count, 1)
        return (accum / count).astype(np.float32)

    elif mode == "max":
        result = np.full(total_length, -np.inf, dtype=np.float64)
        for i, score in enumerate(window_scores):
            start = i * stride
            end = min(start + window_size, total_length)
            result[start:end] = np.maximum(result[start:end], score)
        result[result == -np.inf] = 0.0
        return result.astype(np.float32)

    elif mode == "last":
        result = np.zeros(total_length, dtype=np.float64)
        for i, score in enumerate(window_scores):
            start = i * stride
            end = min(start + window_size, total_length)
            result[start:end] = score
        return result.astype(np.float32)

    elif mode == "center":
        result = np.zeros(total_length, dtype=np.float32)
        if n_windows == 0:
            return result
        half = window_size // 2
        center_indices = np.arange(n_windows) * stride + half
        center_indices = np.clip(center_indices, 0, total_length - 1)
        np.put(result, center_indices, window_scores)
        result[: min(half, total_length)] = window_scores[0]
        right_edge = min(center_indices[-1] + 1, total_length)
        if right_edge < total_length:
            result[right_edge:] = window_scores[-1]
        return result

    raise ValueError(f"Unknown pointwise aggregation mode: {mode}")


class CHARM_AD(BaseDetector):
    """Embedding-based kNN anomaly detector using the hosted CHARM API."""

    def __init__(
        self,
        HP: dict,
        window_size: int = 128,
        stride: int = 1,
        k: int = 3,
        pointwise_agg: str = "mean",
        api_batch_size: int = 64,
        train_stride: int = 1,
        min_window: int = 64,
        normalize: bool = True,
    ):
        super().__init__()
        self.HP = HP
        self.window_size = window_size
        self.stride = stride
        self.k = k
        self.pointwise_agg = pointwise_agg
        self.api_batch_size = api_batch_size
        self.train_stride = train_stride
        self.min_window = min_window
        self.normalize = normalize

        self.client = CharmClient(
            base_url="http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080",
            timeout=120,
            api_key="token",
        )
        self.train_embeddings_ = None

    def _embed_windows(self, windows: np.ndarray) -> np.ndarray:
        """Embed (N, W, C) windows via the CHARM API → (N, D).

        The API returns (N, patches, C, D); we aggregate by averaging
        over patches and channels to get one vector per window.
        """
        N, W, C = windows.shape
        descriptions = [[f"channel_{c}" for c in range(C)]] * N
        ts_list = windows.tolist()

        max_per_request = max(1, 500_000 // (W * C))
        batch_size = min(self.api_batch_size, max_per_request)

        resp = self.client.embeddings.create(
            descriptions=descriptions,
            ts_array=ts_list,
            batch_size=batch_size,
            return_tensors="np",
            progress=True,
        )
        embs = resp.embeds  # (N, patches, C, D) or (N, D)
        if embs.ndim == 4:
            embs = embs.mean(axis=(1, 2))  # → (N, D)
        elif embs.ndim == 3:
            embs = embs.mean(axis=1)

        nan_mask = np.isnan(embs).any(axis=-1)
        if nan_mask.any():
            print(
                f"  WARNING: {nan_mask.sum()}/{len(embs)} embeddings contain NaN, replacing with 0"
            )
            embs = np.nan_to_num(embs, nan=0.0)

        return embs

    @staticmethod
    def _cosine_knn(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
        """Cosine kNN distances: for each query row, mean of top-k distances.

        query: (Nq, D)   ref: (Nr, D)   returns: (Nq,)
        """
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        r_norm = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)

        chunk_size = 2048
        scores = []
        for start in range(0, len(q_norm), chunk_size):
            q_chunk = q_norm[start : start + chunk_size]
            sim = q_chunk @ r_norm.T  # (chunk, Nr)
            dist = 1.0 - sim
            dist = np.nan_to_num(dist, nan=1.0)
            k_actual = min(k, dist.shape[1])
            topk_idx = np.argpartition(dist, k_actual - 1, axis=1)[:, :k_actual]
            topk_dists = np.take_along_axis(dist, topk_idx, axis=1)
            scores.append(topk_dists.mean(axis=1))
        return np.concatenate(scores)

    def fit(self, X, y=None):
        """Embed training windows and store as kNN reference set."""
        T, C = X.shape
        result = _effective_window(
            T, self.window_size, self.train_stride, self.k, self.min_window
        )
        if result is None:
            print(
                f"  WARNING: train series too short ({T} < {self.min_window}), using full series as single window"
            )
            self.eff_train_ws_ = T
            self.eff_train_stride_ = 1
            self.train_embeddings_ = self._embed_windows(X[np.newaxis])
        else:
            self.eff_train_ws_, self.eff_train_stride_ = result
            windows = _create_windows(X, self.eff_train_ws_, self.eff_train_stride_)
            self.train_embeddings_ = self._embed_windows(windows)
        self.decision_scores_ = np.zeros(T)
        return self

    def decision_function(self, X):
        """Score each timestep by kNN distance of its embedding windows."""
        T, C = X.shape
        result = _effective_window(
            T, self.window_size, self.stride, self.k, self.min_window
        )
        if result is None:
            print(
                f"  WARNING: test series too short ({T} < {self.min_window}), returning zeros"
            )
            return np.zeros(T, dtype=np.float32)
        eff_ws, eff_stride = result

        windows = _create_windows(X, eff_ws, eff_stride)
        test_embs = self._embed_windows(windows)

        window_scores = self._cosine_knn(test_embs, self.train_embeddings_, self.k)

        if not np.isfinite(window_scores).all():
            window_scores = np.nan_to_num(
                window_scores, nan=0.0, posinf=0.0, neginf=0.0
            )

        pointwise = _window_scores_to_pointwise(
            window_scores,
            eff_ws,
            eff_stride,
            T,
            mode=self.pointwise_agg,
        )
        return pointwise