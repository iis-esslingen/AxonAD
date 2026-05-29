import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf
from TSB_AD.utils.utility import zscore

EPS = 1e-6
MMPAD_DEFAULT_TIME_BUDGET = float(1.142528e13)
# This proxy threshold corresponds to the current 24-hour runtime target on the
# c5.4xlarge setup in use. It was anchored from a completed run with effective
# n_job=14 and should be recalibrated if the machine class or n_job policy
# changes.
def to_2d_ts(seq):
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return seq.reshape(-1, 1), True
    if seq.ndim == 2:
        return seq, False
    raise ValueError(f'Expected 1D or 2D array, got shape {seq.shape}.')


def validate_score(score, target_len, score_name='score'):
    score = np.asarray(score, dtype=float).reshape(-1)
    if score.shape[0] != target_len:
        raise ValueError(
            f'Unexpected {score_name} length {score.shape[0]} != {target_len}.')
    if np.any(~np.isfinite(score)):
        raise ValueError(f'Non-finite values found in {score_name}.')
    return score


def normalize_budget_mode(value):
    if value is None:
        return None
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in ['', 'none', 'null']:
            return None
        if value_norm == 'downsample':
            return 'downsample'
    raise ValueError(f'Unsupported budget_mode={value!r}.')


def resolve_n_dim(n_dim, total_dim):
    total_dim = int(total_dim)
    if total_dim < 1:
        raise ValueError(f'Invalid total_dim={total_dim}.')
    if n_dim is None:
        return total_dim
    n_dim_value = float(n_dim)
    if n_dim_value <= 0:
        raise ValueError(f'n_dim must be positive, got {n_dim}.')
    if n_dim_value < 1:
        n_dim_resolved = int(np.ceil(n_dim_value * total_dim))
    else:
        n_dim_resolved = int(n_dim_value)
    return max(1, min(n_dim_resolved, total_dim))


def _find_length_rank(data, rank=1):
    data = np.asarray(data, dtype=float).squeeze()
    if len(data.shape) > 1:
        return 0
    if rank == 0:
        return 1
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]

    try:
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]
        max_local_max = sorted_local_max[0]
        if rank == 1:
            max_local_max = sorted_local_max[0]
        if rank == 2:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    max_local_max = i
                    break
        if rank == 3:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]:
                    max_local_max = i
                    break

        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except Exception:
        return 125


def infer_periodic_sub_len(data_2d, periodicity):
    seq = np.asarray(data_2d, dtype=float)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    if seq.shape[0] <= 1:
        return 1
    seq = zscore(seq, axis=0, ddof=0)
    sub_len = int(_find_length_rank(seq[:, 0], rank=int(periodicity)))
    sub_len = max(64, sub_len)
    return max(1, min(sub_len, seq.shape[0]))


def compute_mmpad_proxy_cost(n_samples, n_dim, sub_len):
    n_samples = int(max(1, int(n_samples)))
    n_dim = int(max(1, int(n_dim)))
    sub_len = int(max(1, min(int(sub_len), n_samples)))
    n_eff = max(1, n_samples - sub_len + 1)
    proxy_cost = (n_eff ** 2) * n_dim * np.log2(max(n_dim, 2))
    return int(n_eff), float(proxy_cost)


def downsample_sequence(seq, factor):
    seq_arr = np.asarray(seq, dtype=float)
    if seq_arr.ndim == 1:
        seq_arr = seq_arr.reshape(-1, 1)
    factor = int(max(1, int(factor)))
    if factor == 1:
        return np.array(seq_arr, copy=True)
    return np.array(seq_arr[::factor], copy=True)


def resolve_mmpad_budget_state(data_2d, periodicity, budget_mode=None,
                               time_budget=MMPAD_DEFAULT_TIME_BUDGET):
    seq = np.asarray(data_2d, dtype=float)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    seq = np.array(seq, copy=True)
    budget_mode_norm = normalize_budget_mode(budget_mode)
    time_budget = float(MMPAD_DEFAULT_TIME_BUDGET if time_budget is None else time_budget)
    downsample_factor = 1

    while True:
        sub_len = infer_periodic_sub_len(seq, periodicity)
        _, proxy_cost = compute_mmpad_proxy_cost(
            n_samples=seq.shape[0],
            n_dim=seq.shape[1],
            sub_len=sub_len,
        )
        if (
            budget_mode_norm != 'downsample' or
            proxy_cost <= time_budget or
            seq.shape[0] <= 1
        ):
            break
        seq = np.array(seq[::2], copy=True)
        downsample_factor *= 2

    return {
        'data': seq,
        'downsample_factor': int(downsample_factor),
        'sub_len': int(sub_len),
    }


def upsample_score_linear(score, target_len):
    score = np.asarray(score, dtype=float).reshape(-1)
    target_len = int(max(1, int(target_len)))
    if score.shape[0] == target_len:
        return np.array(score, copy=True)
    if score.shape[0] == 1:
        return np.full(target_len, float(score[0]), dtype=float)

    x_src = np.arange(score.shape[0], dtype=float)
    x_dst = np.linspace(0.0, float(score.shape[0] - 1), target_len)
    return np.interp(x_dst, x_src, score).astype(float, copy=False)


def apply_exclude(dist_profile, idx, exclude_len, n_sub):
    exclude_start = max(idx - exclude_len, 0)
    exclude_end = min(idx + exclude_len, n_sub)
    dist_profile[exclude_start:exclude_end, :] = -np.inf
    return dist_profile


def mass_pre(seq, sub_len, freq_len):
    seq_len = len(seq)
    seq_pad = np.zeros(freq_len)
    seq_pad[0:seq_len] = seq
    seq_freq = np.fft.fft(seq_pad)
    seq_cum = np.cumsum(seq_pad)
    seq_sq_cum = np.cumsum(np.square(seq_pad))
    seq_sum = (
        seq_cum[sub_len - 1:seq_len] -
        np.concatenate(([0], seq_cum[:seq_len - sub_len]))
    )
    seq_sq_sum = (
        seq_sq_cum[sub_len - 1:seq_len] -
        np.concatenate(([0], seq_sq_cum[:seq_len - sub_len]))
    )
    seq_mu = seq_sum / sub_len
    seq_sig_sq = seq_sq_sum / sub_len - np.square(seq_mu)
    seq_sig_sq[seq_sig_sq < 0] = 0
    seq_sig = np.sqrt(seq_sig_sq)
    return seq_freq, seq_mu, seq_sig


def mass(que_info, seq_info, dim_idx):
    que = que_info['que'][::-1]
    que_mu = que_info['que_mu']
    que_sig = que_info['que_sig']

    seq_mu = seq_info['seq_mu'][dim_idx]
    seq_sig = seq_info['seq_sig'][dim_idx]
    seq_freq = seq_info['seq_freq'][dim_idx]
    sub_len = seq_info['sub_len']
    freq_len = seq_info['freq_len']
    n_sub = seq_info['n_sub']

    que_pad = np.zeros(freq_len)
    que_pad[:sub_len] = que
    que_freq = np.fft.fft(que_pad)
    product = np.real(np.fft.ifft(seq_freq * que_freq))
    product_ = product[sub_len - 1:sub_len - 1 + n_sub]

    sigma_x_eff = max(float(que_sig), EPS)
    sigma_y_eff = np.maximum(seq_sig, EPS)
    dist_profile = (
        (product_ - sub_len * seq_mu * que_mu) /
        (sub_len * sigma_y_eff * sigma_x_eff)
    )
    return np.real(dist_profile), product_
