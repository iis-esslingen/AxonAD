import time

import numpy as np
import torch

from .find_knn import find_knn_0, find_knn_1, find_knn_2
from .mstomp import _preprocess_seq, mstomp as cpu_mstomp
from .util import EPS, apply_exclude


VALID_PRECISIONS = ('float64', 'float32')
VALID_EXECUTIONS = ('auto', 'gpu_pipeline', 'cpu_reference')
EXACT_PRODUCT_MIN_CHUNK_ROWS = 32
EXACT_PRODUCT_MIN_TARGET_BYTES = 64 * 1024 * 1024
EXACT_PRODUCT_MAX_TARGET_BYTES = 256 * 1024 * 1024


def available_backends():
    available = bool(torch.cuda.is_available())
    details = {
        'name': 'torch',
        'available': available,
        'version': torch.__version__,
        'cuda_available': available,
    }
    if available:
        details['device_name'] = torch.cuda.get_device_name(0)
    else:
        details['reason'] = 'torch.cuda.is_available() is false'
    return {'torch': details}


def normalize_precision(precision):
    precision_norm = str(precision).strip().lower()
    if precision_norm not in VALID_PRECISIONS:
        raise ValueError(
            f'Unsupported precision={precision!r}; '
            f'expected one of {VALID_PRECISIONS}.')
    return precision_norm


def validate_execution(execution):
    execution_norm = str(execution).strip().lower()
    if execution_norm not in VALID_EXECUTIONS:
        raise ValueError(
            f'Unsupported execution={execution!r}; '
            f'expected one of {VALID_EXECUTIONS}.')
    return execution_norm


def _torch_dtype(precision):
    if normalize_precision(precision) == 'float64':
        return torch.float64
    return torch.float32


def _resolve_device_name(device):
    if device is None:
        return 'cuda'
    return str(device)


def _parse_sorting_place(sorting_place):
    sorting_place_norm = str(sorting_place).strip().lower()
    if sorting_place_norm in {'pre', 'post', 'pre-max', 'post-max'}:
        return sorting_place_norm, None
    if sorting_place_norm.startswith('pre-') or sorting_place_norm.startswith('post-'):
        head, tail = sorting_place_norm.split('-', 1)
        if tail.isdigit():
            return head, int(tail)
    raise ValueError(f'Unsupported sorting_place={sorting_place!r}.')


def _output_dim(n_dim, sorting_place):
    _, kth = _parse_sorting_place(sorting_place)
    if kth is not None:
        return 1
    return int(n_dim)


def _infer_selfjoin(seq_0, seq_1):
    if seq_0 is seq_1:
        return True
    seq_0_arr = np.asarray(seq_0)
    seq_1_arr = np.asarray(seq_1)
    return (
        seq_0_arr.shape == seq_1_arr.shape and
        np.array_equal(seq_0_arr, seq_1_arr, equal_nan=True)
    )


def _build_preprocessed_cpu(seq, sub_len, flat_mode):
    seq_arr = np.asarray(seq, dtype=float)
    if seq_arr.ndim != 2:
        raise ValueError(f'Expected a 2D array, got shape {seq_arr.shape}.')
    info = _preprocess_seq(
        np.array(seq_arr, copy=True),
        int(sub_len),
        int(seq_arr.shape[0] * 2),
        flat_mode=flat_mode,
    )
    return {
        'seq': np.array(info['seq'], copy=True),
        'sub_len': int(info['sub_len']),
        'seq_len': int(info['seq_len']),
        'n_sub': int(info['n_sub']),
        'n_dim': int(info['n_dim']),
        'skip_loc': np.array(info['skip_loc'], copy=True),
        'seq_mu': np.stack(info['seq_mu'], axis=1),
        'seq_sig': np.stack(info['seq_sig'], axis=1),
        'ref_info': info,
    }


def _build_state_from_cpu(seq, sub_len, flat_mode, device, dtype):
    info = _build_preprocessed_cpu(seq, sub_len, flat_mode=flat_mode)
    return {
        'seq': info['seq'],
        'seq_t': torch.as_tensor(info['seq'], device=device, dtype=dtype),
        'seq_mu': info['seq_mu'],
        'seq_mu_t': torch.as_tensor(info['seq_mu'], device=device, dtype=dtype),
        'seq_sig': info['seq_sig'],
        'seq_sig_t': torch.as_tensor(info['seq_sig'], device=device, dtype=dtype),
        'skip_loc': info['skip_loc'],
        'skip_loc_t': torch.as_tensor(info['skip_loc'], device=device, dtype=torch.bool),
        'sub_len': info['sub_len'],
        'seq_len': info['seq_len'],
        'n_sub': info['n_sub'],
        'n_dim': info['n_dim'],
        'ref_info': info['ref_info'],
    }


def _gpu_find_knn_0_torch_impl(dist_t, n_neighbor, exclude_len):
    n_sub, n_dim = dist_t.shape
    if int(n_neighbor) == 1:
        best_val_t, best_idx_t = torch.max(dist_t, dim=0)
        values_t = best_val_t.unsqueeze(1)
        indices_t = best_idx_t.unsqueeze(1)
        invalid_t = ~torch.isfinite(values_t)
        indices_t = torch.where(invalid_t, -torch.ones_like(indices_t), indices_t)
        return values_t, indices_t

    work_t = dist_t.clone()
    values_t = torch.full(
        (n_dim, int(n_neighbor)),
        -torch.inf,
        dtype=work_t.dtype,
        device=work_t.device,
    )
    indices_t = torch.zeros(
        (n_dim, int(n_neighbor)),
        dtype=torch.long,
        device=work_t.device,
    )
    row_ids_t = None
    if int(exclude_len) > 0:
        row_ids_t = torch.arange(n_sub, device=work_t.device).unsqueeze(1)

    for neighbor_idx in range(int(n_neighbor)):
        best_val_t, best_idx_t = torch.max(work_t, dim=0)
        values_t[:, neighbor_idx] = best_val_t
        indices_t[:, neighbor_idx] = best_idx_t
        if int(exclude_len) > 0:
            exclude_start_t = torch.clamp(best_idx_t - int(exclude_len), min=0)
            exclude_end_t = torch.clamp(best_idx_t + int(exclude_len), max=n_sub)
            exclude_mask_t = (
                (row_ids_t >= exclude_start_t.unsqueeze(0)) &
                (row_ids_t < exclude_end_t.unsqueeze(0))
            )
            work_t = work_t.masked_fill(exclude_mask_t, -torch.inf)

    invalid_t = ~torch.isfinite(values_t)
    indices_t = torch.where(invalid_t, -torch.ones_like(indices_t), indices_t)
    return values_t, indices_t


def _gpu_find_knn_1d_torch_impl(dist_t, n_neighbor, exclude_len):
    work_t = dist_t[:, 0]
    if int(n_neighbor) == 1:
        best_val_t, best_idx_t = torch.max(work_t, dim=0)
        values_t = best_val_t.reshape(1, 1)
        indices_t = best_idx_t.reshape(1, 1)
        invalid_t = ~torch.isfinite(values_t)
        indices_t = torch.where(invalid_t, -torch.ones_like(indices_t), indices_t)
        return values_t, indices_t

    values_t = torch.full(
        (1, int(n_neighbor)),
        -torch.inf,
        dtype=work_t.dtype,
        device=work_t.device,
    )
    indices_t = torch.zeros(
        (1, int(n_neighbor)),
        dtype=torch.long,
        device=work_t.device,
    )
    for neighbor_idx in range(int(n_neighbor)):
        best_val_t, best_idx_t = torch.max(work_t, dim=0)
        values_t[0, neighbor_idx] = best_val_t
        indices_t[0, neighbor_idx] = best_idx_t
        if int(exclude_len) > 0 and torch.isfinite(best_val_t):
            start = max(0, int(best_idx_t.item()) - int(exclude_len))
            end = min(int(work_t.shape[0]), int(best_idx_t.item()) + int(exclude_len))
            work_t[start:end] = -torch.inf

    invalid_t = ~torch.isfinite(values_t)
    indices_t = torch.where(invalid_t, -torch.ones_like(indices_t), indices_t)
    return values_t, indices_t


def _apply_post_sorting_torch(mpval_t, mpidx_t, mode, sorting_place):
    sorting_kind, kth = _parse_sorting_place(sorting_place)

    if sorting_kind == 'post' and kth is None:
        if mode == 'discord':
            work_t = torch.where(
                torch.isfinite(mpval_t),
                mpval_t,
                torch.full_like(mpval_t, torch.inf),
            )
            sorted_vals_t, order_t = torch.sort(work_t, dim=0, descending=False)
        else:
            work_t = torch.where(
                torch.isfinite(mpval_t),
                mpval_t,
                torch.full_like(mpval_t, -torch.inf),
            )
            sorted_vals_t, order_t = torch.sort(work_t, dim=0, descending=True)
        sorted_idx_t = torch.gather(mpidx_t, 0, order_t)
        sorted_vals_t = torch.where(
            torch.isfinite(sorted_vals_t),
            sorted_vals_t,
            torch.full_like(sorted_vals_t, -torch.inf),
        )
        if mode == 'motif':
            sorted_vals_t = torch.cumsum(sorted_vals_t, dim=0)
        return sorted_vals_t, sorted_idx_t

    if sorting_kind == 'post-max':
        if mode == 'discord':
            work_t = torch.where(
                torch.isfinite(mpval_t),
                mpval_t,
                torch.full_like(mpval_t, torch.inf),
            )
            best_idx_t = torch.argmin(work_t, dim=0, keepdim=True)
        else:
            work_t = torch.where(
                torch.isfinite(mpval_t),
                mpval_t,
                torch.full_like(mpval_t, -torch.inf),
            )
            best_idx_t = torch.argmax(work_t, dim=0, keepdim=True)
        best_val_t = torch.gather(mpval_t, 0, best_idx_t)
        best_mpidx_t = torch.gather(mpidx_t, 0, best_idx_t)
        best_val_t = torch.where(
            torch.isfinite(best_val_t),
            best_val_t,
            torch.full_like(best_val_t, -torch.inf),
        )
        return best_val_t, best_mpidx_t

    if sorting_kind == 'post' and kth is not None:
        if mode == 'discord':
            work_t = torch.where(
                torch.isfinite(mpval_t),
                mpval_t,
                torch.full_like(mpval_t, torch.inf),
            )
            sorted_vals_t, order_t = torch.sort(work_t, dim=0, descending=False)
            sorted_idx_t = torch.gather(mpidx_t, 0, order_t)
            out_val_t = sorted_vals_t[kth:kth + 1]
            out_idx_t = sorted_idx_t[kth:kth + 1]
            out_val_t = torch.where(
                torch.isfinite(out_val_t),
                out_val_t,
                torch.full_like(out_val_t, -torch.inf),
            )
            return out_val_t, out_idx_t

        work_t = torch.where(
            torch.isfinite(mpval_t),
            mpval_t,
            torch.full_like(mpval_t, -torch.inf),
        )
        sorted_vals_t, order_t = torch.sort(work_t, dim=0, descending=True)
        sorted_idx_t = torch.gather(mpidx_t, 0, order_t)
        out_val_t = torch.sum(sorted_vals_t[:kth + 1], dim=0, keepdim=True)
        out_idx_t = sorted_idx_t[kth:kth + 1]
        out_val_t = torch.where(
            torch.isfinite(out_val_t),
            out_val_t,
            torch.full_like(out_val_t, -torch.inf),
        )
        return out_val_t, out_idx_t

    return mpval_t.clone(), mpidx_t.clone()


def _auto_execution(seq_0, seq_1, sub_len, n_neighbor=1):
    seq_0_arr = np.asarray(seq_0)
    seq_1_arr = np.asarray(seq_1)
    n_sub_0 = max(1, int(seq_0_arr.shape[0]) - int(sub_len) + 1)
    n_sub_1 = max(1, int(seq_1_arr.shape[0]) - int(sub_len) + 1)
    n_dim = int(seq_0_arr.shape[1])
    n_sub_max = max(n_sub_0, n_sub_1)
    rolling_score = n_sub_0 * max(1, n_dim) * max(1, int(sub_len))
    total_score = n_sub_0 * n_sub_1 * max(1, n_dim)
    n_neighbor = int(n_neighbor)

    # These thresholds are tuned from measured v2 crossover points on the A10G.
    # They deliberately bias toward CPU on small or skinny workloads and toward
    # GPU once the row-wise fixed overhead is sufficiently amortized.
    if n_dim == 1:
        if n_neighbor <= 1:
            if n_sub_max >= 60_000 or rolling_score >= 8_000_000:
                return 'gpu_pipeline'
            return 'cpu_reference'
        if n_sub_max >= 120_000 and total_score >= 6_000_000_000:
            return 'gpu_pipeline'
        return 'cpu_reference'

    if n_dim <= 4:
        if n_neighbor <= 1:
            if rolling_score >= 4_000_000 or total_score >= 120_000_000:
                return 'gpu_pipeline'
            return 'cpu_reference'
        if rolling_score >= 16_000_000 or total_score >= 500_000_000:
            return 'gpu_pipeline'
        return 'cpu_reference'

    if n_dim <= 8 and n_neighbor > 1:
        if rolling_score >= 8_000_000 or total_score >= 240_000_000:
            return 'gpu_pipeline'
        return 'cpu_reference'

    if rolling_score >= 4_000_000 or total_score >= 120_000_000:
        return 'gpu_pipeline'
    return 'cpu_reference'


def _sliding_products_cpu(target_seq, query):
    sub_len = int(query.shape[0])
    out = np.zeros((target_seq.shape[0] - sub_len + 1, target_seq.shape[1]), dtype=float)
    for dim_idx in range(target_seq.shape[1]):
        windows = np.lib.stride_tricks.sliding_window_view(target_seq[:, dim_idx], sub_len)
        out[:, dim_idx] = windows @ query[:, dim_idx]
    return out


def _resolve_exact_product_chunk_rows(target_seq_t, query_t):
    n_sub = max(1, int(target_seq_t.shape[0]) - int(query_t.shape[0]) + 1)
    per_row_bytes = max(
        1,
        int(query_t.shape[0]) *
        max(1, int(target_seq_t.shape[1])) *
        max(1, int(target_seq_t.element_size())) * 2,
    )
    target_bytes = EXACT_PRODUCT_MAX_TARGET_BYTES
    if target_seq_t.is_cuda:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(target_seq_t.device)
            target_bytes = min(
                EXACT_PRODUCT_MAX_TARGET_BYTES,
                max(EXACT_PRODUCT_MIN_TARGET_BYTES, int(free_bytes) // 16),
            )
        except Exception:
            target_bytes = EXACT_PRODUCT_MAX_TARGET_BYTES
    chunk_rows = max(EXACT_PRODUCT_MIN_CHUNK_ROWS, int(target_bytes) // per_row_bytes)
    return max(1, min(n_sub, int(chunk_rows)))


def _exact_product_matrix_gpu(query_info, target_info, query_idx):
    sub_len = int(query_info['sub_len'])
    query_idx = int(query_idx)
    query_t = query_info['seq_t'][query_idx:query_idx + sub_len]
    try:
        return _sliding_products_gpu_impl(target_info['seq_t'], query_t)
    except torch.OutOfMemoryError:
        if target_info['seq_t'].is_cuda:
            torch.cuda.empty_cache()
        query = query_info['seq'][query_idx:query_idx + sub_len]
        product = _sliding_products_cpu(target_info['seq'], query)
        return torch.as_tensor(
            product,
            device=target_info['seq_t'].device,
            dtype=target_info['seq_t'].dtype,
        )


def _resolve_reseed_period(n_profile, sub_len, n_dim, n_neighbor, reseed_period):
    n_profile = max(1, int(n_profile))
    if reseed_period is not None:
        return max(1, min(n_profile, int(reseed_period)))

    base = max(64, min(1024, int(sub_len) // 2))
    if int(n_neighbor) > 1:
        base = max(64, base // 2)
    if int(n_dim) == 1:
        base = min(2048, base * 2)

    # Large multivariate workloads do better with much sparser anchor reseeds.
    # The smaller default is still useful on short profiles where drift can
    # dominate, but it is far too aggressive once the profile gets large.
    if int(n_dim) > 1:
        if n_profile >= 20_000:
            base *= 4
        if n_profile >= 100_000:
            base *= 2
        if n_profile >= 250_000:
            base *= 2
        if n_profile >= 500_000:
            base *= 2
        if int(n_dim) >= 16:
            base = max(base, 256)
        if int(n_dim) >= 64:
            base = max(base, 512)
        if int(n_dim) >= 128:
            base = max(base, 1024)
        base = min(4096, base)

    return max(1, min(n_profile, int(base)))


def _resolve_anchor_starts(n_profile, reseed_period):
    n_profile = int(n_profile)
    starts = {0}
    if reseed_period is not None:
        period = max(1, int(reseed_period))
        starts.update(range(0, n_profile, period))
    return starts


def _sliding_products_gpu_impl(target_seq_t, query_t):
    sub_len = int(query_t.shape[0])
    n_sub = int(target_seq_t.shape[0]) - sub_len + 1
    if n_sub <= 0:
        return torch.empty((0, int(target_seq_t.shape[1])), dtype=target_seq_t.dtype, device=target_seq_t.device)

    chunk_rows = _resolve_exact_product_chunk_rows(target_seq_t, query_t)
    while True:
        try:
            out_t = torch.empty(
                (n_sub, int(target_seq_t.shape[1])),
                dtype=target_seq_t.dtype,
                device=target_seq_t.device,
            )
            for start in range(0, n_sub, chunk_rows):
                stop = min(n_sub, start + chunk_rows)
                seq_chunk_t = target_seq_t[start:stop + sub_len - 1]
                windows_t = seq_chunk_t.unfold(0, sub_len, 1)
                out_t[start:stop] = torch.einsum('ndm,md->nd', windows_t, query_t)
            return out_t
        except torch.OutOfMemoryError:
            if (not target_seq_t.is_cuda) or chunk_rows <= EXACT_PRODUCT_MIN_CHUNK_ROWS:
                raise
            torch.cuda.empty_cache()
            next_chunk_rows = max(EXACT_PRODUCT_MIN_CHUNK_ROWS, chunk_rows // 2)
            if next_chunk_rows == chunk_rows:
                raise
            chunk_rows = next_chunk_rows


def _masked_profile_from_product_impl(
        product_t, target_mu_t, target_sig_t, query_mu_t, query_sig_t,
        query_skip_t, target_skip_t, invalid_fill, sub_len):
    sigma_x = torch.clamp(query_sig_t, min=float(EPS)).unsqueeze(0)
    sigma_y = torch.clamp(target_sig_t, min=float(EPS))
    numer = product_t - float(sub_len) * target_mu_t * query_mu_t.unsqueeze(0)
    dist_t = numer / (float(sub_len) * sigma_y * sigma_x)
    dist_t = dist_t.masked_fill(query_skip_t.unsqueeze(0), invalid_fill)
    dist_t = dist_t.masked_fill(target_skip_t, invalid_fill)
    return dist_t


def _apply_pre_sorting_torch(dist_t, mode, sorting_place):
    sorting_kind, kth = _parse_sorting_place(sorting_place)
    if sorting_kind == 'pre' and kth is None:
        if mode == 'motif':
            sorted_t = torch.sort(dist_t, dim=1, descending=True).values
            return torch.cumsum(sorted_t, dim=1)
        return torch.sort(dist_t, dim=1, descending=False).values

    if sorting_kind == 'pre-max':
        if mode == 'motif':
            return torch.max(dist_t, dim=1, keepdim=True).values
        return torch.min(dist_t, dim=1, keepdim=True).values

    if sorting_kind == 'pre' and kth is not None:
        if mode == 'motif':
            sorted_t = torch.sort(dist_t, dim=1, descending=True).values
            return torch.sum(sorted_t[:, :kth + 1], dim=1, keepdim=True)
        sorted_t = torch.sort(dist_t, dim=1, descending=False).values
        return sorted_t[:, kth:kth + 1]

    return dist_t


def _apply_selfjoin_exclude_torch(dist_t, idx, exclude_len):
    if exclude_len <= 0:
        return dist_t
    start = max(0, int(idx) - int(exclude_len))
    end = min(dist_t.shape[0], int(idx) + int(exclude_len))
    dist_t[start:end] = -torch.inf
    return dist_t


def _resolve_find_knn_impl(find_knn_variant):
    if int(find_knn_variant) == 0:
        return find_knn_0
    if int(find_knn_variant) == 1:
        return find_knn_1
    return find_knn_2


def _apply_pre_sorting_cpu(dist_profile, mode, sorting_place):
    dist_profile = np.array(dist_profile, copy=True)
    if sorting_place == 'pre':
        if mode == 'motif':
            dist_profile = -np.sort(-dist_profile, axis=1)
            dist_profile = np.cumsum(dist_profile, axis=1)
        else:
            dist_profile = np.sort(dist_profile, axis=1)
    elif sorting_place == 'pre-max':
        if mode == 'motif':
            order = np.argmin(-dist_profile, axis=1, keepdims=True)
        else:
            order = np.argmin(dist_profile, axis=1, keepdims=True)
        dist_profile[:, 0] = np.take_along_axis(dist_profile, order, axis=1)[:, 0]
    elif sorting_place.startswith('pre-') and sorting_place.split('-', 1)[1].isdigit():
        kth = int(sorting_place.split('-', 1)[1])
        if mode == 'motif':
            order = np.argpartition(-dist_profile, kth, axis=1)
            dist_profile_0 = np.take_along_axis(dist_profile, order[:, :kth + 1], axis=1)
            dist_profile[:, 0] = np.sum(dist_profile_0, axis=1)
        else:
            order = np.argpartition(dist_profile, kth, axis=1)
            dist_profile[:, 0] = np.take_along_axis(
                dist_profile, order[:, kth:kth + 1], axis=1)[:, 0]
    return dist_profile


def _apply_post_sorting_cpu(mpval, mpidx, mode, sorting_place):
    mpval = np.array(mpval, copy=True)
    mpidx = np.array(mpidx, copy=True)

    if sorting_place == 'post':
        for i in range(mpval.shape[1]):
            mpval_i = mpval[:, i]
            mpidx_i = mpidx[:, i]
            if mode == 'discord':
                mpval_i[~np.isfinite(mpval_i)] = np.inf
                order = np.argsort(mpval_i)
            else:
                mpval_i[~np.isfinite(mpval_i)] = -np.inf
                order = np.argsort(-mpval_i)

            mpval_i = np.take_along_axis(mpval_i, order, axis=0)
            mpidx_i = np.take_along_axis(mpidx_i, order, axis=0)
            mpval_i[~np.isfinite(mpval_i)] = -np.inf
            if mode == 'motif':
                mpval_i = np.nancumsum(mpval_i, axis=0)
            mpval[:, i] = mpval_i
            mpidx[:, i] = mpidx_i
    elif sorting_place == 'post-max':
        for i in range(mpval.shape[1]):
            mpval_i = mpval[:, i]
            mpidx_i = mpidx[:, i]
            if mode == 'discord':
                mpval_i[~np.isfinite(mpval_i)] = np.inf
                order = np.argmin(mpval_i, keepdims=True)
            else:
                mpval_i[~np.isfinite(mpval_i)] = -np.inf
                order = np.argmin(-mpval_i, keepdims=True)

            mpval_i = np.take_along_axis(mpval_i, order, axis=0)
            mpidx_i = np.take_along_axis(mpidx_i, order, axis=0)
            mpval_i[~np.isfinite(mpval_i)] = -np.inf
            mpval[0, i] = mpval_i[0]
            mpidx[0, i] = mpidx_i[0]
    elif sorting_place.startswith('post-') and sorting_place.split('-', 1)[1].isdigit():
        kth = int(sorting_place.split('-', 1)[1])
        for i in range(mpval.shape[1]):
            mpval_i = mpval[:, i]
            mpidx_i = mpidx[:, i]
            if mode == 'discord':
                mpval_i[~np.isfinite(mpval_i)] = np.inf
                order = np.argpartition(mpval_i, kth)
                mpval_i = np.take_along_axis(mpval_i, order[kth:kth + 1], axis=0)
            else:
                mpval_i[~np.isfinite(mpval_i)] = -np.inf
                order = np.argpartition(-mpval_i, kth)
                mpval_i = np.take_along_axis(mpval_i, order[:kth + 1], axis=0)
                mpval_i = np.sum(mpval_i, keepdims=True)

            mpidx_i = np.take_along_axis(mpidx_i, order[kth:kth + 1], axis=0)
            mpval_i[~np.isfinite(mpval_i)] = -np.inf
            mpval[0, i] = mpval_i[0]
            mpidx[0, i] = mpidx_i[0]

    if '-' in sorting_place:
        mpval = mpval[0:1, :]
        mpidx = mpidx[0:1, :]

    return mpval, mpidx


def _select_row_profile_cpu(dist_t, idx, n_sub, n_neighbor, exclude_len,
                            mode, sorting_place, is_selfjoin, find_knn_variant):
    dist_profile = dist_t.cpu().numpy().astype(float, copy=False)
    dist_profile = _apply_pre_sorting_cpu(
        dist_profile,
        mode=mode,
        sorting_place=sorting_place,
    )
    dist_profile[~np.isfinite(dist_profile)] = -np.inf
    if is_selfjoin:
        dist_profile = apply_exclude(dist_profile, idx, exclude_len, n_sub)
    find_knn = _resolve_find_knn_impl(find_knn_variant)
    row_val, row_idx = find_knn(dist_profile, n_neighbor, exclude_len)
    row_val, row_idx = _apply_post_sorting_cpu(
        row_val,
        row_idx,
        mode=mode,
        sorting_place=sorting_place,
    )
    return row_val, row_idx


def _resolve_row_selection_mode(row_selection):
    row_selection_norm = str(row_selection).strip().lower()
    if row_selection_norm not in {'gpu', 'cpu'}:
        raise ValueError(f'Unsupported row_selection={row_selection!r}.')
    return row_selection_norm


def _is_degenerate_mpval(mpval):
    mpval_arr = np.asarray(mpval, dtype=float)
    if mpval_arr.ndim != 3 or mpval_arr.shape[0] == 0:
        return False
    core = mpval_arr[:, -1, -1]
    finite = np.isfinite(core)
    if not np.any(finite):
        return True
    core_finite = core[finite]
    return float(np.max(core_finite) - np.min(core_finite)) <= 1e-12


def _select_row_profile_torch(dist_t, idx, n_sub, n_neighbor, exclude_len,
                              mode, sorting_place, is_selfjoin, find_knn_variant):
    del n_sub, find_knn_variant
    dist_t = _apply_pre_sorting_torch(dist_t, mode=mode, sorting_place=sorting_place)
    dist_t = torch.where(
        torch.isfinite(dist_t),
        dist_t,
        torch.full_like(dist_t, -torch.inf),
    )
    if is_selfjoin:
        dist_t = _apply_selfjoin_exclude_torch(dist_t, idx, exclude_len)
    if int(dist_t.shape[1]) == 1:
        row_val_t, row_idx_t = _gpu_find_knn_1d_torch_impl(
            dist_t,
            n_neighbor=n_neighbor,
            exclude_len=exclude_len,
        )
    else:
        row_val_t, row_idx_t = _gpu_find_knn_0_torch_impl(
            dist_t,
            n_neighbor=n_neighbor,
            exclude_len=exclude_len,
        )
    row_val_t, row_idx_t = _apply_post_sorting_torch(
        row_val_t,
        row_idx_t,
        mode=mode,
        sorting_place=sorting_place,
    )
    return row_val_t, row_idx_t


def _log_progress(verbose, label, step_pct, next_pct, idx, total, tic, device):
    if not verbose:
        return next_pct
    current_pct = (int(idx + 1) * 100) // max(int(total), 1)
    if current_pct < next_pct and current_pct != 100:
        return next_pct
    torch.cuda.synchronize(device)
    elapsed = max(0.0, time.time() - tic)
    time_left = (elapsed / max(current_pct, 1)) * (100 - current_pct)
    prefix = f'[{label}] ' if label else ''
    print(f'{prefix}gpu {current_pct: 4d}% ({idx + 1}/{total}), {time_left:0.4f} sec left')
    while next_pct <= current_pct:
        next_pct += step_pct
    return next_pct


def _mstomp_torch(seq_0, seq_1, sub_len, n_neighbor=1,
                  mode='motif', sorting_place='pre',
                  exclude_frac=0.5, find_knn_variant=2,
                  n_job=1, verbose=False, flat_mode='invalid',
                  progress_label=None, progress_step_pct=10,
                  precision='float64', allow_tf32=False,
                  device=None, reseed_period=None, row_selection='gpu'):
    if not torch.cuda.is_available():
        raise RuntimeError('Torch MSTOMP backend requires CUDA.')

    precision = normalize_precision(precision)
    device_name = _resolve_device_name(device)
    dev = torch.device(device_name)
    dtype = _torch_dtype(precision)
    tf32_prev = bool(torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32 and precision == 'float32')
    inference_context = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad

    try:
        with inference_context():
            row_selection = _resolve_row_selection_mode(row_selection)
            seq_0_info = _build_state_from_cpu(seq_0, sub_len, flat_mode=flat_mode, device=dev, dtype=dtype)
            is_selfjoin = _infer_selfjoin(seq_0, seq_1)
            if is_selfjoin:
                seq_1_info = seq_0_info
            else:
                seq_1_info = _build_state_from_cpu(seq_1, sub_len, flat_mode=flat_mode, device=dev, dtype=dtype)

            masked_profile_from_product = _masked_profile_from_product_impl

            exclude_len = int(float(exclude_frac) * int(sub_len))
            n_profile = int(seq_0_info['n_sub'])
            out_dim = _output_dim(seq_0_info['n_dim'], sorting_place)
            step_pct = max(1, int(progress_step_pct))
            next_pct = step_pct
            tic = time.time()
            auto_reseed_period = _resolve_reseed_period(
                n_profile=n_profile,
                sub_len=sub_len,
                n_dim=seq_0_info['n_dim'],
                n_neighbor=n_neighbor,
                reseed_period=reseed_period,
            )
            anchor_starts = _resolve_anchor_starts(
                n_profile=n_profile,
                reseed_period=auto_reseed_period,
            )

            if verbose:
                print('preprocessing input time series ... done!')
                prefix = f'[{progress_label}] ' if progress_label else ''
                print(
                    f'{prefix}gpu backend uses anchor reseeds every '
                    f'{auto_reseed_period} rows on one CUDA device.')
                print('compute multidimensional matrix profile ... ')

            first_product_t = _exact_product_matrix_gpu(
                query_info=seq_1_info,
                target_info=seq_0_info,
                query_idx=0,
            )

            mpval_t = torch.full(
                (n_profile, out_dim, int(n_neighbor)),
                -torch.inf,
                dtype=dtype,
                device=dev,
            )
            mpidx_t = -torch.ones(
                (n_profile, out_dim, int(n_neighbor)),
                dtype=torch.long,
                device=dev,
            )
            last_product_t = None
            invalid_fill = -torch.inf if mode == 'motif' else torch.inf

            for idx in range(n_profile):
                if idx in anchor_starts:
                    last_product_t = _exact_product_matrix_gpu(
                        query_info=seq_0_info,
                        target_info=seq_1_info,
                        query_idx=idx,
                    )
                else:
                    drop_t = seq_0_info['seq_t'][idx - 1]
                    add_t = seq_0_info['seq_t'][idx + sub_len - 1]
                    next_product_t = torch.empty_like(last_product_t)
                    next_product_t[1:] = (
                        last_product_t[:-1] -
                        seq_1_info['seq_t'][:-sub_len] * drop_t.unsqueeze(0) +
                        seq_1_info['seq_t'][sub_len:] * add_t.unsqueeze(0)
                    )
                    next_product_t[0] = first_product_t[idx]
                    last_product_t = next_product_t

                dist_t = masked_profile_from_product(
                    last_product_t,
                    seq_1_info['seq_mu_t'],
                    seq_1_info['seq_sig_t'],
                    seq_0_info['seq_mu_t'][idx],
                    seq_0_info['seq_sig_t'][idx],
                    seq_0_info['skip_loc_t'][idx],
                    seq_1_info['skip_loc_t'],
                        invalid_fill,
                        sub_len,
                    )

                if row_selection == 'cpu':
                    row_val, row_idx = _select_row_profile_cpu(
                        dist_t=dist_t,
                        idx=idx,
                        n_sub=seq_1_info['n_sub'],
                        n_neighbor=n_neighbor,
                        exclude_len=exclude_len,
                        mode=mode,
                        sorting_place=sorting_place,
                        is_selfjoin=is_selfjoin,
                        find_knn_variant=find_knn_variant,
                    )
                    mpval_t[idx] = torch.as_tensor(
                        row_val[:out_dim],
                        dtype=dtype,
                        device=dev,
                    )
                    mpidx_t[idx] = torch.as_tensor(
                        row_idx[:out_dim],
                        dtype=torch.long,
                        device=dev,
                    )
                else:
                    row_val_t, row_idx_t = _select_row_profile_torch(
                        dist_t=dist_t,
                        idx=idx,
                        n_sub=seq_1_info['n_sub'],
                        n_neighbor=n_neighbor,
                        exclude_len=exclude_len,
                        mode=mode,
                        sorting_place=sorting_place,
                        is_selfjoin=is_selfjoin,
                        find_knn_variant=find_knn_variant,
                    )
                    mpval_t[idx] = row_val_t[:out_dim]
                    mpidx_t[idx] = row_idx_t[:out_dim]
                next_pct = _log_progress(
                    verbose=verbose,
                    label=progress_label,
                    step_pct=step_pct,
                    next_pct=next_pct,
                    idx=idx,
                    total=n_profile,
                    tic=tic,
                    device=dev,
                )

            invalid_mask_t = ~torch.isfinite(mpval_t)
            mpidx_t = torch.where(invalid_mask_t, -torch.ones_like(mpidx_t), mpidx_t)
            mpval = mpval_t.cpu().numpy().astype(float, copy=False)
            mpidx = mpidx_t.cpu().numpy().astype(int, copy=False)
            mpval[~np.isfinite(mpval)] = np.nan
            return mpval, mpidx
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_prev


def mstomp(seq_0, seq_1, sub_len, n_neighbor=1,
           mode='motif', sorting_place='pre',
           exclude_frac=0.5, find_knn_variant=2,
           n_job=1, verbose=False, flat_mode='invalid',
           progress_label=None, progress_step_pct=10,
           precision='float64', allow_tf32=False,
           device=None, execution='auto', reseed_period=None):
    precision = normalize_precision(precision)
    execution = validate_execution(execution)

    if execution == 'auto':
        if not torch.cuda.is_available():
            execution = 'cpu_reference'
        else:
            execution = _auto_execution(seq_0, seq_1, sub_len, n_neighbor=n_neighbor)

    if execution == 'cpu_reference':
        return cpu_mstomp(
            seq_0,
            seq_1,
            sub_len,
            n_neighbor=n_neighbor,
            mode=mode,
            sorting_place=sorting_place,
            exclude_frac=exclude_frac,
            find_knn_variant=find_knn_variant,
            n_job=n_job,
            verbose=verbose,
            flat_mode=flat_mode,
        )

    mpval, mpidx = _mstomp_torch(
        seq_0=seq_0,
        seq_1=seq_1,
        sub_len=sub_len,
        n_neighbor=n_neighbor,
        mode=mode,
        sorting_place=sorting_place,
        exclude_frac=exclude_frac,
        find_knn_variant=find_knn_variant,
        n_job=n_job,
        verbose=verbose,
        flat_mode=flat_mode,
        progress_label=progress_label,
        progress_step_pct=progress_step_pct,
        precision=precision,
        allow_tf32=allow_tf32,
        device=device,
        reseed_period=reseed_period,
        row_selection='gpu',
    )
    if _is_degenerate_mpval(mpval):
        if verbose:
            prefix = f'[{progress_label}] ' if progress_label else ''
            print(f'{prefix}gpu output degenerated; retrying with safe row selection and reseed_period=1')
        mpval, mpidx = _mstomp_torch(
            seq_0=seq_0,
            seq_1=seq_1,
            sub_len=sub_len,
            n_neighbor=n_neighbor,
            mode=mode,
            sorting_place=sorting_place,
            exclude_frac=exclude_frac,
            find_knn_variant=find_knn_variant,
            n_job=n_job,
            verbose=verbose,
            flat_mode=flat_mode,
            progress_label=progress_label,
            progress_step_pct=progress_step_pct,
            precision=precision,
            allow_tf32=allow_tf32,
            device=device,
            reseed_period=1,
            row_selection='cpu',
        )
    return mpval, mpidx
