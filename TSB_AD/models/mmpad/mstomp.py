import multiprocessing
import time
from multiprocessing import Pool

import numpy as np

from .find_knn import find_knn_0, find_knn_1, find_knn_2
from .util import EPS, apply_exclude, mass, mass_pre


def mstomp(seq_0, seq_1, sub_len, n_neighbor=1,
           mode='motif', sorting_place='pre',
           exclude_frac=0.5, find_knn_variant=2,
           n_job=1, verbose=False, flat_mode='invalid'):
    assert mode in ['motif', 'discord']
    assert flat_mode in ['invalid', 'eps']
    if n_job == -1:
        n_job = multiprocessing.cpu_count()
        if verbose:
            print(f'the number of job is automatically set to {n_job}')
    if verbose:
        print('preprocessing input time series ... ', end='')
        tic = time.time()
    n_dim = seq_0.shape[1]
    assert n_dim == seq_1.shape[1]
    freq_len = seq_0.shape[0] + seq_1.shape[0]
    seq_0_info = _preprocess_seq(seq_0, sub_len, freq_len, flat_mode=flat_mode)
    if seq_0.shape[0] == seq_1.shape[0] and np.allclose(seq_0, seq_1):
        seq_0_info['is_selfjoin'] = True
        seq_0_info['n_neighbor'] = n_neighbor
        seq_0_info['mode'] = mode
        seq_0_info['sorting_place'] = sorting_place
        seq_0_info['exclude_frac'] = exclude_frac
        seq_0_info['find_knn_variant'] = find_knn_variant
        seq_1_info = seq_0_info
    else:
        seq_1_info = _preprocess_seq(seq_1, sub_len, freq_len, flat_mode=flat_mode)
        seq_0_info['is_selfjoin'] = False
        seq_1_info['is_selfjoin'] = False
        seq_0_info['n_neighbor'] = n_neighbor
        seq_1_info['n_neighbor'] = n_neighbor
        seq_0_info['mode'] = mode
        seq_1_info['mode'] = mode
        seq_0_info['sorting_place'] = sorting_place
        seq_1_info['sorting_place'] = sorting_place
        seq_0_info['exclude_frac'] = exclude_frac
        seq_1_info['exclude_frac'] = exclude_frac
        seq_0_info['find_knn_variant'] = find_knn_variant
        seq_1_info['find_knn_variant'] = find_knn_variant

    first_product = []
    for i in range(n_dim):
        que_info = {
            'que': seq_1_info['seq'][:sub_len, i],
            'que_mu': seq_1_info['seq_mu'][i][0],
            'que_sig': seq_1_info['seq_sig'][i][0],
        }
        _, first_product_i = mass(que_info, seq_0_info, i)
        first_product.append(first_product_i)

    job_args = []
    n_sub_per_job = seq_0_info['n_sub'] // n_job
    for i in range(n_job):
        job_args.append([
            i, n_job, i * n_sub_per_job, (i + 1) * n_sub_per_job,
            seq_0_info, seq_1_info, first_product, verbose,
        ])
    job_args[-1][3] = seq_0_info['n_sub']
    if verbose:
        print(f'done! {time.time() - tic:0.4f} sec')

    if verbose:
        print('compute multidimensional matrix profile ... ')
        tic = time.time()
    if n_job == 1:
        results = _process_chunk(job_args[0])
        mpval = results['mpval']
        mpidx = results['mpidx']
    else:
        with Pool(n_job) as pool:
            results = pool.map(_process_chunk, job_args)
        mpval = np.concatenate([result['mpval'] for result in results], axis=0)
        mpidx = np.concatenate([result['mpidx'] for result in results], axis=0)

    if sorting_place == 'post':
        for i in range(n_neighbor):
            mpval_i = mpval[:, :, i]
            mpidx_i = mpidx[:, :, i]
            if mode == 'discord':
                mpval_i[~np.isfinite(mpval_i)] = np.inf
                order = np.argsort(mpval_i, axis=1)
            else:
                mpval_i[~np.isfinite(mpval_i)] = -np.inf
                order = np.argsort(-mpval_i, axis=1)

            mpval_i = np.take_along_axis(mpval_i, order, axis=1)
            mpidx_i = np.take_along_axis(mpidx_i, order, axis=1)
            mpval_i[~np.isfinite(mpval_i)] = -np.inf
            if mode == 'motif':
                mpval_i = np.nancumsum(mpval_i, axis=1)
            mpval[:, :, i] = mpval_i
            mpidx[:, :, i] = mpidx_i
    elif sorting_place == 'post-max':
        for i in range(n_neighbor):
            mpval_i = mpval[:, :, i]
            mpidx_i = mpidx[:, :, i]
            if mode == 'discord':
                mpval_i[~np.isfinite(mpval_i)] = np.inf
                order = np.argmin(mpval_i, axis=1, keepdims=True)
            else:
                mpval_i[~np.isfinite(mpval_i)] = -np.inf
                order = np.argmin(-mpval_i, axis=1, keepdims=True)

            mpval_i = np.take_along_axis(mpval_i, order, axis=1)
            mpidx_i = np.take_along_axis(mpidx_i, order, axis=1)
            mpval_i[~np.isfinite(mpval_i)] = -np.inf
            mpval[:, 0, i] = mpval_i[:, 0]
            mpidx[:, 0, i] = mpidx_i[:, 0]
    elif sorting_place.startswith('post-') and sorting_place.split('-', 1)[1].isdigit():
        kth = int(sorting_place.split('-', 1)[1])
        for i in range(n_neighbor):
            mpval_i = mpval[:, :, i]
            mpidx_i = mpidx[:, :, i]
            if mode == 'discord':
                mpval_i[~np.isfinite(mpval_i)] = np.inf
                order = np.argpartition(mpval_i, kth, axis=1)
                mpval_i = np.take_along_axis(mpval_i, order[:, kth:kth + 1], axis=1)
            else:
                mpval_i[~np.isfinite(mpval_i)] = -np.inf
                order = np.argpartition(-mpval_i, kth, axis=1)
                mpval_i = np.take_along_axis(mpval_i, order[:, :kth + 1], axis=1)
                mpval_i = np.sum(mpval_i, axis=1, keepdims=True)

            mpidx_i = np.take_along_axis(mpidx_i, order[:, kth:kth + 1], axis=1)
            mpval_i[~np.isfinite(mpval_i)] = -np.inf
            mpval[:, 0, i] = mpval_i[:, 0]
            mpidx[:, 0, i] = mpidx_i[:, 0]

    if '-' in sorting_place:
        mpval = mpval[:, 0:1, :]
        mpidx = mpidx[:, 0:1, :]

    if verbose:
        print(f'    done! {time.time() - tic:0.4f} sec')
    return mpval, mpidx


def _process_chunk(args):
    chunk_id, n_job, idx_start, idx_end, seq_0_info, seq_1_info, first_product, verbose = args
    if verbose:
        n_id_digit = 1 if chunk_id == 0 else int(np.log10(chunk_id))
        n_job_digit = 1 if n_job - 1 == 0 else int(np.log10(n_job - 1))
        spaces = ' ' * (n_job_digit - n_id_digit)

    seq = seq_1_info['seq']
    seq_mu = seq_1_info['seq_mu']
    seq_sig = seq_1_info['seq_sig']
    sub_len = seq_1_info['sub_len']
    n_sub = seq_1_info['n_sub']
    n_dim = seq_1_info['n_dim']
    is_selfjoin = seq_1_info['is_selfjoin']
    n_neighbor = seq_1_info['n_neighbor']
    mode = seq_1_info['mode']
    sorting_place = seq_1_info['sorting_place']
    exclude_frac = seq_1_info['exclude_frac']
    exclude_len = int(exclude_frac * sub_len)
    find_knn_variant = seq_1_info['find_knn_variant']

    if find_knn_variant == 0:
        find_knn = find_knn_0
    elif find_knn_variant == 1:
        find_knn = find_knn_1
    else:
        find_knn = find_knn_2

    n_profile = idx_end - idx_start
    mpval = np.zeros((n_profile, n_dim, n_neighbor))
    mpidx = np.zeros((n_profile, n_dim, n_neighbor))
    if verbose:
        last_pct = 0
        tic = time.time()
    last_product = np.zeros((n_sub, n_dim))
    for i, idx in enumerate(range(idx_start, idx_end)):
        dist_profile = np.zeros((n_sub, n_dim))
        for j in range(n_dim):
            que = seq_0_info['seq'][idx:idx + sub_len, j]
            que_mu = seq_0_info['seq_mu'][j][idx]
            que_sig = seq_0_info['seq_sig'][j][idx]
            if i == 0:
                que_info = {'que': que, 'que_mu': que_mu, 'que_sig': que_sig}
                dist_profile[:, j], last_product[:, j] = mass(que_info, seq_1_info, j)
            else:
                drop_val = seq_0_info['seq'][idx - 1, j]
                add_val = seq_0_info['seq'][idx + sub_len - 1, j]
                last_product[1:, j] = (
                    last_product[:-1, j] -
                    seq[:-sub_len, j] * drop_val +
                    seq[sub_len:, j] * add_val
                )
                last_product[0, j] = first_product[j][idx]
                sigma_x_eff = max(float(que_sig), EPS)
                sigma_y_eff = np.maximum(seq_sig[j], EPS)
                dist_profile[:, j] = (
                    (last_product[:, j] - sub_len * seq_mu[j] * que_mu) /
                    (sub_len * sigma_y_eff * sigma_x_eff)
                )

        invalid_val = np.inf if mode == 'discord' else -np.inf
        for j in range(n_dim):
            if seq_0_info['skip_loc'][idx, j]:
                dist_profile[:, j] = invalid_val

        dist_profile[seq_1_info['skip_loc']] = invalid_val
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
                dist_profile_0 = np.take_along_axis(
                    dist_profile, order[:, :kth + 1], axis=1)
                dist_profile[:, 0] = np.sum(dist_profile_0, axis=1)
            else:
                order = np.argpartition(dist_profile, kth, axis=1)
                dist_profile[:, 0] = np.take_along_axis(
                    dist_profile, order[:, kth:kth + 1], axis=1)[:, 0]
        dist_profile[~np.isfinite(dist_profile)] = -np.inf

        if is_selfjoin:
            dist_profile = apply_exclude(dist_profile, idx, exclude_len, n_sub)

        mpval[i, :, :], mpidx[i, :, :] = find_knn(dist_profile, n_neighbor, exclude_len)

        if verbose:
            current_pct = (i + 1) * 100 // n_profile
            if current_pct != last_pct:
                last_pct = current_pct
                elapsed = time.time() - tic
                time_left = (elapsed / current_pct) * (100 - current_pct)
                print(f'  chunk {spaces}{chunk_id:d},{current_pct: 4d}%, {time_left:0.4f} sec left')

    mpidx = mpidx.astype(int)
    mpval[~np.isfinite(mpval)] = np.nan
    return {'mpval': mpval, 'mpidx': mpidx}


def _preprocess_seq(seq, sub_len, freq_len, flat_mode='invalid'):
    seq_len = seq.shape[0]
    n_dim = seq.shape[1]
    n_sub = seq_len - sub_len + 1

    skip_loc = np.zeros((n_sub, n_dim), dtype=bool)
    for i in range(n_sub):
        for j in range(n_dim):
            if not np.all(np.isfinite(seq[i:i + sub_len, j])):
                skip_loc[i, j] = True
    seq[~np.isfinite(seq)] = 0

    seq_freq = []
    seq_mu = []
    seq_sig = []
    for i in range(n_dim):
        seq_freq_i, seq_mu_i, seq_sig_i = mass_pre(seq[:, i], sub_len, freq_len)
        seq_freq.append(seq_freq_i)
        seq_mu.append(seq_mu_i)
        seq_sig.append(seq_sig_i)

    if flat_mode == 'invalid':
        for i in range(n_sub):
            for j in range(n_dim):
                if seq_sig[j][i] < EPS:
                    skip_loc[i, j] = True
    for i in range(n_dim):
        seq_sig[i][seq_sig[i] < EPS] = EPS

    return {
        'seq': seq,
        'sub_len': sub_len,
        'seq_len': seq_len,
        'freq_len': freq_len,
        'n_sub': n_sub,
        'n_dim': n_dim,
        'skip_loc': skip_loc,
        'seq_freq': seq_freq,
        'seq_mu': seq_mu,
        'seq_sig': seq_sig,
    }
