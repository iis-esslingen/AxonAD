import copy

import numpy as np

from .mstomp import mstomp as cpu_mstomp


VALID_BACKENDS = ('cpu', 'gpu')


def normalize_backend(backend):
    backend_norm = str(backend).strip().lower()
    if backend_norm not in VALID_BACKENDS:
        raise ValueError(
            f'Unsupported backend={backend!r}; '
            f'expected one of {VALID_BACKENDS}.')
    return backend_norm


def resolve_backend_impl(backend):
    backend_norm = normalize_backend(backend)
    if backend_norm == 'cpu':
        return cpu_mstomp

    from .mstomp_gpu import mstomp as gpu_mstomp
    return gpu_mstomp


class MMatProAD:
    def __init__(self, sub_len, n_dim=1, n_neighbor=1, sorting_place='pre',
                 mode='discord', post_processing=2, flat_mode='eps',
                 backend='cpu', gpu_precision='float64',
                 gpu_execution='auto', gpu_allow_tf32=False,
                 gpu_device=None, gpu_reseed_period=None):
        if post_processing not in [0, 1, 2]:
            raise ValueError(
                f'post_processing must be 0, 1, or 2, got {post_processing}.')
        self.sub_len = sub_len
        self.n_dim = n_dim
        self.n_neighbor = n_neighbor
        self.sorting_place = sorting_place
        self.mode = mode
        self.post_processing = post_processing
        self.flat_mode = flat_mode
        self.backend = normalize_backend(backend)
        self.gpu_precision = str(gpu_precision)
        self.gpu_execution = str(gpu_execution)
        self.gpu_allow_tf32 = bool(gpu_allow_tf32)
        self.gpu_device = gpu_device
        self.gpu_reseed_period = gpu_reseed_period
        self.mpval = None
        self.mpidx = None

    def _mstomp_kwargs(self):
        if self.backend == 'cpu':
            return {}
        return {
            'precision': self.gpu_precision,
            'execution': self.gpu_execution,
            'allow_tf32': self.gpu_allow_tf32,
            'device': self.gpu_device,
            'reseed_period': self.gpu_reseed_period,
        }

    def get_matpro(self, seq_test, seq_train=None, n_job=1, verbose=False):
        mstomp_impl = resolve_backend_impl(self.backend)
        mstomp_kwargs = self._mstomp_kwargs()
        if seq_train is None:
            mpval, mpidx = mstomp_impl(
                seq_test,
                seq_test,
                self.sub_len,
                n_neighbor=self.n_neighbor,
                mode=self.mode,
                sorting_place=self.sorting_place,
                n_job=n_job,
                verbose=verbose,
                flat_mode=self.flat_mode,
                **mstomp_kwargs,
            )
        else:
            mpval, mpidx = mstomp_impl(
                seq_test,
                seq_train,
                self.sub_len,
                n_neighbor=self.n_neighbor,
                mode=self.mode,
                n_job=n_job,
                verbose=verbose,
                flat_mode=self.flat_mode,
                **mstomp_kwargs,
            )
        self.set_matpro(mpval, mpidx)

    def set_matpro(self, mpval, mpidx):
        assert np.all(mpval.shape == mpidx.shape)
        assert mpval.shape[2] >= self.n_neighbor
        assert mpidx.shape[2] >= self.n_neighbor
        self.mpval = mpval[:, :self.n_dim, :self.n_neighbor]
        self.mpidx = mpidx[:, :self.n_dim, :self.n_neighbor]

    def get_score(self):
        assert self.mpval is not None
        score = copy.deepcopy(self.mpval)

        for i in range(1, score.shape[1]):
            score_i = score[:, i, :]
            score_previous = score[:, i - 1, :]
            mask = np.logical_not(np.isfinite(score_i))
            score_i[mask] = score_previous[mask]
            score[:, i, :] = score_i
        score = score[:, -1, :]

        for i in range(1, self.n_neighbor):
            score_i = score[:, i]
            score_previous = score[:, i - 1]
            mask = np.logical_not(np.isfinite(score_i))
            score_i[mask] = score_previous[mask]
            score[:, i] = score_i
        score = score[:, -1]

        score = -score
        finite = np.isfinite(score)
        if np.any(finite):
            score_valid = score[finite]
            score_valid -= np.min(score_valid)
            max_val = np.max(score_valid)
            if max_val > 0:
                score_valid /= max_val
            else:
                score_valid[:] = 0
            score[finite] = score_valid
            score[~finite] = 0
        else:
            score = np.zeros_like(score)

        ts_len = score.shape[0] + self.sub_len - 1
        if self.post_processing == 0:
            score = np.pad(score, (0, self.sub_len - 1), mode='constant')
        elif self.post_processing == 1:
            pre_pad_len = (self.sub_len - 1) // 2
            post_pad_len = self.sub_len - 1 - pre_pad_len
            score = np.pad(score, (pre_pad_len, post_pad_len), mode='constant')
        else:
            score = np.pad(
                score,
                (self.sub_len - 1, self.sub_len - 1),
                mode='constant',
                constant_values=(np.nan, np.nan),
            )
            score_new = np.zeros(ts_len)
            for i in range(ts_len):
                score_new[i] = np.nanmean(score[i:i + self.sub_len])
            score = score_new
        return score
