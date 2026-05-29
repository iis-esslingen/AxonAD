"""
This submission is by Michael Yeh.

The implementation comes from my repository:
https://github.com/mcyeh/mmpad_tsb

This submission adapts that method to the TSB-AD model interface.
When CUDA is available, the implementation can use the GPU backend.
Otherwise it runs on CPU.

Related papers:

Matrix Profile for Time-Series Anomaly Detection: A Reproducible Open-Source
Benchmark on TSB-AD
Author: Chin-Chia Michael Yeh
Link: https://arxiv.org/abs/2604.02445

Matrix profile for anomaly detection on multidimensional time series
Authors: Chin-Chia Michael Yeh, Audrey Der, Uday Singh Saini, Vivian Lai,
Yan Zheng, Junpeng Wang, Xin Dai, Zhongfang Zhuang, Yujie Fan,
Huiyuan Chen, Prince Osei Aboagye, Liang Wang, Wei Zhang, Eamonn Keogh
Link: https://arxiv.org/abs/2409.09298
"""

import multiprocessing

import numpy as np
import torch


from .mmpad.mmp_ad import MMatProAD, normalize_backend
from .mmpad.util import (
    MMPAD_DEFAULT_TIME_BUDGET,
    downsample_sequence,
    normalize_budget_mode,
    resolve_mmpad_budget_state,
    resolve_n_dim,
    to_2d_ts,
    upsample_score_linear,
    validate_score,
    zscore,
)

MMPAD_RUN_DEFAULTS = {
    'periodicity': 1,
    'n_dim': None,
    'n_neighbor': 1,
    'sorting_place': 'pre',
    'mode': 'discord',
    'post_processing': 2,
    'n_job': 1,
    'use_train_reference': False,
    'verbose': False,
    'flat_mode': 'eps',
    'budget_mode': 'downsample',
    'time_budget': MMPAD_DEFAULT_TIME_BUDGET,
    'backend': 'cpu',
    'gpu_precision': 'float64',
    'gpu_execution': 'auto',
    'gpu_allow_tf32': False,
    'gpu_device': None,
    'gpu_reseed_period': None,
}


def resolve_mmpad_params(params=None):
    params_eff = dict(MMPAD_RUN_DEFAULTS)
    if params:
        params_eff.update(params)
    params_eff['budget_mode'] = normalize_budget_mode(params_eff['budget_mode'])
    params_eff['time_budget'] = float(params_eff['time_budget'])
    params_eff['backend'] = normalize_backend(params_eff['backend'])
    params_eff['gpu_precision'] = str(params_eff['gpu_precision'])
    params_eff['gpu_execution'] = str(params_eff['gpu_execution'])
    params_eff['gpu_allow_tf32'] = bool(params_eff['gpu_allow_tf32'])
    if params_eff['gpu_reseed_period'] in {'', 'None'}:
        params_eff['gpu_reseed_period'] = None
    elif params_eff['gpu_reseed_period'] is not None:
        params_eff['gpu_reseed_period'] = int(params_eff['gpu_reseed_period'])
    if params_eff['gpu_device'] in {'', 'None'}:
        params_eff['gpu_device'] = None
    return params_eff


class MMPAD:
    def __init__(self, periodicity=1, n_dim=None, n_neighbor=1,
                 sorting_place='pre', mode='discord', post_processing=2,
                 n_job=None, use_train_reference=False, verbose=False,
                 flat_mode='eps', budget_mode='downsample',
                 time_budget=MMPAD_DEFAULT_TIME_BUDGET,
                 backend=None, gpu_precision='float64',
                 gpu_execution='auto', gpu_allow_tf32=False,
                 gpu_device=None, gpu_reseed_period=None):
        
        if n_job is None:
            n_job = max(1, multiprocessing.cpu_count() - 1)
        if backend is None:
            if torch.cuda.is_available():
                backend = "gpu"
            else:
                backend = "cpu"

        self.periodicity = periodicity
        self.n_dim = n_dim
        self.n_neighbor = n_neighbor
        self.sorting_place = sorting_place
        self.mode = mode
        self.post_processing = post_processing
        self.n_job = n_job
        self.use_train_reference = use_train_reference
        self.verbose = verbose
        self.flat_mode = flat_mode
        self.budget_mode = normalize_budget_mode(budget_mode)
        self.time_budget = float(time_budget)
        self.backend = normalize_backend(backend)
        self.gpu_precision = str(gpu_precision)
        self.gpu_execution = str(gpu_execution)
        self.gpu_allow_tf32 = bool(gpu_allow_tf32)
        self.gpu_device = gpu_device
        self.gpu_reseed_period = gpu_reseed_period

    def _build_detector(self, n_dim, sub_len):
        return MMatProAD(
            sub_len=sub_len,
            n_dim=n_dim,
            n_neighbor=int(self.n_neighbor),
            sorting_place=self.sorting_place,
            mode=self.mode,
            post_processing=int(self.post_processing),
            flat_mode=self.flat_mode,
            backend=self.backend,
            gpu_precision=self.gpu_precision,
            gpu_execution=self.gpu_execution,
            gpu_allow_tf32=self.gpu_allow_tf32,
            gpu_device=self.gpu_device,
            gpu_reseed_period=self.gpu_reseed_period,
        )

    def fit(self, x, y=None):
        seq_raw, _ = to_2d_ts(x)
        n_samples = seq_raw.shape[0]
        budget_state = resolve_mmpad_budget_state(
            seq_raw,
            periodicity=self.periodicity,
            budget_mode=self.budget_mode,
            time_budget=self.time_budget,
        )
        seq_work = np.asarray(budget_state['data'], dtype=float)
        seq_work = zscore(seq_work, axis=0, ddof=0)
        working_length = int(seq_work.shape[0])

        sub_len = int(budget_state['sub_len'])
        n_dim = resolve_n_dim(self.n_dim, seq_work.shape[1])

        self.downsample_factor_ = int(budget_state['downsample_factor'])
        self.sub_len_ = sub_len
        self.n_dim_ = n_dim
        self.seq_train_ = seq_work.copy()
        self.detector_ = self._build_detector(n_dim=n_dim, sub_len=sub_len)
        self.detector_.get_matpro(
            seq_test=seq_work,
            seq_train=None,
            n_job=int(self.n_job),
            verbose=bool(self.verbose),
        )

        score = self.detector_.get_score()
        score = validate_score(score, working_length, score_name='MMPAD scores')
        if self.downsample_factor_ > 1:
            score = upsample_score_linear(score, n_samples)
        self.decision_scores_ = validate_score(score, n_samples, score_name='MMPAD scores')
        return self

    def decision_function(self, x):
        if not hasattr(self, 'detector_'):
            raise RuntimeError('Call fit() before decision_function().')

        seq_raw, _ = to_2d_ts(x)
        n_samples = seq_raw.shape[0]
        seq_work = downsample_sequence(seq_raw, factor=getattr(self, 'downsample_factor_', 1))
        if seq_work.shape[0] < self.sub_len_:
            raise ValueError(
                'Input is too short for the fitted MMPAD window after '
                f'downsampling: working_length={seq_work.shape[0]} < '
                f'sub_len={self.sub_len_}.'
            )
        seq_work = zscore(seq_work, axis=0, ddof=0)

        seq_train = None
        if self.use_train_reference and seq_work.shape[1] == self.seq_train_.shape[1]:
            seq_train = self.seq_train_

        detector = self._build_detector(n_dim=self.n_dim_, sub_len=self.sub_len_)
        detector.get_matpro(
            seq_test=seq_work,
            seq_train=seq_train,
            n_job=int(self.n_job),
            verbose=bool(self.verbose),
        )
        score = detector.get_score()
        score = validate_score(score, seq_work.shape[0], score_name='MMPAD scores')
        if getattr(self, 'downsample_factor_', 1) > 1:
            score = upsample_score_linear(score, n_samples)
        return validate_score(score, n_samples, score_name='MMPAD scores')
