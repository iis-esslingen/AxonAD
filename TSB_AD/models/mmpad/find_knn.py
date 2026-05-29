import copy

import numpy as np

from .util import apply_exclude


def find_knn_0(dist_profile, n_neighbor, exclude_len):
    dist_profile = copy.deepcopy(dist_profile)
    n_sub = dist_profile.shape[0]
    n_dim = dist_profile.shape[1]
    mpval = np.zeros((n_dim, n_neighbor))
    mpidx = np.zeros((n_dim, n_neighbor))
    for i in range(n_dim):
        for j in range(n_neighbor):
            idx = np.argmax(dist_profile[:, i])
            if dist_profile[idx, i] == -np.inf:
                mpidx[i, j] = -1
                mpval[i, j] = -np.inf
            else:
                mpidx[i, j] = idx
                mpval[i, j] = dist_profile[idx, i]
            dist_profile[:, i:i + 1] = apply_exclude(
                dist_profile[:, i:i + 1], idx, exclude_len, n_sub)
    return mpval, mpidx


def find_knn_1(dist_profile, n_neighbor, exclude_len):
    if n_neighbor == 1:
        return find_knn_0(dist_profile, n_neighbor, exclude_len)
    dist_profile = copy.deepcopy(dist_profile)
    n_sub = dist_profile.shape[0]
    n_dim = dist_profile.shape[1]
    mpval = np.full((n_dim, n_neighbor), -np.inf)
    mpidx = -np.ones((n_dim, n_neighbor))
    for i in range(n_dim):
        max_idx = np.argsort(-dist_profile[:, i])
        neighbor_count = 0
        for idx in max_idx:
            if dist_profile[idx, i] == -np.inf:
                continue
            mpidx[i, neighbor_count] = idx
            mpval[i, neighbor_count] = dist_profile[idx, i]
            dist_profile[:, i:i + 1] = apply_exclude(
                dist_profile[:, i:i + 1], idx, exclude_len, n_sub)
            neighbor_count += 1
            if neighbor_count == n_neighbor:
                break
    return mpval, mpidx


def find_knn_2(dist_profile, n_neighbor, exclude_len):
    if n_neighbor == 1:
        return find_knn_0(dist_profile, n_neighbor, exclude_len)
    dist_profile = copy.deepcopy(dist_profile)
    n_sub = dist_profile.shape[0]
    n_dim = dist_profile.shape[1]
    mpval = np.full((n_dim, n_neighbor), -np.inf)
    mpidx = -np.ones((n_dim, n_neighbor))
    kth = min(n_neighbor * exclude_len * 2, n_sub - 1)
    for i in range(n_dim):
        max_idx_sub = np.argpartition(-dist_profile[:, i], kth)[:kth]
        max_idx = max_idx_sub[np.argsort(-dist_profile[max_idx_sub, i])]
        neighbor_count = 0
        for idx in max_idx:
            if dist_profile[idx, i] == -np.inf:
                continue
            mpidx[i, neighbor_count] = idx
            mpval[i, neighbor_count] = dist_profile[idx, i]
            dist_profile[:, i:i + 1] = apply_exclude(
                dist_profile[:, i:i + 1], idx, exclude_len, n_sub)
            neighbor_count += 1
            if neighbor_count == n_neighbor:
                break
    return mpval, mpidx
