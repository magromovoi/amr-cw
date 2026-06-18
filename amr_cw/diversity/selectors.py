from math import inf, sqrt

import numpy as np

from amr_cw.diversity.mmr import jaccard, normalized_quality

QUALITY_FLOOR = 1e-3


def budget(min_k, n):
    if min_k <= 0:
        return n
    return min(min_k, n)


def extent_features(candidates):
    return [set(map(str, c['extent'])) for c in candidates]


def sim_matrix(features, sim_fn):
    n = len(features)
    S = np.zeros((n, n))
    for i in range(n):
        S[i, i] = sim_fn(features[i], features[i])
        for j in range(i + 1, n):
            s = sim_fn(features[i], features[j])
            S[i, j] = s
            S[j, i] = s
    return S


def facility_location_from_matrix(S, k):
    n = S.shape[0]
    k = min(k, n)
    selected = []
    remaining = set(range(n))
    coverage = np.zeros(n)
    while len(selected) < k:
        best_gain = -inf
        best_j = -1
        for j in remaining:
            gain = np.maximum(coverage, S[:, j]).sum() - coverage.sum()
            if gain > best_gain:
                best_gain = gain
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
        coverage = np.maximum(coverage, S[:, best_j])
    return selected


def greedy_map_from_kernel(L, k):
    n = L.shape[0]
    k = min(k, n)
    d2 = np.diag(L).astype(float).copy()
    chol = [[] for _ in range(n)]

    first = int(np.argmax(d2))
    selected = [first]
    remaining = [i for i in range(n) if i != first]

    while len(selected) < k and remaining:
        pivot = selected[-1]
        cp = np.array(chol[pivot])
        dp = sqrt(d2[pivot]) if d2[pivot] > 0 else 0.0

        for i in remaining:
            if dp <= 1e-12:
                e = 0.0
            else:
                ci = np.array(chol[i])
                dot = float(np.dot(cp, ci)) if cp.size else 0.0
                e = (L[pivot, i] - dot) / dp
            chol[i].append(e)
            d2[i] -= e * e

        best_i = max(remaining, key=lambda i: d2[i])
        if d2[best_i] <= 1e-10:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


def facility_location(candidates, min_k, sim_fn=jaccard, feature_fn=extent_features):
    n = len(candidates)
    if n <= 1:
        return candidates
    S = sim_matrix(feature_fn(candidates), sim_fn)
    return [candidates[i] for i in facility_location_from_matrix(S, budget(min_k, n))]


def set_cover(candidates, min_k):
    n = len(candidates)
    if n <= 1:
        return candidates
    k = budget(min_k, n)
    extents = extent_features(candidates)
    quality = normalized_quality(candidates)

    covered = set()
    selected = []
    remaining = set(range(n))
    while len(selected) < k and remaining:
        best_j = -1
        best_key = None
        for j in remaining:
            key = (len(extents[j] - covered), quality[j])
            if best_key is None or key > best_key:
                best_key = key
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
        covered |= extents[best_j]
    return [candidates[i] for i in selected]


def greedy_map_dpp(candidates, min_k, sim_fn=jaccard, feature_fn=extent_features):
    n = len(candidates)
    if n <= 1:
        return candidates
    S = sim_matrix(feature_fn(candidates), sim_fn)
    q = np.array([max(v, QUALITY_FLOOR) for v in normalized_quality(candidates)])
    L = np.outer(q, q) * S
    return [candidates[i] for i in greedy_map_from_kernel(L, budget(min_k, n))]
