import os
import json
import pickle

import numpy as np

from amr_cw.diversity.mmr import jaccard, mmr_select
from amr_cw.diversity.selectors import facility_location, set_cover, greedy_map_dpp


def whitened_path(run_dir, dataset, concepts, graph_conv_type, graph_residual_connections, concept_type):
    residual = '_residual' if graph_residual_connections else '_non_residual'
    concepts_name = '_' + '_'.join(concepts)
    name = (dataset + concepts_name + '_' + graph_conv_type + residual + '_' + concept_type
            + '_whitened_graph_model.pt')
    return os.path.join(run_dir, 'checkpoints', name)


def pool_to_candidates(pool):
    return [{
        'graph_concept_name': c['id'],
        'extent': c['extent'],
        'support': c['supports'][0],
        'penalty': c['penalty_5'],
        'graph_concept': c['subgraphs'][0] if c.get('subgraphs') else None,
    } for c in pool]


def load_pool(pickles_prefix, concept_type, class_name):
    path = f"{pickles_prefix}_{concept_type}/{class_name}/{class_name}_{concept_type}.pickle"
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_concept_techniques(path):
    with open(path) as f:
        return json.load(f)


def apply_rule(candidates, rule, k, seed=0):
    if rule == 'random':
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        return [candidates[i] for i in sorted(idx)]
    if rule == 'quality_only':
        return mmr_select(candidates, lam=1.0, min_k=k)[:k]
    if rule == 'mmr':
        return mmr_select(candidates, lam=0.5, min_k=k)[:k]
    if rule == 'facility_location':
        return facility_location(candidates, min_k=k)
    if rule == 'set_cover':
        return set_cover(candidates, min_k=k)
    if rule == 'dpp':
        return greedy_map_dpp(candidates, min_k=k)
    raise ValueError(f"unknown rule {rule}")


def distinct_techniques(selected, concept_techniques, true_technique_ids):
    true = set(true_technique_ids)
    recovered = set()
    for c in selected:
        recovered |= set(concept_techniques.get(c['graph_concept_name'], [])) & true
    return len(recovered)


def extent_jaccard_stats(candidates, identical_threshold=1.0):
    n = len(candidates)
    if n < 2:
        return {'mean_pairwise_jaccard': 0.0, 'identical_pairs': 0, 'n': n}
    extents = [set(map(str, c['extent'])) for c in candidates]
    total = 0.0
    pairs = 0
    identical = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = jaccard(extents[i], extents[j])
            total += s
            pairs += 1
            if s >= identical_threshold:
                identical += 1
    return {'mean_pairwise_jaccard': total / pairs, 'identical_pairs': identical, 'n': n}


def coverage(selected):
    covered = set()
    for c in selected:
        covered |= set(map(str, c['extent']))
    return len(covered)


def paired_permutation_test(deltas, n_perm=10000, seed=0):
    deltas = np.asarray([d for d in deltas if d is not None], dtype=float)
    n = len(deltas)
    if n == 0:
        return {'n': 0, 'mean_delta': 0.0, 'p_value': 1.0}
    observed = deltas.mean()
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * np.abs(deltas)).mean(axis=1)
    p = (np.abs(perm_means) >= abs(observed) - 1e-12).mean()
    return {'n': n, 'mean_delta': float(observed), 'p_value': float(p)}
