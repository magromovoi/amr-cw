import os
import json
import argparse

import numpy as np

from amr_cw.diversity.mmr import cosine_sim
from amr_cw.evaluation.core import (
    load_pool, pool_to_candidates, apply_rule, paired_permutation_test, whitened_path,
)

RULES = ('quality_only', 'facility_location', 'activation_space')


def mean_pairwise_cosine(vecs_a, vecs_b=None):
    if vecs_b is None:
        vecs_b = vecs_a
    total = 0.0
    for a in vecs_a:
        for b in vecs_b:
            total += cosine_sim(a, b)
    return total / (len(vecs_a) * len(vecs_b))


def q_matrix(instances_by_concept, names):
    m = len(names)
    d_ii = [mean_pairwise_cosine(instances_by_concept[name]) for name in names]
    Q = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            d_ij = mean_pairwise_cosine(instances_by_concept[names[i]], instances_by_concept[names[j]])
            denom = (d_ii[i] * d_ii[j]) ** 0.5
            q = d_ij / denom if denom > 0 else 0.0
            Q[i, j] = Q[j, i] = q
    return Q


def mean_off_diagonal(Q):
    m = Q.shape[0]
    if m < 2:
        return 0.0
    mask = ~np.eye(m, dtype=bool)
    return float(Q[mask].mean())


def selected_mean_off_diagonal(selected, instances_by_concept):
    names = [c['graph_concept_name'] for c in selected if c['graph_concept_name'] in instances_by_concept]
    if len(names) < 2:
        return 0.0
    return mean_off_diagonal(q_matrix(instances_by_concept, names))


def select_by_rule(candidates, rule, k, seed, activations):
    if rule == 'activation_space':
        from amr_cw.diversity.activation_space import attach_activations, select_precomputed
        attach_activations(candidates, activations)
        return select_precomputed(candidates, k)
    return apply_rule(candidates, rule, k, seed)


def run(run_dir, dataset, classes, concepts, graphs_dataset_prefix,
        graph_concepts_dataset_prefix, concept_type,
        pickles_prefix, k, rules=RULES, out_path=None):
    from amr_cw.graph.classification import get_loaders, get_model_and_device
    from amr_cw.diversity.activation_space import collect_concept_activations

    whitened = whitened_path(run_dir, dataset, concepts, 'gcn_conv', False, concept_type)
    train_loader, _test_loader, train_concept_loaders, _ = get_loaders(
        dataset, classes, graphs_dataset_prefix,
        graph_concepts_dataset_prefix + '_' + concept_type, concepts)
    model, device, *_ = get_model_and_device(
        train_loader.dataset, len(classes), whitened, whitening=True)

    per_rule = {rule: [] for rule in rules}
    for class_name in concepts:
        activations = collect_concept_activations(model, device, train_concept_loaders[class_name])
        instances = {name: [vec] for name, vec in activations.items()}
        pool_cands = pool_to_candidates(load_pool(pickles_prefix, concept_type, class_name))
        cands = [c for c in pool_cands if c['graph_concept_name'] in instances]
        for rule in rules:
            selected = select_by_rule([dict(c) for c in cands], rule, k, 0, activations)
            per_rule[rule].append(selected_mean_off_diagonal(selected, instances))

    result = {'k': k, 'concepts': list(concepts),
              'mean_off_diagonal_Q': {rule: float(np.mean(vals)) for rule, vals in per_rule.items()},
              'per_class': {rule: [float(v) for v in vals] for rule, vals in per_rule.items()}}
    baseline = np.asarray(per_rule['quality_only'])
    for rule in rules:
        if rule == 'quality_only':
            continue
        deltas = (np.asarray(per_rule[rule]) - baseline).tolist()
        result[f'{rule}_vs_quality_only'] = paired_permutation_test(deltas)
    result['diversity_reduces_similarity'] = all(
        result['mean_off_diagonal_Q'][rule] < result['mean_off_diagonal_Q']['quality_only']
        for rule in rules if rule != 'quality_only')

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--concepts", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--graphs-dataset-prefix", required=True)
    ap.add_argument("--graph-concepts-dataset-prefix", required=True)
    ap.add_argument("--concept-type", default="weighted_frequent_subgraphs")
    ap.add_argument("--pickles-prefix", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run(args.run_dir, args.dataset, args.classes, args.concepts,
        args.graphs_dataset_prefix, args.graph_concepts_dataset_prefix, args.concept_type,
        args.pickles_prefix,
        args.k, tuple(args.rules), args.out or os.path.join(args.run_dir, 'inter_concept_similarity.json'))


if __name__ == "__main__":
    main()
