import json
import argparse

from amr_cw.evaluation.core import (
    load_pool, load_concept_techniques, pool_to_candidates, apply_rule, distinct_techniques,
    paired_permutation_test,
)
from amr_cw.evaluation.matched_k import true_techniques_for_class


def decoy_rejection(classes, pickles_prefix, concept_type, concept_techniques, decoy_ids, rule, k, seeds):
    technique_rate = []
    decoy_rate = []
    decoys = set(decoy_ids)
    for class_name in classes:
        pool = load_pool(pickles_prefix, concept_type, class_name)
        cands = pool_to_candidates(pool)
        true_techniques = true_techniques_for_class(concept_techniques, [c['id'] for c in pool])
        n_true = max(len(true_techniques), 1)
        for seed in seeds:
            sel = apply_rule(cands, rule, k, seed)
            technique_rate.append(distinct_techniques(sel, concept_techniques, true_techniques) / n_true)
            sel_names = {c['graph_concept_name'] for c in sel}
            decoy_rate.append(len(sel_names & decoys) / len(decoys) if decoys else 0.0)
    mean_technique = sum(technique_rate) / len(technique_rate) if technique_rate else 0.0
    mean_decoy = sum(decoy_rate) / len(decoy_rate) if decoy_rate else 0.0
    return {'mean_technique_recovery_rate': mean_technique, 'mean_decoy_recovery_rate': mean_decoy,
            'decoys_rejected': mean_technique > mean_decoy}


def null_non_effect(classes, null_pickles_prefix, concept_type, null_concept_techniques, k, seeds, diversity_rule):
    deltas = []
    for class_name in classes:
        pool = load_pool(null_pickles_prefix, concept_type, class_name)
        cands = pool_to_candidates(pool)
        true_techniques = true_techniques_for_class(null_concept_techniques, [c['id'] for c in pool])
        for seed in seeds:
            div = distinct_techniques(apply_rule(cands, diversity_rule, k, seed), null_concept_techniques, true_techniques)
            qual = distinct_techniques(apply_rule(cands, 'quality_only', k, seed), null_concept_techniques, true_techniques)
            deltas.append(div - qual)
    test = paired_permutation_test(deltas)
    no_benefit = not (test['mean_delta'] > 0 and test['p_value'] < 0.05)
    return {'diversity_vs_quality_only': test, 'null_shows_no_benefit': no_benefit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--pickles_prefix", required=True)
    ap.add_argument("--concept_techniques", required=True)
    ap.add_argument("--decoy_ids", nargs="+", default=[])
    ap.add_argument("--null_pickles_prefix", required=True)
    ap.add_argument("--null_concept_techniques", required=True)
    ap.add_argument("--concept_type", default="weighted_frequent_subgraphs")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--diversity_rule", default="facility_location")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cr = load_concept_techniques(args.concept_techniques)
    null_cr = load_concept_techniques(args.null_concept_techniques)
    decoy = decoy_rejection(args.classes, args.pickles_prefix, args.concept_type, cr,
                            args.decoy_ids, args.diversity_rule, args.k, args.seeds)
    null = null_non_effect(args.classes, args.null_pickles_prefix, args.concept_type, null_cr,
                           args.k, args.seeds, args.diversity_rule)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({'decoy_rejection': decoy, 'null_non_effect': null}, f, indent=2)


if __name__ == "__main__":
    main()
