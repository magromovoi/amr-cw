import json
import argparse

from amr_cw.evaluation.core import (
    load_pool, load_concept_techniques, pool_to_candidates, apply_rule, distinct_techniques,
    paired_permutation_test,
)

DIVERSITY_RULES = ['facility_location', 'set_cover', 'dpp']


def true_techniques_for_class(concept_techniques, class_pool_ids):
    techniques = set()
    for cid in class_pool_ids:
        techniques |= set(concept_techniques.get(cid, []))
    return sorted(techniques)


def run(classes, pickles_prefix, concept_type, concept_techniques_path, k, seeds, diversity_rule):
    concept_techniques = load_concept_techniques(concept_techniques_path)
    cells = []
    for class_name in classes:
        pool = load_pool(pickles_prefix, concept_type, class_name)
        cands = pool_to_candidates(pool)
        true_techniques = true_techniques_for_class(concept_techniques, [c['id'] for c in pool])
        for seed in seeds:
            div = distinct_techniques(apply_rule(cands, diversity_rule, k, seed), concept_techniques, true_techniques)
            rnd = distinct_techniques(apply_rule(cands, 'random', k, seed), concept_techniques, true_techniques)
            qual = distinct_techniques(apply_rule(cands, 'quality_only', k, seed), concept_techniques, true_techniques)
            cells.append({'class': class_name, 'seed': seed,
                          'diversity': div, 'random': rnd, 'quality_only': qual})

    vs_random = paired_permutation_test([c['diversity'] - c['random'] for c in cells])
    vs_quality = paired_permutation_test([c['diversity'] - c['quality_only'] for c in cells])
    beats_both = vs_random['mean_delta'] > 0 and vs_quality['mean_delta'] > 0
    return {'rule': diversity_rule, 'k': k, 'effective_n': len(cells),
            'cells': cells, 'diversity_vs_random': vs_random,
            'diversity_vs_quality_only': vs_quality,
            'diversity_beats_both': beats_both}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--pickles_prefix", required=True)
    ap.add_argument("--concept_type", default="weighted_frequent_subgraphs")
    ap.add_argument("--concept_techniques", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--diversity_rule", default="facility_location", choices=DIVERSITY_RULES)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(args.classes, args.pickles_prefix, args.concept_type, args.concept_techniques,
                 args.k, args.seeds, args.diversity_rule)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
