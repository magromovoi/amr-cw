import json
import argparse
from time import perf_counter

from amr_cw.evaluation.core import (
    load_pool, load_concept_techniques, pool_to_candidates, apply_rule, distinct_techniques,
    extent_jaccard_stats, coverage,
)
from amr_cw.evaluation.matched_k import true_techniques_for_class

METHODS = ['quality_only', 'facility_location', 'set_cover', 'dpp']


def run(classes, pickles_prefix, concept_type, concept_techniques, k, seeds):
    per_method = {}
    for method in METHODS:
        techniques, covs, sizes, redund, secs = [], [], [], [], []
        for class_name in classes:
            pool = load_pool(pickles_prefix, concept_type, class_name)
            cands = pool_to_candidates(pool)
            true_techniques = true_techniques_for_class(concept_techniques, [c['id'] for c in pool])
            full_cov = coverage(cands)
            for seed in seeds:
                t0 = perf_counter()
                sel = apply_rule(cands, method, k, seed)
                secs.append(perf_counter() - t0)
                techniques.append(distinct_techniques(sel, concept_techniques, true_techniques))
                covs.append(coverage(sel) / full_cov if full_cov else 0.0)
                sizes.append(len(sel))
                redund.append(extent_jaccard_stats(sel)['mean_pairwise_jaccard'])
        per_method[method] = {
            'mean_techniques': sum(techniques) / len(techniques),
            'mean_coverage_retained': sum(covs) / len(covs),
            'mean_selected_size': sum(sizes) / len(sizes),
            'mean_pairwise_jaccard': sum(redund) / len(redund),
            'mean_select_seconds': sum(secs) / len(secs),
        }
    baseline = per_method['quality_only']['mean_techniques']
    cross_method = {m: per_method[m]['mean_techniques'] > baseline
                    for m in METHODS if m != 'quality_only'}
    return {'per_method': per_method, 'cross_method_improves': cross_method,
            'all_diversity_improve': all(cross_method.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--pickles_prefix", required=True)
    ap.add_argument("--concept_type", default="weighted_frequent_subgraphs")
    ap.add_argument("--concept_techniques", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(args.classes, args.pickles_prefix, args.concept_type,
                 load_concept_techniques(args.concept_techniques), args.k, args.seeds)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
