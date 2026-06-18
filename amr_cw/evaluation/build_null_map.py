import json
import argparse

import numpy as np

from amr_cw.evaluation.core import load_pool, load_concept_techniques


def build_null_map(classes, pickles_prefix, concept_type, concept_techniques, seed):
    rng = np.random.RandomState(seed)
    null = {}
    for class_name in classes:
        pool = load_pool(pickles_prefix, concept_type, class_name)
        ids = [c['id'] for c in pool]
        labels = [concept_techniques.get(cid, []) for cid in ids]
        perm = rng.permutation(len(ids))
        for cid, j in zip(ids, perm):
            null[cid] = labels[j]
    return null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--pickles_prefix", required=True)
    ap.add_argument("--concept_type", default="weighted_frequent_subgraphs")
    ap.add_argument("--concept_techniques", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    real = load_concept_techniques(args.concept_techniques)
    null = build_null_map(args.classes, args.pickles_prefix, args.concept_type, real, args.seed)
    with open(args.out, "w") as f:
        json.dump(null, f, indent=2)


if __name__ == "__main__":
    main()
