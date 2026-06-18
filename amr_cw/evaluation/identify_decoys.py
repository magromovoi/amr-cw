import json
import argparse

from amr_cw.evaluation.core import load_pool

DECOY_FRAME_ANCHORS = {'follow-up-03', 'circle-01'}


def decoy_concept_ids(pool, anchors=DECOY_FRAME_ANCHORS):
    out = []
    for concept in pool:
        labels = {str(n).lower() for g in concept['subgraphs'] if hasattr(g, 'nodes') for n in g.nodes()}
        if labels & anchors:
            out.append(concept['id'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--pickles_prefix", required=True)
    ap.add_argument("--concept_type", default="weighted_frequent_subgraphs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = []
    for class_name in args.classes:
        pool = load_pool(args.pickles_prefix, args.concept_type, class_name)
        hits = decoy_concept_ids(pool)
        ids.extend(hits)

    with open(args.out, "w") as f:
        json.dump(ids, f, indent=2)


if __name__ == "__main__":
    main()
