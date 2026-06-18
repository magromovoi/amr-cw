import os
import json
import pickle
import argparse

CONCEPT_TYPE = "weighted_frequent_subgraphs"
CLASSES = ["coercive", "deceptive", "benign"]


def _pool_path(prefix, concept_type, class_name):
    return f"{prefix}_{concept_type}/{class_name}/{class_name}_{concept_type}.pickle"


def _tech_path(prefix, concept_type):
    return f"{prefix}_{concept_type}_concept_techniques.json"


def _namespace_concept(concept, tag):
    c = dict(concept)
    c["id"] = f"{tag}_{concept['id']}"
    c["extent"] = [f"{tag}:{e}" for e in concept["extent"]]
    return c


def merge(sources, base_out, classes, concept_type):
    out_prefix = os.path.join(base_out, "mined_concepts")
    merged_tech = {}

    for class_name in classes:
        merged_pool = []
        for tag, prefix in sources:
            with open(_pool_path(prefix, concept_type, class_name), "rb") as f:
                pool = pickle.load(f)
            with open(_tech_path(prefix, concept_type)) as f:
                tech = json.load(f)
            for concept in pool:
                ns = _namespace_concept(concept, tag)
                merged_pool.append(ns)
                if concept["id"] in tech:
                    merged_tech[ns["id"]] = tech[concept["id"]]

        out_dir = os.path.join(f"{out_prefix}_{concept_type}", class_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{class_name}_{concept_type}.pickle"), "wb") as f:
            pickle.dump(merged_pool, f)

    with open(_tech_path(out_prefix, concept_type), "w") as f:
        json.dump(merged_tech, f, indent=2)
    return out_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--base-out", required=True)
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    ap.add_argument("--concept-type", default=CONCEPT_TYPE)
    args = ap.parse_args()

    sources = []
    for s in args.sources:
        tag, prefix = s.split("=", 1)
        sources.append((tag, prefix))

    os.makedirs(args.base_out, exist_ok=True)
    merge(sources, args.base_out, args.classes, args.concept_type)


if __name__ == "__main__":
    main()
