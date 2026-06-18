import os
import glob
import json
import pickle
import argparse

from amr_cw.concepts.subgraph_miner import human_sort_key

DEFAULT_THRESHOLD = 0.6


def doc_id_from_graph_filename(filename):
    base = filename[:-4] if filename.endswith('.gml') else filename
    return base.replace('_graph_', '_', 1)


def train_graph_order(class_name, graphs_dir):
    pattern = os.path.join(graphs_dir, f"{class_name}_train_graph_*.gml")
    files = sorted((os.path.basename(f) for f in glob.glob(pattern)), key=human_sort_key)
    return [doc_id_from_graph_filename(f) for f in files]


def doc_index_to_id(class_name, graphs_dir):
    return {i: doc_id for i, doc_id in enumerate(train_graph_order(class_name, graphs_dir))}


def load_annotations(annotations_path):
    with open(annotations_path) as f:
        records = json.load(f)
    return {r['doc_id']: set(r.get('annotated_technique_ids', [])) for r in records}


def build_concept_techniques(pool, idx_to_id, annotated_by_doc, threshold=DEFAULT_THRESHOLD):
    concept_techniques = {}
    for concept in pool:
        counts = {}
        n = 0
        for idx in concept['extent']:
            doc_id = idx_to_id.get(idx)
            if doc_id is None:
                continue
            n += 1
            for t in annotated_by_doc.get(doc_id, ()):
                counts[t] = counts.get(t, 0) + 1
        techniques = [t for t, c in counts.items() if n > 0 and c / n >= threshold]
        concept_techniques[concept['id']] = sorted(techniques)
    return concept_techniques


def build_class_concept_techniques(class_name, pickle_path, graphs_dir, annotated_by_doc, threshold=DEFAULT_THRESHOLD):
    with open(pickle_path, 'rb') as f:
        pool = pickle.load(f)
    idx_to_id = doc_index_to_id(class_name, graphs_dir)
    return pool, idx_to_id, build_concept_techniques(pool, idx_to_id, annotated_by_doc, threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--pickles_prefix", required=True)
    ap.add_argument("--concept_type", default="weighted_frequent_subgraphs")
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    annotated_by_doc = load_annotations(args.annotations)
    out = {}
    for class_name in args.classes:
        pickle_path = (f"{args.pickles_prefix}_{args.concept_type}/{class_name}/"
                       f"{class_name}_{args.concept_type}.pickle")
        _pool, _idx_to_id, techniques = build_class_concept_techniques(
            class_name, pickle_path, args.graphs_dir, annotated_by_doc, args.threshold)
        out.update(techniques)

    out_path = args.out or f"{args.pickles_prefix}_{args.concept_type}_concept_techniques.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
