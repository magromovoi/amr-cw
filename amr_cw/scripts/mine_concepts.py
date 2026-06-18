import os
import glob
import pickle
import argparse
import networkx as nx

from amr_cw.concepts.subgraph_miner import mine_frequent_subgraphs, human_sort_key
from amr_cw.concepts.lattice import stable_concepts, closed_connected_patterns
from amr_cw.concepts.weighted_pickles import build_frequent_subgraphs

RAW = "../parakal-original/data/datasets/bbcsport_graphs_dataset/raw"
OUT = "data/output/mined_concepts"

ALL_CLASSES = ["cricket", "football", "rugby", "athletics", "tennis"]


def load_class_train_graphs(class_name, raw_dir):
    pattern = os.path.join(raw_dir, f"{class_name}_train_graph_*.gml")
    files = sorted((os.path.basename(f) for f in glob.glob(pattern)), key=human_sort_key)
    return [nx.read_gml(os.path.join(raw_dir, f)) for f in files]


def load_other_class_triples(class_name, raw_dir, all_classes):
    other = []
    for cls in all_classes:
        if cls == class_name:
            continue
        for f in glob.glob(os.path.join(raw_dir, f"{cls}_train_graph_*.gml")):
            other.append(f)
    return [{(u, v, d["label"]) for u, v, d in nx.read_gml(f).edges(data=True)} for f in other]


def save_pickle(concepts, concept_type, class_name, out_prefix):
    out_dir = f"{out_prefix}_{concept_type}/{class_name}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{class_name}_{concept_type}.pickle")
    with open(path, "wb") as f:
        pickle.dump(concepts, f)
    return path


def build_class(class_name, min_support, stability_threshold, raw_dir, out_prefix, all_classes):
    graphs = load_class_train_graphs(class_name, raw_dir)
    mined, _ = mine_frequent_subgraphs(graphs, min_support)
    stable, _graph_triples, _freq1 = stable_concepts(graphs, min_support, stability_threshold)
    type1, _by_extent = closed_connected_patterns(mined, stable)
    other = load_other_class_triples(class_name, raw_dir, all_classes)
    freq = build_frequent_subgraphs(type1, class_name, other)
    save_pickle(freq, "weighted_frequent_subgraphs", class_name, out_prefix)
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["cricket"])
    ap.add_argument("--min_support", type=int, default=22)
    ap.add_argument("--stability", type=int, default=5)
    ap.add_argument("--graphs_dir", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--all_classes", nargs="+", default=None)
    args = ap.parse_args()

    all_classes = args.all_classes if args.all_classes else ALL_CLASSES
    for class_name in args.classes:
        build_class(class_name, args.min_support, args.stability, args.graphs_dir, args.out, all_classes)


if __name__ == "__main__":
    main()
