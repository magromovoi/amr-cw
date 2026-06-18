import os
import csv
import time
import argparse
import networkx as nx

import amrlib
from amr_cw.amr_parsing.text_to_graph import text_to_graph, load_sentence_splitter

csv.field_size_limit(10 ** 7)

MODEL_DIR = "models/model_parse_xfm_bart_large-v0_1_0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="calibration")
    ap.add_argument("--in", dest="csv_path", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"data/output/synthetic_corpus/{args.corpus}"
    csv_path = args.csv_path or os.path.join(base, f"{args.corpus}_raw_texts.csv")
    out_dir = args.out or os.path.join(base, "document_graphs")
    os.makedirs(out_dir, exist_ok=True)

    stog = amrlib.load_stog_model(MODEL_DIR)
    nlp = load_sentence_splitter()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    cooldown_s = float(os.environ.get("AMR_COOLDOWN_S", "0") or "0")

    for i, row in enumerate(rows):
        label, split, num = row["custom_index"].rsplit("_", 2)
        out_name = f"{label}_{split}_graph_{num}.gml"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            continue
        graph = text_to_graph(row["text"], stog, nlp)
        nx.write_gml(graph, out_path)
        if cooldown_s:
            time.sleep(cooldown_s)


if __name__ == "__main__":
    main()
