import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from amr_cw.evaluation.core import load_pool, pool_to_candidates, apply_rule, distinct_techniques
from amr_cw.evaluation.matched_k import true_techniques_for_class

COERCIVE_TECHS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]


def _pool_means(pickles_prefix, concept_techniques, classes, budgets, seeds):
    with open(concept_techniques) as f:
        ct = json.load(f)
    out = {}
    for k in budgets:
        fl, qo = [], []
        for c in classes:
            pool = load_pool(pickles_prefix, "weighted_frequent_subgraphs", c)
            cands = pool_to_candidates(pool)
            tt = true_techniques_for_class(ct, [x["id"] for x in pool])
            for s in seeds:
                fl.append(distinct_techniques(apply_rule(cands, "facility_location", k, s), ct, tt))
                qo.append(distinct_techniques(apply_rule(cands, "quality_only", k, s), ct, tt))
        out[k] = (float(np.mean(qo)), float(np.mean(fl)))
    return out


def primary_endpoint(calibration_json, pickles_prefix, concept_techniques, out_path):
    with open(calibration_json) as f:
        cal = json.load(f)
    budgets = [4, 6, 8]
    cal_qo, cal_fl = [], []
    for k in budgets:
        qo = np.mean([cal["classes"][c]["budgets"][str(k)]["quality_only"] for c in cal["classes"]])
        fl = np.mean([cal["classes"][c]["budgets"][str(k)]["facility_location"] for c in cal["classes"]])
        cal_qo.append(qo)
        cal_fl.append(fl)
    pool = _pool_means(pickles_prefix, concept_techniques, list(cal["classes"]), budgets, [0, 1, 2, 3, 4])
    ev_qo = [pool[k][0] for k in budgets]
    ev_fl = [pool[k][1] for k in budgets]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(budgets))
    w = 0.38
    for ax, qo, fl, title in [(axes[0], cal_qo, cal_fl, "Calibration dataset (195 docs)"),
                              (axes[1], ev_qo, ev_fl, "Eval dataset (1045 docs)")]:
        ax.bar(x - w / 2, qo, w, label="quality-only (base rule)")
        ax.bar(x + w / 2, fl, w, label="facility location (diversity)")
        for xi, (a, b) in enumerate(zip(qo, fl)):
            ax.text(xi - w / 2, a + 0.05, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
            ax.text(xi + w / 2, b + 0.05, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in budgets])
        ax.set_title(title)
        ax.set_xlabel("review budget")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylim(top=8.2)
    axes[0].set_ylabel("distinct techniques recovered")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Techniques recovered per review budget")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def example(pickles_prefix, concept_techniques, out_path, k=6, seed=0, class_name="coercive"):
    with open(concept_techniques) as f:
        ct = json.load(f)
    pool = load_pool(pickles_prefix, "weighted_frequent_subgraphs", class_name)
    cands = pool_to_candidates(pool)
    tt = set(true_techniques_for_class(ct, [x["id"] for x in pool]))
    n_tech = len(COERCIVE_TECHS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, rule, title in [(axes[0], "quality_only", "quality-only (base rule)"),
                            (axes[1], "facility_location", "facility location (diversity)")]:
        sel = apply_rule(cands, rule, k, seed)
        recovered = set()
        for r, c in enumerate(sel):
            techs = set(ct.get(c["graph_concept_name"], []))
            recovered |= techs & tt
            short = "sg" + c["graph_concept_name"].rsplit("_frequent_subgraph_", 1)[-1]
            y = k - 1 - r
            ax.text(-0.3, y + 0.5, f"{short}", ha="right", va="center", fontsize=9, family="monospace")
            ax.text(-0.3, y + 0.18, f"sup {c['support']}", ha="right", va="center", fontsize=6.5)
            for j, d in enumerate(COERCIVE_TECHS):
                filled = d in techs
                ax.add_patch(Rectangle((j, y), 0.92, 0.92, facecolor="gray" if filled else "white",
                                       edgecolor="black", linewidth=0.6))
        for j, d in enumerate(COERCIVE_TECHS):
            ax.text(j + 0.46, k + 0.15, d, ha="center", va="bottom", fontsize=8, family="monospace")
        ax.text(-3.0, k / 2.0, "sg* = selected subgraph concepts", rotation=90, ha="center", va="center", fontsize=8)
        n_rec = len(recovered)
        missed = sorted(tt - recovered, key=lambda t: int(t[1:]))
        ax.set_title(f"{title}\nrecovers {n_rec}/{n_tech}   misses {', '.join(missed) if missed else 'none'}")
        ax.set_xlim(-3.2, n_tech)
        ax.set_ylim(-0.3, k + 0.8)
        ax.axis("off")
    fig.suptitle("Coercive class, budget B=6: which techniques each rule's 6 concepts cover")
    fig.text(0.5, 0.01, "filled cell = the concept maps to that technique (C1 suspend .. C8 lose)",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def inter_concept_q(bbcsport_json, smoke_json, out_path):
    with open(bbcsport_json) as f:
        bb = json.load(f)
    with open(smoke_json) as f:
        sm = json.load(f)
    rules = ["quality_only", "facility_location", "activation_space"]
    labels = ["quality-only\n(base rule)", "facility\nlocation", "activation\nspace"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, data, title in [(axes[0], bb, "BBC Sport"),
                            (axes[1], sm, "Synthetic eval pool")]:
        vals = [data["mean_off_diagonal_Q"][r] for r in rules]
        ax.bar(range(len(rules)), vals)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylim(top=0.8)
    axes[0].set_ylabel("mean similarity")
    fig.suptitle("Inter-concept similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def convergent(convergent_json, out_path):
    with open(convergent_json) as f:
        data = json.load(f)["per_method"]
    order = ["quality_only", "facility_location", "dpp", "set_cover"]
    labels = ["quality-only", "facility loc.", "DPP", "set cover"]
    techs = [data[m]["mean_techniques"] for m in order]
    covs = [data[m]["mean_coverage_retained"] for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    x = np.arange(len(order))
    axes[0].bar(x, techs)
    for i, v in enumerate(techs):
        axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Distinct techniques @ budget B")
    axes[0].set_ylabel("mean over classes and seeds")
    axes[0].set_ylim(top=6.2)
    axes[1].bar(x, covs)
    for i, v in enumerate(covs):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Document coverage retained")
    axes[1].set_ylim(top=1.1)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.suptitle("Performance of diversity selectors on the eval dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="data/output/synthetic_corpus/combined")
    ap.add_argument("--eval-q-json", default="data/output/synthetic_corpus/eval/inter_concept_q_none.json")
    ap.add_argument("--calibration-json", default="data/output/synthetic_corpus/calibration/mvp_primary_endpoint.json")
    ap.add_argument("--bbcsport-json", default="data/output/parakal_anchor/bbcsport_inter_concept_similarity_k6.json")
    ap.add_argument("--out-dir", default=".resources/paper")
    args = ap.parse_args()

    pickles = os.path.join(args.eval_root, "mined_concepts")
    ct = os.path.join(args.eval_root, "mined_concepts_weighted_frequent_subgraphs_concept_techniques.json")
    os.makedirs(args.out_dir, exist_ok=True)

    primary_endpoint(args.calibration_json, pickles, ct, os.path.join(args.out_dir, "figure-primary.png"))
    example(pickles, ct, os.path.join(args.out_dir, "figure-example.png"))
    inter_concept_q(args.bbcsport_json, args.eval_q_json, os.path.join(args.out_dir, "figure-inter-concept-q.png"))
    convergent(os.path.join(args.eval_root, "convergent_k6.json"), os.path.join(args.out_dir, "figure-convergent.png"))


if __name__ == "__main__":
    main()
