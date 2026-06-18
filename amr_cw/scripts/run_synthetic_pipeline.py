import os
import sys
import copy
import yaml
import argparse
import subprocess

PY = sys.executable
CONCEPT_TYPE = "weighted_frequent_subgraphs"
CLASSES = ["coercive", "deceptive", "benign"]


def variant_paths(base_out, variant):
    vdir = os.path.join(base_out, variant)
    return {
        "dir": vdir,
        "csv": os.path.join(vdir, f"{variant}_raw_texts.csv"),
        "graphs_dir": os.path.join(vdir, "graphs_dataset", "synthetic_graphs_dataset", "raw"),
        "graphs_dataset_prefix": os.path.join(vdir, "graphs_dataset", "synthetic_graphs_dataset"),
        "pickles_prefix": os.path.join(vdir, "mined_concepts"),
        "input_ckpt": os.path.join(vdir, "checkpoints_base"),
        "sidecars": os.path.join(vdir, "sidecars"),
    }


def variant_config(base_config, paths):
    cfg = copy.deepcopy(base_config)
    cfg["synthetic_graphs_dataset_prefix"] = paths["graphs_dataset_prefix"]
    cfg["synthetic_concept_pickles_prefix"] = paths["pickles_prefix"]
    cfg["input_checkpoints_prefix"] = paths["input_ckpt"]
    cfg["output_checkpoints_prefix"] = paths["input_ckpt"]
    return cfg


def build_plan(variant, base_out, min_support, rules, run_dir_for, stability=3):
    p = variant_paths(base_out, variant)
    cfg_path = os.path.join(p["dir"], "config.yaml")
    steps = []

    steps.append(("amr_graphs", [
        PY, "-m", "amr_cw.scripts.build_amr_graphs",
        "--corpus", variant, "--in", p["csv"], "--out", p["graphs_dir"]]))

    steps.append(("mine_pickles", [
        PY, "-m", "amr_cw.scripts.mine_concepts",
        "--classes", *CLASSES, "--all_classes", *CLASSES,
        "--graphs_dir", p["graphs_dir"], "--out", p["pickles_prefix"],
        "--min_support", str(min_support), "--stability", str(stability)]))

    steps.append(("base_gnn_train", [
        PY, "-m", "amr_cw.main", "--config", cfg_path, "--dataset", "synthetic",
        "--operation", "graph_classification", "--mode", "train"]))

    for rule in rules:
        rd = run_dir_for(rule)
        steps.append((f"construct:{rule}", [
            PY, "-m", "amr_cw.main", "--config", cfg_path, "--dataset", "synthetic",
            "--operation", "graph_concept_construction", "--concept_type", CONCEPT_TYPE,
            "--run_dir", rd]))
        steps.append((f"whiten:{rule}", [
            PY, "-m", "amr_cw.main", "--config", cfg_path, "--dataset", "synthetic",
            "--operation", "graph_concept_whitening", "--mode", "train",
            "--concept_type", CONCEPT_TYPE, "--run_dir", rd]))

    return steps, cfg_path, p


def write_variant_config(base_config_path, cfg_path, paths, rule_diversity):
    with open(base_config_path) as f:
        base = yaml.safe_load(f)
    cfg = variant_config(base, paths)
    cfg.update(rule_diversity)
    os.makedirs(paths["dir"], exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def rule_diversity_settings(rule):
    if rule == "none":
        return {"diversity_method": "none"}
    if rule == "quality_only":
        return {"diversity_method": "mmr", "diversity_lambda": 1.0, "diversity_sim": "jaccard"}
    if rule in ("facility_location", "set_cover", "dpp", "mmr"):
        return {"diversity_method": rule, "diversity_sim": "jaccard"}
    raise ValueError(f"unknown rule {rule}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--base-out", default="data/output/synthetic_corpus")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--min-support", type=int, required=True)
    ap.add_argument("--stability", type=int, default=3)
    ap.add_argument("--rules", nargs="+", default=["none", "quality_only", "facility_location"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def run_dir_for(rule):
        return os.path.join(args.base_out, args.variant, "runs", rule)

    steps, cfg_path, paths = build_plan(args.variant, args.base_out, args.min_support,
                                        args.rules, run_dir_for, stability=args.stability)

    if args.dry_run:
        return

    write_variant_config(args.config, cfg_path, paths, rule_diversity_settings(args.rules[0]))

    for label, argv in steps:
        rule = label.split(":", 1)[1] if ":" in label else None
        if label.startswith("construct:"):
            write_variant_config(args.config, cfg_path, paths, rule_diversity_settings(rule))
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
