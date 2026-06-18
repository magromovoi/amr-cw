import os
import json
import argparse

import torch

from amr_cw.evaluation.core import whitened_path


def _acc(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location='cpu')
    return float(ck['best_test_acc']), int(ck['epoch'])


def _doc_cap_mean_f1(whitened, dataset, classes, concepts, graphs_dataset_prefix):
    from amr_cw.graph.classification import get_loaders, get_model_and_device
    from amr_cw.visualization.plots import compute_axis_alignment_accuracy_and_f1

    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix)
    model, device, *_ = get_model_and_device(train_loader.dataset, len(classes), whitened, whitening=True)
    f1s = [compute_axis_alignment_accuracy_and_f1(model, device, test_loader,
                                                  orig_label=classes.index(c), axis_index=i, whitening=True)
           for i, c in enumerate(concepts)]
    return sum(f1s) / len(f1s)


def run(base_checkpoint, run_root, rules, dataset, classes, concepts,
        graphs_dataset_prefix, concept_type, out_path=None):
    base_acc, _ = _acc(base_checkpoint)
    per_rule = {}
    for rule in rules:
        run_dir = os.path.join(run_root, rule)
        whitened = whitened_path(run_dir, dataset, concepts, 'gcn_conv', False, concept_type)
        acc, epoch = _acc(whitened)
        doc_cap = _doc_cap_mean_f1(whitened, dataset, classes, concepts, graphs_dataset_prefix)
        per_rule[rule] = {'whitened_test_acc': acc, 'epoch': epoch, 'doc_cap_mean_f1': doc_cap}

    accs = [v['whitened_test_acc'] for v in per_rule.values()]
    result = {'base_test_acc': base_acc, 'per_rule': per_rule,
              'accuracy_flat_across_rules': max(accs) - min(accs) < 1e-9,
              'doc_cap_degenerate': all(v['doc_cap_mean_f1'] == 0.0 for v in per_rule.values())}
    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--rules", nargs="+", default=["none", "quality_only", "facility_location"])
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--classes", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--concepts", nargs="+", default=["coercive", "deceptive", "benign"])
    ap.add_argument("--graphs-dataset-prefix", required=True)
    ap.add_argument("--concept-type", default="weighted_frequent_subgraphs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run(args.base_checkpoint, args.run_root, args.rules, args.dataset, args.classes,
        args.concepts, args.graphs_dataset_prefix, args.concept_type, args.out)


if __name__ == "__main__":
    main()
