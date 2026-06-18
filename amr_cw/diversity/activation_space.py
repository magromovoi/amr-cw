import os

import torch
from torch_geometric.nn import global_mean_pool

from amr_cw.diversity.mmr import cosine_sim
from amr_cw.diversity.selectors import sim_matrix, facility_location_from_matrix, budget


def _concept_name(file_path):
    base = os.path.basename(file_path)
    return base.replace('_train.gml', '').replace('_test.gml', '')


def collect_concept_activations(model, device, concept_loader):
    model = model.to(device)
    model.eval()

    outputs = []

    def hook(module, input, output):
        from amr_cw.core.iterative_normalization import iterative_normalization_py
        X_hat = iterative_normalization_py.apply(
            input[0], module.running_mean, module.running_wm,
            module.num_channels, module.T, module.eps, module.momentum, module.training)
        N, C = X_hat.shape
        X_hat = X_hat.view(N, C) @ module.running_rot[0].T
        outputs.append(X_hat)

    handle = model.norm1.register_forward_hook(hook)

    activations = {}
    with torch.no_grad():
        for batch in concept_loader:
            batch = batch.to(device)
            outputs.clear()
            model(batch.x, batch.edge_index, batch.batch)
            G = global_mean_pool(outputs[0], batch.batch).cpu().numpy()
            for i, fp in enumerate(batch.file_path):
                activations[_concept_name(fp)] = G[i]

    handle.remove()
    return activations


def attach_activations(candidates, activations):
    for c in candidates:
        vec = activations.get(c['graph_concept_name'])
        if vec is not None:
            c['activation'] = vec
    return candidates


def select_precomputed(candidates, min_k, sim_fn=cosine_sim):
    n = len(candidates)
    if n <= 1:
        return candidates
    missing = [c['graph_concept_name'] for c in candidates if 'activation' not in c]
    if missing:
        raise ValueError(f"{len(missing)} candidate(s) missing precomputed 'activation' vectors")
    vecs = [c['activation'] for c in candidates]
    S = sim_matrix(vecs, sim_fn)
    return [candidates[i] for i in facility_location_from_matrix(S, budget(min_k, n))]
