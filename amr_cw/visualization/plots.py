from pathlib import Path

import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.metrics import f1_score
from torch_geometric.nn import global_mean_pool


def draw_graph(images_prefix, subgraph_path):
    subgraph_path = Path(subgraph_path)
    images_prefix = Path(images_prefix)
    images_prefix.mkdir(parents=True, exist_ok=True)
    graph = nx.read_gml(subgraph_path)
    dot_graph = nx.nx_agraph.to_agraph(graph)
    dot_graph.layout(prog="dot")
    output_path = images_prefix / f"{subgraph_path.stem}.png"
    dot_graph.draw(str(output_path))


def concept_gradient_importance(model, test_loader, classes, concepts, class_index, device):
    model = model.to(device)
    model.eval()

    outputs = []

    def hook(module, grad_input, grad_output):
        outputs.append(grad_output[0])

    model.norm1.register_backward_hook(hook)

    class_count = [0] * len(classes)
    concept_importance_per_class = [None] * len(classes)

    for graph_batch_index, graph_batch in enumerate(test_loader):
        x_var = torch.autograd.Variable(graph_batch.x).to(device)
        edge_index_var = torch.autograd.Variable(graph_batch.edge_index).to(device)
        batch_var = torch.autograd.Variable(graph_batch.batch).to(device)
        target_var = torch.autograd.Variable(graph_batch.y).to(device)

        output = model(x_var, edge_index_var, batch_var)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1)

        for i in range(len(prediction_result)):
            class_id = prediction_result[i].item()
            class_count[class_id] += 1

            torch.autograd.set_detect_anomaly(True)
            output[i, class_id].backward(retain_graph=True)

            directional_derivatives = torch.unsqueeze(outputs[0], 0).mean(dim=1).flatten().cpu().numpy()
            is_positive = (directional_derivatives > 0).astype(np.int64)

            if concept_importance_per_class[class_id] is None:
                concept_importance_per_class[class_id] = is_positive
            else:
                concept_importance_per_class[class_id] += is_positive

            outputs = []
            model.zero_grad()

    for i in range(len(classes)):
        if concept_importance_per_class[i] is None:
            concept_importance_per_class[i] = np.zeros(len(concepts), dtype=np.float32)
        else:
            concept_importance_per_class[i] = concept_importance_per_class[i].astype(np.float32)
            concept_importance_per_class[i] /= class_count[i]

    concepts_importances = concept_importance_per_class[class_index][:len(concepts)]

    return concepts_importances


def plot_concept_gradient_importance(aggregate_concept_importances, concepts, target_class, dataset,
                                     graph_conv_type, graph_residual_connections, images_prefix):
    if dataset == 'ten_newsgroups':
        dataset = '10 Newsgroups'
    elif dataset == 'bbcsport':
        dataset = 'BBC Sport'

    images_prefix = Path(images_prefix)
    images_prefix.mkdir(parents=True, exist_ok=True)

    n_concept_types = len(aggregate_concept_importances)
    x = np.arange(len(concepts))
    width = 1.0 / n_concept_types

    fig, ax = plt.subplots()
    for i, (concept_type, importances) in enumerate(aggregate_concept_importances.items()):
        offset = (i - (n_concept_types - 1) / 2) * width
        rects = ax.bar(x + offset, np.round(importances, 3), width, label=concept_type)
        ax.bar_label(rects, padding=3)

    ax.set_ylabel('Concept Gradient Importance')
    ax.set_xlabel('Concepts')
    ax.set_title(f'Concept Gradient Importances for Class "{target_class}"\n'
                 f'({graph_conv_type}, Residuals: {graph_residual_connections}, Dataset: {dataset})')
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.legend()

    plt.tight_layout()
    image_filename = f"{dataset}_{target_class}_{graph_conv_type}_{graph_residual_connections}.png"
    plt.savefig(images_prefix / image_filename, bbox_inches='tight')
    plt.close()


def intra_concept_dot_product_vs_inter_concept_dot_product(model, concept_loader, selected_concepts, all_concepts,
                                                           device, whitening=True):
    model = model.to(device)
    model.eval()

    representations = {}

    for concept in selected_concepts:
        representations[concept] = []

    for c_idx, concept in enumerate(selected_concepts):
        with torch.no_grad():
            outputs = []

            def hook(module, input, output):
                if not whitening:
                    outputs.append(output.cpu().numpy())
                else:
                    from amr_cw.core.iterative_normalization import iterative_normalization_py
                    X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm,
                                                             module.num_channels, module.T, module.eps,
                                                             module.momentum, module.training)
                    size_X = X_hat.size()
                    size_R = module.running_rot.size()
                    X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])
                    X_hat = torch.einsum('bgc,gdc->bgd', X_hat, module.running_rot)
                    X_hat = X_hat.view(*size_X)

                    outputs.append(X_hat.cpu().numpy())

            model.norm1.register_forward_hook(hook)

            print(c_idx, concept)

            for j, graph_batch in enumerate(concept_loader):
                x_var = torch.autograd.Variable(graph_batch.x).to(device)
                edge_index_var = torch.autograd.Variable(graph_batch.edge_index).to(device)
                batch_var = torch.autograd.Variable(graph_batch.batch).to(device)
                labels = graph_batch.y.cpu().numpy().flatten().astype(np.int32).tolist()

                outputs = []
                model(x_var, edge_index_var, batch_var)

                for instance_index in range(len(labels)):
                    instance_concept_index = labels[instance_index]

                    if all_concepts[instance_concept_index] == concept:
                        representation_mean = outputs[0].mean(axis=0)
                        representations[concept].append(representation_mean)

    dot_product_matrix = np.zeros((len(selected_concepts), len(selected_concepts))).astype('float')
    m_representations = {}
    m_representations_normed = {}
    intra_dot_product_means = {}
    intra_dot_product_means_normed = {}

    for i, concept in enumerate(selected_concepts):
        m_representations[concept] = np.stack(representations[concept], axis=0)
        m_representations_normed[concept] = m_representations[concept]/LA.norm(m_representations[concept], axis=1, keepdims=True)
        intra_dot_product_means[concept] = np.matmul(m_representations[concept], m_representations[concept].transpose()).mean()
        intra_dot_product_means_normed[concept] = np.matmul(m_representations_normed[concept], m_representations_normed[concept].transpose()).mean()
        dot_product_matrix[i, i] = 1.0

    inter_dot_product_means = {}
    inter_dot_product_means_normed = {}

    for i in range(len(selected_concepts)):
        for j in range(i + 1, len(selected_concepts)):
            cpt_1 = selected_concepts[i]
            cpt_2 = selected_concepts[j]
            inter_dot_product_means[cpt_1 + '_' + cpt_2] = np.matmul(m_representations[cpt_1], m_representations[cpt_2].transpose()).mean()
            inter_dot_product_means_normed[cpt_1 + '_' + cpt_2] = np.matmul(m_representations_normed[cpt_1], m_representations_normed[cpt_2].transpose()).mean()
            dot_product_matrix[i, j] = abs(inter_dot_product_means_normed[cpt_1 + '_' + cpt_2]) / np.sqrt(abs(intra_dot_product_means_normed[cpt_1] * intra_dot_product_means_normed[cpt_2]))
            dot_product_matrix[j, i] = dot_product_matrix[i, j]

    return dot_product_matrix


def compute_axis_alignment_accuracy_and_f1(model, device, loader, orig_label, axis_index, whitening=True):
    model = model.to(device)
    model.eval()

    outputs = []
    all_X, all_y = [], []

    def hook(module, input, output):
        if not whitening:
            X_hat = output
        else:
            from amr_cw.core.iterative_normalization import iterative_normalization_py

            X_hat = iterative_normalization_py.apply(input[0],module.running_mean, module.running_wm,
                                                     module.num_channels, module.T, module.eps, module.momentum,
                                                     module.training)
            N, C = X_hat.shape
            X_hat = X_hat.view(N, C) @ module.running_rot[0].T

        outputs.append(X_hat)

    handle = model.norm1.register_forward_hook(hook)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs.clear()

            model(batch.x, batch.edge_index, batch.batch)

            X = outputs[0]
            G = global_mean_pool(X, batch.batch)

            all_X.append(G.cpu().numpy())
            all_y.append(batch.y.view(-1).cpu().numpy())

    handle.remove()

    X_all = np.vstack(all_X)
    y_all = np.hstack(all_y)

    mask = (y_all == orig_label)
    if not mask.any():
        return float('nan')

    pred_axes = X_all.argmax(axis=1)

    y_true = mask
    y_pred = (pred_axes == axis_index)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return f1


def extract_top_activation_subgraphs_per_concept(model, device, loader, axis_index):
    model = model.to(device)
    model.eval()

    outputs = []

    def hook(module, input, output):
        from amr_cw.core.iterative_normalization import iterative_normalization_py

        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm,
                                                 module.num_channels, module.T, module.eps, module.momentum,
                                                 module.training)

        N, C = X_hat.shape
        X_hat = X_hat.view(N, C) @ module.running_rot[0].T
        outputs.append(X_hat)

    handle = model.norm1.register_forward_hook(hook)

    best_val = -float("inf")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs.clear()
            model(batch.x, batch.edge_index, batch.batch)
            X = outputs[0]
            G = global_mean_pool(X, batch.batch)
            axis_vals = G[:, axis_index].cpu().numpy()
            fpnames = batch.file_path

            for i, val in enumerate(axis_vals):
                if val > best_val:
                    best_val = val
                    best_file_path = fpnames[i]

    handle.remove()
    return best_file_path, float(best_val)


def create_elegant_heatmap(data, row_labels, col_labels, ax=None, cbarlabel=""):
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(data, vmin=0, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(row_labels)

    return im, cbar


def annotate_elegant_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = textcolors[int(im.norm(data[i, j]) > threshold)]
            text = im.axes.text(j, i, valfmt(data[i, j], None), ha="center", va="center", color=color)
            texts.append(text)

    return texts


def plot_elegant_concept_similarities(dot_matrix, concepts, concept_type, model_type, dataset,
                                      graph_conv_type, graph_residual_connections, images_prefix):
    fig, ax = plt.subplots()

    im, cbar = create_elegant_heatmap(dot_matrix, concepts, concepts, ax=ax, cbarlabel="Concept Similarity")
    annotate_elegant_heatmap(im, valfmt="{x:.2f}")

    title_dataset = dataset
    if dataset == 'ten_newsgroups':
        title_dataset = '10 Newsgroups'
    elif dataset == 'bbcsport':
        title_dataset = 'BBC Sport'
    ax.set_title(f"{concept_type.title()} Concept Similarities\n"
                 f"{title_dataset} Dataset ({model_type}, {graph_conv_type}, "
                 f"Residuals: {graph_residual_connections})")

    plt.tight_layout()

    images_prefix = Path(images_prefix)
    images_prefix.mkdir(parents=True, exist_ok=True)
    image_path = images_prefix / f"{dataset}_{graph_conv_type}_{graph_residual_connections}_{concept_type}_{model_type}.png"
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

    print(f"saved heatmap to {image_path}")
