# — Standard library
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# — Third‑party
import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import seaborn as sns
import torch
from sklearn.metrics import f1_score
from torch_geometric.nn import global_mean_pool


def draw_graph(images_prefix: Union[str, Path], subgraph_path: Union[str, Path]):
    """
    Draw a graph from a GML file and save as PNG.

    Args:
        images_prefix: Directory where the output image should be saved
        subgraph_path: Path to the GML file containing the graph

    Returns:
        Path object pointing to the saved PNG file

    Raises:
        FileNotFoundError: If the subgraph_path doesn't exist
        ValueError: If the file cannot be loaded as a graph
    """
    # Convert to Path objects for better handling
    subgraph_path = Path(subgraph_path)
    images_prefix = Path(images_prefix)

    # Validate input file exists
    if not subgraph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {subgraph_path}")

    # Create output directory if it doesn't exist
    images_prefix.mkdir(parents=True, exist_ok=True)

    try:
        # Load the graph from GML file
        graph = nx.read_gml(subgraph_path)

        # Convert to AGraph for drawing
        dot_graph = nx.nx_agraph.to_agraph(graph)
        dot_graph.layout(prog="dot")

        # Create output filename (remove .gml extension, add .png)
        base_name = subgraph_path.stem  # Gets filename without extension
        output_filename = f"{base_name}.png"
        output_path = images_prefix / output_filename

        # Draw and save the graph
        dot_graph.draw(str(output_path))

    except Exception as e:
        raise ValueError(f"Failed to process graph file {subgraph_path}: {str(e)}")


def concept_gradient_importance(model, test_loader, classes, concepts, class_index, device):

    model = model.to(device)
    model.eval()

    outputs = []

    # Instead of taking the input of ReLU, we directly take the output of the cw layer
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
        # print(concept_importance_per_class[i])
        # print(concept_importance_per_class[i].mean())

    concepts_importances = concept_importance_per_class[class_index][:len(concepts)]

    return concepts_importances


def plot_concept_gradient_importance(
    aggregate_concept_importances: Dict[str, np.ndarray],
    concepts: List[str],
    target_class: str,
    dataset: str,
    graph_conv_type: str,
    graph_residual_connections: Union[str, bool],
    images_prefix: Union[str, Path],
    figsize: tuple = (14, 8),
    dpi: int = 600,
    show_plot: bool = True,
    ylim: Optional[tuple] = None
):
    """
    Plot concept gradient importance as grouped bar chart.

    Args:
        aggregate_concept_importances: Dict mapping concept types to importance arrays
        concepts: List of concept names
        target_class: Target class name for the plot
        dataset: Dataset name
        graph_conv_type: Graph convolution type
        graph_residual_connections: Whether residual connections are used
        images_prefix: Directory path to save the image
        figsize: Figure size tuple (width, height)
        dpi: DPI for saved image
        show_plot: Whether to display the plot
        ylim: Y-axis limits tuple (min, max). If None, uses (0, 1.5)

    Returns:
        Path object of the saved image file

    Raises:
        ValueError: If concepts and importance arrays have mismatched lengths
    """

    if dataset == 'ten_newsgroups':
        dataset = '10 Newsgroups'

    elif dataset == 'bbcsport':
        dataset = 'BBC Sport'

    # Input validation
    if not aggregate_concept_importances:
        raise ValueError("aggregate_concept_importances cannot be empty")

    for concept_type, importances in aggregate_concept_importances.items():
        if len(importances) != len(concepts):
            raise ValueError(f"Length mismatch: {len(concepts)} concepts but "
                           f"{len(importances)} importances for {concept_type}")

    font_config = {'title': {'family': 'serif', 'name': 'Times New Roman', 'weight': 'bold'},
        'axes_labels': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'medium'},
        'tick_labels': {'family': 'monospace', 'name': 'Consolas', 'weight': 'normal'},
        'bar_labels': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'bold'},
        'legend': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'medium'}}

    # Setup
    images_prefix = Path(images_prefix)
    images_prefix.mkdir(parents=True, exist_ok=True)

    n_concepts = len(concepts)
    n_concept_types = len(aggregate_concept_importances)
    x = np.arange(n_concepts)

    # Dynamic bar width based on number of concept types
    total_width = 1.0  # Total width for all bars per concept
    width = total_width / n_concept_types

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each concept type
    colors = plt.cm.Set3(np.linspace(0, 1, n_concept_types))  # Use colormap for consistency

    for i, (concept_type, concept_gradient_importance) in enumerate(aggregate_concept_importances.items()):
        # Center the bars around each x position
        offset = (i - (n_concept_types - 1) / 2) * width

        rects = ax.bar(
            x + offset,
            np.round(concept_gradient_importance, 3),  # More precision
            width,
            label=concept_type,
            color=colors[i],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Add value labels on bars
        ax.bar_label(rects, padding=3, fontsize=12, rotation=0,
                     fontfamily=font_config['bar_labels']['family'],
                     fontweight=font_config['bar_labels']['weight'])

    # Styling
    ax.set_ylabel('Concept Gradient Importance', fontsize=16,
                  fontfamily=font_config['axes_labels']['family'],
                  fontweight=font_config['axes_labels']['weight'])

    ax.set_xlabel('Concepts', fontsize=16,
                  fontfamily=font_config['axes_labels']['family'],
                  fontweight=font_config['axes_labels']['weight'])

    ax.set_title(f'Concept Gradient Importances for Class "{target_class}"\n'
                 f'({graph_conv_type}, Residuals: {graph_residual_connections}, Dataset: {dataset})',
                 fontsize=18, pad=20, fontfamily=font_config['title']['family'], fontweight=font_config['title']['weight'])

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, fontsize=13, rotation=45, ha='right',
                       fontfamily=font_config['tick_labels']['family'],
                       fontweight=font_config['tick_labels']['weight'])

    # Y-axis
    if ylim is None:
        ylim = (0, max(1.5, max(max(importances) for importances in aggregate_concept_importances.values()) * 1.1))
    ax.set_ylim(ylim)

    # Set y-tick font
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_config['tick_labels']['family'])
        label.set_fontweight(font_config['tick_labels']['weight'])
        label.set_fontsize(13)

    # Grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')

    # Add vertical grid lines after each concept group
    for i in range(len(concepts)):
        if i < len(concepts) - 1:  # Don't add line after the last concept
            # Position line halfway between current concept and next concept
            line_position = x[i] + (x[i + 1] - x[i]) / 2
            ax.axvline(x=line_position, color='gray', alpha=0.3, linestyle='--', linewidth=0.8)

    ax.set_axisbelow(True)

    # Legend with custom font
    legend = ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily(font_config['legend']['family'])
        text.set_fontweight(font_config['legend']['weight'])

    # Tight layout
    plt.tight_layout()

    # Save figure
    image_filename = f"{dataset}_{target_class}_{graph_conv_type}_{graph_residual_connections}.png"
    image_path = images_prefix / image_filename

    plt.savefig(image_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
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
                    from iterative_normalization import iterative_normalization_py
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


def compute_axis_alignment_accuracy_and_f1(model, device, loader, orig_label: int,
                                           axis_index: int, whitening: bool = True) -> (float, float):
    """
    Returns (f1) for “among graphs with true label == orig_label,
    how many have their maximal‐activation axis == axis_index?”
    """

    model = model.to(device)
    model.eval()

    outputs = []
    all_X, all_y = [], []

    def hook(module, input, output):
        if not whitening:
            X_hat = output
        else:
            from iterative_normalization import iterative_normalization_py

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

            X = outputs[0]  # (total_nodes, C)
            G = global_mean_pool(X, batch.batch)  # (batch_size, C)

            # **move to CPU numpy before storing**
            all_X.append(G.cpu().numpy())
            all_y.append(batch.y.view(-1).cpu().numpy())

    handle.remove()

    X_all = np.vstack(all_X)  # (n_graphs, C)
    y_all = np.hstack(all_y)  # (n_graphs,)

    # pick only those graphs whose true label matches this concept
    mask = (y_all == orig_label)
    if not mask.any():
        return float('nan'), float('nan')

    pred_axes = X_all.argmax(axis=1)  # (n_graphs,)

    # F1
    y_true = mask
    y_pred = (pred_axes == axis_index)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return f1


def extract_top_activation_subgraphs_per_concept(model, device, loader, axis_index: int) -> Tuple[str, float]:
    """
    Returns (best_file_name, best_activation_value) for the single
    graph in this concept‐loader that attains the highest mean activation
    along channel `axis_index` of your CW layer.
    """

    model = model.to(device)
    model.eval()

    outputs = []

    def hook(module, input, output):
        from iterative_normalization import iterative_normalization_py

        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm,
                                                 module.num_channels, module.T, module.eps, module.momentum,
                                                 module.training)

        N, C = X_hat.shape
        X_hat = X_hat.view(N, C) @ module.running_rot[0].T
        outputs.append(X_hat)

    handle = model.norm1.register_forward_hook(hook)

    best_val = -float("inf")
    best_file = None

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs.clear()
            model(batch.x, batch.edge_index, batch.batch)
            X = outputs[0]                     # [N_nodes, C]
            G = global_mean_pool(X, batch.batch)  # [B, C]
            axis_vals = G[:, axis_index].cpu().numpy()  # length B
            fpnames = batch.file_path  # now a list of length B

            for i, val in enumerate(axis_vals):
                if val > best_val:
                    best_val = val
                    best_file_path = fpnames[i]

    handle.remove()
    return best_file_path, float(best_val)

def create_elegant_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    ax=None,
    cbar_kw: Optional[dict] = None,
    cbarlabel: str = "",
    concept_type: Optional[str] = None,
    model_type: Optional[str] = None,
    dataset: Optional[str] = None,
    graph_conv_type: Optional[str] = None,
    graph_residual_connections: Optional[Union[str, bool]] = None,
    color_palette: str = "elegant_blue",
    font_config: Optional[dict] = None,
    **kwargs
):
    """
    Create an elegant, publication-ready heatmap with beautiful color palettes.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array of shape (M, N).
    row_labels : List[str]
        Labels for the rows.
    col_labels : List[str]
        Labels for the columns.
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on.
    cbar_kw : dict, optional
        Arguments for colorbar.
    cbarlabel : str
        Label for the colorbar.
    concept_type, model_type, dataset, graph_conv_type, graph_residual_connections : str
        Parameters for title generation.
    color_palette : str
        Color palette name. Options: 'elegant_blue', 'warm_sunset', 'cool_viridis',
        'academic_sequential', 'diverging_elegant', 'monochrome_elegant'
    font_config : dict, optional
        Font configuration for different elements.
    **kwargs
        Additional arguments for imshow.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The heatmap image.
    cbar : matplotlib.colorbar.Colorbar
        The colorbar.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Define elegant color palettes
    color_palettes = {
        'elegant_blue': 'Blues',
        'warm_sunset': sns.blend_palette(['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f'], as_cmap=True),
        'cool_viridis': 'viridis',
        'academic_sequential': sns.blend_palette(['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'], as_cmap=True),
        'diverging_elegant': sns.diverging_palette(240, 10, n=256, as_cmap=True),
        'monochrome_elegant': sns.blend_palette(['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525'], as_cmap=True),
        'nature_inspired': sns.blend_palette(['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'], as_cmap=True),
        'coral_gradient': sns.blend_palette(['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'], as_cmap=True),
        'sunset_fire': sns.blend_palette(['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000'], as_cmap=True),
        'orange_purple_fusion': sns.blend_palette(['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000'], as_cmap=True),
        'fire_orchid': sns.blend_palette(['#ffeee6', '#ffdbcc', '#ffb380', '#ff8c42', '#e8684a', '#cc4452', '#9d2560', '#6b0848'], as_cmap=True),
        'sage_blue': sns.blend_palette(['#f8fbf8', '#e8f4f0', '#d1e7dd', '#b3d9c7', '#8cc8aa', '#5fb3a3', '#4a9d94', '#2e8b7b', '#1e5f5a'], as_cmap=True),
        'storm_grey': sns.blend_palette(['#f7f9fb', '#e9f1f5', '#d4e4ea', '#b8d4dd', '#96c0ce', '#6fa8ba', '#4d8fa3', '#2c758a', '#1a5c6e'], as_cmap=True),
        'arctic_blue': sns.blend_palette(['#fafcfd', '#f0f6f8', '#e1eef3', '#cde3ed', '#b3d5e5', '#94c5db', '#6fb2ce', '#4a9dbd', '#2e86a8'], as_cmap=True)
    }

    # Configure fonts
    if font_config is None:
        font_config = {
            'title': {'family': 'serif', 'name': 'Times New Roman', 'weight': 'bold', 'size': 18},
            'labels': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'medium', 'size': 16},
            'ticks': {'family': 'monospace', 'name': 'Consolas', 'weight': 'normal', 'size': 15},
            'colorbar': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'medium', 'size': 15},
            'annotations': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'bold', 'size': 14}  # Larger default
        }

    # Get colormap
    if color_palette in color_palettes:
        cmap = color_palettes[color_palette]
    else:
        cmap = color_palette  # Assume it's a valid matplotlib colormap name

    # Set default kwargs
    default_kwargs = {
        'cmap': cmap,
        'aspect': 'auto',
        'interpolation': 'nearest',
        'vmin': 0,  # Fixed minimum value
        'vmax': 1   # Fixed maximum value
    }

    default_kwargs.update(kwargs)

    # Plot the heatmap
    im = ax.imshow(data, **default_kwargs)

    # Create elegant colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)

    # Set fixed colorbar ticks from 0 to 1 with 0.2 intervals
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    # Style colorbar
    cbar.ax.set_ylabel(
        cbarlabel,
        rotation=-90,
        va="bottom",
        fontfamily=font_config['colorbar']['family'],
        fontweight=font_config['colorbar']['weight'],
        fontsize=font_config['colorbar']['size']
    )

    # Style colorbar tick labels
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(font_config['ticks']['family'])
        label.set_fontsize(font_config['ticks']['size'])

    # Set ticks and labels with elegant styling
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(
        col_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontfamily=font_config['ticks']['family'],
        fontsize=font_config['ticks']['size']
    )

    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(
        row_labels,
        fontfamily=font_config['ticks']['family'],
        fontsize=font_config['ticks']['size']
    )

    # Position labels elegantly
    ax.tick_params(
        top=False, bottom=True,
        labeltop=False, labelbottom=True,
        length=0  # Remove tick marks for cleaner look
    )

    # Remove spines for cleaner appearance
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create elegant title
    if all(param is not None for param in [concept_type, dataset, model_type, graph_conv_type, graph_residual_connections]):
        # Clean up dataset name
        if dataset == 'ten_newsgroups':
            dataset = '10 Newsgroups'
        elif dataset == 'bbcsport':
            dataset = 'BBC Sport'

        title = (f"{concept_type.title()} Concept Similarities\n"
                f"{dataset} Dataset ({model_type}, {graph_conv_type}, "
                f"Residuals: {graph_residual_connections})")

        ax.set_title(
            title,
            fontfamily=font_config['title']['family'],
            fontweight=font_config['title']['weight'],
            fontsize=font_config['title']['size'],
            pad=20
        )

    # Add subtle grid
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_elegant_heatmap(
    im,
    data: Optional[np.ndarray] = None,
    valfmt: str = "{x:.2f}",
    textcolors: Tuple[str, str] = ("black", "white"),
    threshold: Optional[float] = None,
    font_config: Optional[dict] = None,
    adaptive_text_size: bool = True,
    base_text_size: Optional[float] = None,
    **textkw
):
    """
    Annotate a heatmap with elegant styling and adaptive text sizing.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The heatmap image to annotate.
    data : np.ndarray, optional
        Data to annotate. If None, uses image data.
    valfmt : str
        Format string for annotations.
    textcolors : tuple
        Colors for text (below_threshold, above_threshold).
    threshold : float, optional
        Threshold for color switching.
    font_config : dict, optional
        Font configuration.
    adaptive_text_size : bool
        Whether to automatically adapt text size to cell size.
    base_text_size : float, optional
        Base text size. If None, calculates automatically.
    **textkw
        Additional text properties.

    Returns
    -------
    texts : list
        List of text objects.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Set font configuration
    if font_config is None:
        font_config = {'annotations': {'family': 'sans-serif', 'name': 'Helvetica', 'weight': 'bold', 'size': 14}}

    # Calculate adaptive text size based on figure and data dimensions
    if adaptive_text_size:
        # Get figure size and data dimensions
        fig = im.axes.figure
        fig_width, fig_height = fig.get_size_inches()
        data_rows, data_cols = data.shape

        # Calculate cell size in figure coordinates
        cell_width = fig_width / data_cols
        cell_height = fig_height / data_rows
        avg_cell_size = (cell_width + cell_height) / 2

        # Adaptive text size: scale with cell size, with reasonable bounds
        if base_text_size is None:
            adaptive_size = max(8, min(20, avg_cell_size * 8))  # Scale factor of 8
        else:
            adaptive_size = base_text_size * (avg_cell_size / 1.5)  # Scale relative to base

        text_size = adaptive_size
    else:
        # Use configured or provided text size
        text_size = base_text_size if base_text_size else font_config['annotations']['size']

    # Normalize threshold
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Default text properties
    kw = dict(
        horizontalalignment="center",
        verticalalignment="center",
        fontfamily=font_config['annotations']['family'],
        fontweight=font_config['annotations']['weight'],
        fontsize=text_size
    )
    kw.update(textkw)

    # Format annotations
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Create annotations
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Choose text color based on background
            text_color = textcolors[int(im.norm(data[i, j]) > threshold)]
            kw.update(color=text_color)

            # Add text
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_elegant_concept_similarities(
    dot_matrix: np.ndarray,
    concepts: List[str],
    concept_type: str,
    model_type: str,
    dataset: str,
    graph_conv_type: str,
    graph_residual_connections: Union[str, bool],
    images_prefix: Union[str, Path],
    color_palette: str = "sage_blue",
    figsize: Tuple[float, float] = (12, 12),
    dpi: int = 600,
    show_annotations: bool = True,
    adaptive_text_size: bool = True,
    annotation_text_size: Optional[float] = None,
    show_plot: bool = True
):
    """
    Create an elegant concept similarity heatmap.

    Parameters
    ----------
    dot_matrix : np.ndarray
        The similarity matrix to visualize.
    concepts : List[str]
        List of concept names.
    concept_type : str
        Type of concepts being compared.
    model_type : str
        Model type used.
    dataset : str
        Dataset name.
    graph_conv_type : str
        Graph convolution type.
    graph_residual_connections : Union[str, bool]
        Whether residual connections are used.
    images_prefix : Union[str, Path]
        Directory to save the image.
    color_palette : str
        Color palette to use.
    figsize : Tuple[float, float]
        Figure size.
    dpi : int
        Image resolution.
    show_annotations : bool
        Whether to show value annotations.
    adaptive_text_size : bool
        Whether to automatically adapt annotation text size to cell size.
    annotation_text_size : float, optional
        Manual text size for annotations. If None and adaptive_text_size=False, uses default.
    show_plot : bool
        Whether to display the plot.

    Returns
    -------
    Path
        Path to the saved image.
    """

    # Set up the plot with elegant styling
    plt.style.use('default')  # Reset any previous styling

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im, cbar = create_elegant_heatmap(
        dot_matrix, concepts, concepts, ax=ax,
        cbarlabel="Concept Similarity",
        concept_type=concept_type,
        model_type=model_type,
        dataset=dataset,
        graph_conv_type=graph_conv_type,
        graph_residual_connections=graph_residual_connections,
        color_palette=color_palette
    )

    # Add annotations if requested
    if show_annotations:
        texts = annotate_elegant_heatmap(
            im,
            valfmt="{x:.2f}",
            adaptive_text_size=adaptive_text_size,
            base_text_size=annotation_text_size
        )

    # Adjust layout
    plt.tight_layout(pad=1.0)

    # Save the figure
    images_prefix = Path(images_prefix)
    images_prefix.mkdir(parents=True, exist_ok=True)

    # Create filename
    image_filename = f"{dataset}_{graph_conv_type}_{graph_residual_connections}_{concept_type}_{model_type}.png"
    image_path = images_prefix / image_filename

    # Save with high quality
    plt.savefig(
        image_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none'
    )

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Elegant heatmap saved to: {image_path}")