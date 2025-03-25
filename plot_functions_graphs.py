import torch
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from numpy import linalg as LA


def concept_gradient_importance(model, test_loader, classes, concepts, class_index, device):

    model = model.to(device)

    outputs = []

    # Instead of taking the input of ReLU, we directly take the output of the cw layer
    def hook(module, grad_input, grad_output):
        outputs.append(grad_output[0])

    model.norm1.register_backward_hook(hook)

    class_count = [0] * len(classes)
    concept_importance_per_class = [None] * len(classes)

    for graph_batch_index, graph_batch in enumerate(test_loader):
        x_var = torch.autograd.Variable(graph_batch.x).cuda()
        edge_index_var = torch.autograd.Variable(graph_batch.edge_index).cuda()
        batch_var = torch.autograd.Variable(graph_batch.batch).cuda()
        target_var = torch.autograd.Variable(graph_batch.y).cuda()

        output = model(x_var, edge_index_var, batch_var)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1).flatten().tolist()[0]
        class_count[prediction_result] += 1
        output[:, prediction_result].backward()
        directional_derivatives = torch.unsqueeze(outputs[0], 0).mean(dim=1).flatten().cpu().numpy()
        is_positive = (directional_derivatives > 0).astype(np.int64)
        if concept_importance_per_class[prediction_result] is None:
            concept_importance_per_class[prediction_result] = is_positive
        else:
            concept_importance_per_class[prediction_result] += is_positive
        outputs = []

    for i in range(len(classes)):
        concept_importance_per_class[i] = concept_importance_per_class[i].astype(np.float32)
        concept_importance_per_class[i] /= class_count[i]
        # print(concept_importance_per_class[i])
        # print(concept_importance_per_class[i].mean())

    concepts_importances = concept_importance_per_class[class_index][:len(concepts)]

    return concepts_importances


def plot_concept_gradient_importance(aggregate_concept_importances, concepts, target_class):

    x = np.arange(len(concepts))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 1.5

    yticks = np.arange(1, step=0.1)

    fig, ax = plt.subplots(layout='constrained')

    for concept_type, concept_gradient_importance in aggregate_concept_importances.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, np.round(concept_gradient_importance, 2), width, label=concept_type)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Concept gradient importance')
    ax.set_title(f'Concept gradient importances for the {target_class} class')
    ax.set_xticks(x + width, concepts)
    ax.set_yticks(yticks)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.3)

    plt.show()


def intra_concept_dot_product_vs_inter_concept_dot_product(model, concept_loader, selected_concepts, all_concepts, device, whitening=True):

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
                x_var = torch.autograd.Variable(graph_batch.x).cuda()
                edge_index_var = torch.autograd.Variable(graph_batch.edge_index).cuda()
                batch_var = torch.autograd.Variable(graph_batch.batch).cuda()
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


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", concept_type=None, model_type=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_title(f'Normalized concept {concept_type} similarities for the {model_type} model')
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_concept_dot_product(dot_matrix, concepts, concept_type, model_type):
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots()

    im, cbar = heatmap(dot_matrix, concepts, concepts, ax=ax,
                       cmap="YlOrRd", cbarlabel="normalized concept similarity", concept_type=concept_type, model_type=model_type)
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()
