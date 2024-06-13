import torch
import numpy as np
import matplotlib.pyplot as plt
from models import GCN_5
from itertools import chain
from sklearn.manifold import TSNE
from example_dataset_2 import ExampleDataset2
from matplotlib.colors import ListedColormap
from torch_geometric.loader import DataLoader


def get_model_and_device(dataset, pretrained=True):
    model = GCN_5(dataset, hidden_channels=64, pretrained=pretrained)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders():
    test_dataset = ExampleDataset2(root='example_data_2/', test=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    return test_loader


def evaluate():
    loader = get_loaders()
    model, device = get_model_and_device(loader.dataset, pretrained=True)

    model = model.to(device)
    model.eval()

    correct = 0

    y_pred = []
    y_true = []
    graph_embeddings = []

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        pred = out.argmax(dim=1)

        graph_embedding = model.get_graph_embeddings(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        graph_embeddings.append(graph_embedding.detach().cpu().numpy().tolist())

        y_pred.append(pred.detach().cpu().numpy().tolist())
        y_true.append(data.y.detach().cpu().numpy().tolist())
        correct += int((pred == data.y.to(device)).sum())

    print(f"Accuracy: {correct / len(loader.dataset)}")

    return list(chain.from_iterable(y_pred)), list(chain.from_iterable(y_true)), list(chain.from_iterable(graph_embeddings))


def plot_t_sne(graph_embeddings, predictions):

    classes = ['athletics', 'cricket', 'football', 'rugby', 'tennis']

    colors = ListedColormap(["blue", "orange", "green", "red", "purple"])

    graph_t_sne = TSNE(n_components=2, perplexity=3).fit_transform(graph_embeddings)

    sizes = [100 for i in range(len(predictions))]

    scatter = plt.scatter(graph_t_sne[:, 0], graph_t_sne[:, 1], s=sizes, c=predictions, cmap=colors)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=12)
    plt.title("Graph t-SNE for test document graphs belonging to various classes of the BBC sports dataset.", fontsize=15)
    plt.xlabel("Graph t-SNE component 1", fontsize=15)
    plt.ylabel("Graph t-SNE component 2", fontsize=15)
    plt.show()


if __name__ == "__main__":
    predictions, ground_truth, embeddings = evaluate()
    print(predictions)
    print(ground_truth)

    plot_t_sne(np.array(embeddings), predictions)
