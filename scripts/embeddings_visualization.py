import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from text_dataset import TextDataset
from transformers import BertTokenizer
from text_models import BERTTextClassifier
from graph_models import GCN
from sklearn.manifold import TSNE
from graph_dataset import GraphDataset
from hybrid_dataset import HybridDataset
from hybrid_models import HybridModel


def get_model_and_device(dataset, model_path=None, mode='text_embeddings', pretrained=False):

    if mode == 'text_embeddings':
        if model_path is None and pretrained is False:
            model = BERTTextClassifier(dataset)

        else:
            model = BERTTextClassifier(dataset)
            model.load_state_dict(torch.load(model_path))

    elif mode == 'graph_embeddings':
        if model_path is None and pretrained is False:
            model = GCN(dataset, hidden_channels=64)

        else:
            model = GCN(dataset, hidden_channels=64)
            model.load_state_dict(torch.load(model_path))

    elif mode == 'hybrid_embeddings':
        if model_path is None and pretrained is False:
            model = HybridModel(dataset)

        else:
            model = HybridModel(dataset)
            model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders(dataset, classes, data_prefix, train_batch_size, test_batch_size, shuffle=True, mode='text_embeddings'):

    if mode == 'text_embeddings':
        raw_texts_csv_prefix = data_prefix + '/' + dataset + '_raw_texts.csv'
        df = pd.read_csv(raw_texts_csv_prefix)

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        texts_train_df = df[df['split'] == 'train']
        train_dataset = TextDataset(texts_train_df, tokenizer, 512)

        texts_test_df = df[df['split'] == 'test']
        test_dataset = TextDataset(texts_test_df, tokenizer, 512)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle)

    elif mode == 'graph_embeddings':
        train_dataset = GraphDataset(root=data_prefix, labels=classes, dataset=dataset)
        test_dataset = GraphDataset(root=data_prefix, labels=classes, dataset=dataset, test=True)

        train_loader = GeometricDataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
        test_loader = GeometricDataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle)

    elif mode == 'hybrid_embeddings':
        hybrid_embeddings_file_name = data_prefix + "/" + f"{dataset}_hybrid_embeddings.json"

        with open(hybrid_embeddings_file_name) as f:
            d = json.load(f)

        df = pd.DataFrame.from_dict(d)

        hybrids_train_df = df[df['split'] == 'train']
        train_dataset = HybridDataset(hybrids_train_df)

        hybrids_test_df = df[df['split'] == 'test']
        test_dataset = HybridDataset(hybrids_test_df)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle)

    return train_loader, test_loader


def get_plot_data(model, loader, device, mode='text_embeddings'):

    y_pred = []
    y_true = []
    embeddings = []

    model = model.to(device)
    model.eval()

    if mode == 'text_embeddings':
        for data in loader:
            out = model(data['input_ids'].to(device), data['attention_mask'].to(device))
            y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
            y_true.extend(data['targets'].to(device).cpu().detach().numpy())
            text_embeddings = model.get_text_embeddings(data['input_ids'].to(device), data['attention_mask'].to(device))
            embeddings.extend(np.array(text_embeddings.to(device).cpu().detach().numpy()).tolist())

    elif mode == 'graph_embeddings':
        for data in loader:
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
            y_true.extend(data.y.to(device).cpu().detach().numpy())
            graph_embeddings = model.get_graph_embeddings(data.x.to(device), data.edge_index.to(device),
                                                          data.batch.to(device))
            embeddings.extend(np.array(graph_embeddings.to(device).cpu().detach().numpy()).tolist())

    elif mode == 'hybrid_embeddings':
        for data in loader:
            out = model(data['hybrid_embeddings'].to(device))
            y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
            y_true.extend(data['targets'].to(device).cpu().detach().numpy())
            hybrid_embeddings = model.get_hybrid_embeddings(data['hybrid_embeddings'].to(device))
            embeddings.extend(np.array(hybrid_embeddings.to(device).cpu().detach().numpy()).tolist())

    return y_pred, y_true, embeddings


def plot_t_sne(dataset, classes, embeddings, y_pred, mode):

    mode_string = mode.replace("_embeddings", "")
    dataset_string = dataset.replace("_", " ")

    title_message = f"{mode_string} embeddings t-SNE for test documents belonging to various classes of the {dataset_string} dataset"
    x_label_message = f"{mode_string} embeddings t-SNE component 1"
    y_label_message = f"{mode_string} embeddings t-SNE component 2"

    embeddings_t_sne = TSNE(n_components=2, perplexity=3).fit_transform(np.array(embeddings))

    sizes = [100 for i in range(len(y_pred))]

    scatter = plt.scatter(embeddings_t_sne[:, 0], embeddings_t_sne[:, 1], s=sizes, c=y_pred, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=12)
    plt.title(title_message, fontsize=15)
    plt.xlabel(x_label_message, fontsize=15)
    plt.ylabel(y_label_message, fontsize=15)
    plt.show()


def inference(dataset, classes, data_prefix, train_batch_size, test_batch_size, model_path, mode):

    train_loader, test_loader = get_loaders(dataset, classes, data_prefix, train_batch_size, test_batch_size,
                                            shuffle=True, mode=mode)
    model, device = get_model_and_device(test_loader.dataset, model_path, mode, pretrained=True)

    y_pred, y_true, embeddings = get_plot_data(model, test_loader, device, mode)

    plot_t_sne(dataset, classes, embeddings, y_pred, mode)


def visualize_embeddings(dataset, classes, text_data_prefix, graph_data_prefix, hybrid_data_prefix,
                         text_model_path, graph_model_path, hybrid_model_path,
                         mode='text_embeddings'):

    if mode == 'text_embeddings':
        inference(dataset, classes, text_data_prefix, 8, 4, text_model_path, mode='text_embeddings')

    elif mode == 'graph_embeddings':
        inference(dataset, classes, graph_data_prefix, 512, 64, graph_model_path, mode='graph_embeddings')

    elif mode == 'hybrid_embeddings':
        inference(dataset, classes, hybrid_data_prefix, 512, 64, hybrid_model_path, mode='hybrid_embeddings')
