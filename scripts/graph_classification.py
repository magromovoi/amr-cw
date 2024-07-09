import json
import torch
import shutil
import numpy as np
import pandas as pd
from itertools import chain
from graph_models import GCN
from graph_dataset import GraphDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report


def generate_classification_report(classes, model, loader, device):

    y_pred = []
    y_true = []

    model = model.to(device)
    model.eval()

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
        y_true.extend(data.y.to(device).cpu().detach().numpy())

    cr = classification_report(y_true, y_pred, target_names=classes)

    print(cr)

    '''
    for index, ele in enumerate(y_pred):
        print(f"{classes[y_true[index]]} predicted as {classes[y_pred[index]]}")
    '''


def get_model_and_device(dataset, model_path=None, pretrained=False):

    if model_path is None and pretrained is False:
        model = GCN(dataset, hidden_channels=64)

    else:
        model = GCN(dataset, hidden_channels=64)
        model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders(dataset, classes, graphs_dataset_prefix, shuffle=True):

    train_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset)
    test_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset, test=True)

    train_loader = DataLoader(train_dataset, batch_size=768, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=shuffle)

    return train_loader, test_loader


def get_optimizer_and_criterion(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion


def train(model, loader, optimizer, criterion, device):
    model = model.to(device)
    model.train()

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        loss = criterion(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, loader, device):

    model = model.to(device)
    model.eval()

    correct_counter = 0

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data.y.to(device)).sum())

    return correct_counter / len(loader.dataset)


def epoch_iterator(dataset, classes, graphs_dataset_prefix, model_path):
    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix)
    model, device = get_model_and_device(train_loader.dataset)
    optimizer, criterion = get_optimizer_and_criterion(model)

    best_train_acc = 0
    early_stop_counter = 0

    for epoch in range(1, 1000):
        train(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            early_stop_counter = 0

        if early_stop_counter == 20 or train_acc > 0.95:
            break

        early_stop_counter += 1

    torch.save(model.state_dict(), model_path)

    print("Post training classification report")
    generate_classification_report(classes, model, test_loader, device)


def inference(dataset, classes, graphs_dataset_prefix, model_path):
    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix, shuffle=True)
    model, device = get_model_and_device(test_loader.dataset, model_path, pretrained=True)

    generate_classification_report(classes, model, test_loader, device)


def get_graph_embeddings(dataset, classes, graphs_dataset_prefix, graphs_index_prefix, embeddings_prefix, model_path):
    d = {'custom_index': [], 'graph_embeddings': [], 'label': [], 'split': [], 'target': []}

    train_graphs_index_df = pd.read_csv(graphs_index_prefix + '/' + dataset + '_train_graph_indices.csv')
    test_graphs_index_df = pd.read_csv(graphs_index_prefix + '/' + dataset + '_test_graph_indices.csv')

    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix, shuffle=False)
    model, device = get_model_and_device(test_loader.dataset, model_path, pretrained=True)

    model = model.to(device)
    model.eval()

    for data in train_loader:
        graph_embeddings = model.get_graph_embeddings(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        d['custom_index'].extend(np.array(train_graphs_index_df.custom_index.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['graph_embeddings'].extend(np.array(graph_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend(np.array(train_graphs_index_df.label.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['split'].extend(np.array(train_graphs_index_df.split.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['target'].extend(np.array(train_graphs_index_df.target.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())

    for data in test_loader:
        graph_embeddings = model.get_graph_embeddings(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        d['custom_index'].extend(np.array(test_graphs_index_df.custom_index.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['graph_embeddings'].extend(np.array(graph_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend(np.array(test_graphs_index_df.label.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['split'].extend(np.array(test_graphs_index_df.split.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['target'].extend(np.array(test_graphs_index_df.target.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())

    embeddings_file_name = f"{dataset}_graph_embeddings.json"

    with open(embeddings_file_name, 'w') as f:
        json.dump(d, f)

    embeddings_file_path = embeddings_prefix + '/' + embeddings_file_name
    shutil.move(embeddings_file_name, embeddings_file_path)


def classify_graphs(dataset, classes, graphs_dataset_prefix, graphs_index_prefix, embeddings_prefix, model_path, mode='inference'):

    if mode == 'training':
        epoch_iterator(dataset, classes, graphs_dataset_prefix, model_path)

    elif mode == 'inference':
        inference(dataset, classes, graphs_dataset_prefix, model_path)

    elif mode == 'get_graph_embeddings':
        get_graph_embeddings(dataset, classes, graphs_dataset_prefix, graphs_index_prefix, embeddings_prefix, model_path)

