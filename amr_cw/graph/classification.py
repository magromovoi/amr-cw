import os
import json
import torch
import shutil
import numpy as np
from amr_cw.core.utils import save_checkpoint
from amr_cw.graph.models import GNN
from amr_cw.graph.dataset import GraphDataset
from torch.utils.data import ConcatDataset
from amr_cw.graph.concept_dataset import GraphConceptDataset
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

    cr = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    print(cr['macro avg']['f1-score'])


def evaluate_model_performance(model, loader, device):
    model = model.to(device)
    model.eval()

    correct_counter = 0

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data.y.to(device)).sum())

    return correct_counter / len(loader.dataset)


def get_model_and_device(dataset, num_classes, graph_model_path=None, graph_conv_type='gcn_conv',
                         graph_residual_connections=False, whitening=False, embedding=False):

    try:
        model = GNN(dataset, num_classes=num_classes, hidden_channels=300, conv_type=graph_conv_type,
                    residual_connections=graph_residual_connections, whitening=whitening, embedding=embedding)

        if whitening and '_whitened_' in graph_model_path:
            model.replace_norm_layers(300)

        checkpoint = torch.load(graph_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if whitening and '_whitened_' not in graph_model_path:
            model.replace_norm_layers(300)

        last_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']

        if 'best_neg_con_align' in checkpoint:
            best_neg_con_align = checkpoint['best_neg_con_align']

        else:
            best_neg_con_align = None

    except FileNotFoundError:
        model = GNN(dataset, num_classes=num_classes, hidden_channels=300, conv_type=graph_conv_type,
                    residual_connections=graph_residual_connections)

        last_epoch = 0
        best_test_acc = 0
        best_neg_con_align = None

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')

    print(model)

    if best_neg_con_align is not None:
        return model, device, last_epoch, best_test_acc, best_neg_con_align

    else:
        return model, device, last_epoch, best_test_acc


def get_loaders(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix=None, concepts=None,
                concept_gradient_importance_flag=False, concept_dot_product_flag=False,
                concept_concept_axis_visualization_flag=False, top_activation_subgraphs_flag=False):

    train_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset)
    test_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset, test=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    if concept_gradient_importance_flag:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if graph_concepts_dataset_prefix and concepts:

        train_concept_datasets = {}
        train_concept_loaders = {}

        test_concept_datasets = {}
        test_concept_loaders = {}

        for concept in concepts:
            train_concept_datasets[concept] = GraphConceptDataset(root=f"{graph_concepts_dataset_prefix + '/' + concept}",
                                                                  label=concept, labels=classes)

            train_concept_loaders[concept] = DataLoader(train_concept_datasets[concept], batch_size=32, shuffle=True)

            test_concept_datasets[concept] = GraphConceptDataset(root=f"{graph_concepts_dataset_prefix + '/' + concept}",
                                                                 label=concept, labels=classes, test=True)

            test_concept_loaders[concept] = DataLoader(test_concept_datasets[concept], batch_size=32, shuffle=False)

        if concept_dot_product_flag:

            test_concept_dataset_temp = []

            for concept, test_concept_dataset in test_concept_datasets.items():
                test_concept_dataset_temp.append(test_concept_dataset)

            merged_test_concept_dataset = ConcatDataset(test_concept_dataset_temp)

            merged_test_concept_loader = DataLoader(merged_test_concept_dataset, batch_size=1, shuffle=False)

            return train_loader, merged_test_concept_loader

        elif concept_concept_axis_visualization_flag or top_activation_subgraphs_flag:

            test_concept_dataset_temp = []

            for concept, test_concept_dataset in test_concept_datasets.items():
                test_concept_dataset_temp.append(test_concept_dataset)

            merged_test_concept_dataset = ConcatDataset(test_concept_dataset_temp)

            merged_test_concept_loader = DataLoader(merged_test_concept_dataset, batch_size=32, shuffle=False)

            return train_loader, merged_test_concept_loader

        else:
            return train_loader, test_loader, train_concept_loaders, test_concept_loaders

    else:
        return train_loader, test_loader


def get_optimizer_and_criterion(model, model_path=None):

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    except FileNotFoundError:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion


def train(model, loader, optimizer, criterion, device):
    model = model.to(device)
    model.train()

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        loss = criterion(out, data.y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def epoch_iterator(dataset, classes, graphs_dataset_prefix, graph_model_path, graph_conv_type,
                   graph_residual_connections):

    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix)
    model, device, last_epoch, best_test_acc = get_model_and_device(train_loader.dataset, len(classes), graph_model_path,
                                                                    graph_conv_type, graph_residual_connections)

    optimizer, criterion = get_optimizer_and_criterion(model, graph_model_path)

    if last_epoch != 0:
        train_acc = evaluate_model_performance(model, train_loader, device)
        test_acc = evaluate_model_performance(model, test_loader, device)
        print(f"Resuming training from Epoch: {last_epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    for epoch in range(last_epoch + 1, last_epoch + 21):
        train(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate_model_performance(model, train_loader, device)
        test_acc = evaluate_model_performance(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'best_test_acc': test_acc},
                        graph_model_path)

    print("Post training classification report")
    generate_classification_report(classes, model, test_loader, device)
    print(f"Final test accuracy post training: {evaluate_model_performance(model, test_loader, device)}")


def inference(dataset, classes, graphs_dataset_prefix, graph_model_path, graph_conv_type, graph_residual_connections):
    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix)

    model, device, last_epoch, best_test_acc = get_model_and_device(test_loader.dataset, len(classes), graph_model_path,
                                                                    graph_conv_type, graph_residual_connections)

    print("Inference classification report")
    generate_classification_report(classes, model, test_loader, device)
    print(f"Final test accuracy inference: {evaluate_model_performance(model, test_loader, device)}")


def get_graph_embeddings(dataset, classes, graphs_dataset_prefix, embeddings_prefix, graph_model_path, graph_conv_type,
                         graph_residual_connections):
    d = {'custom_index': [], 'graph_embeddings': [], 'label': [], 'split': [], 'target': []}

    train_loader, test_loader = get_loaders(dataset, classes, graphs_dataset_prefix)
    model, device, last_epoch, best_test_acc = get_model_and_device(train_loader.dataset, len(classes),
                                                                    graph_model_path, graph_conv_type,
                                                                    graph_residual_connections, embedding=True)

    model = model.to(device)
    model.eval()

    for train_data in train_loader:
        train_graph_embeddings = model(train_data.x.to(device), train_data.edge_index.to(device), train_data.batch.to(device))

        d['custom_index'].extend([train_loader.dataset.idx_to_custom_index[int(graph_id)]
                                 for graph_id in train_data.graph_idx])

        d['graph_embeddings'].extend(np.array(train_graph_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend([train_loader.dataset.idx_to_class(int(label_id)) for label_id in train_data.label_idx])
        d['split'].extend([train_loader.dataset.idx_to_split(int(split_id)) for split_id in train_data.split_idx])
        d['target'].extend([int(label_id) for label_id in train_data.label_idx])

    for test_data in test_loader:
        test_graph_embeddings = model(test_data.x.to(device), test_data.edge_index.to(device), test_data.batch.to(device))

        d['custom_index'].extend([test_loader.dataset.idx_to_custom_index[int(graph_id)]
                                  for graph_id in test_data.graph_idx])

        d['graph_embeddings'].extend(np.array(test_graph_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend([test_loader.dataset.idx_to_class(int(label_id)) for label_id in test_data.label_idx])
        d['split'].extend([test_loader.dataset.idx_to_split(int(split_id)) for split_id in test_data.split_idx])
        d['target'].extend([int(label_id) for label_id in test_data.label_idx])

    if graph_residual_connections:
        graph_residual_connections = 'residual'
    else:
        graph_residual_connections = 'non_residual'

    embeddings_file_name = f"{dataset}_{graph_conv_type}_{graph_residual_connections}_graph_embeddings.json"

    with open(embeddings_file_name, 'w') as f:
        json.dump(d, f)

    os.makedirs(embeddings_prefix, exist_ok=True)
    embeddings_file_path = embeddings_prefix + '/' + embeddings_file_name
    shutil.move(embeddings_file_name, embeddings_file_path)


def classify_graphs(dataset, classes, graphs_dataset_prefix, embeddings_prefix, graph_model_path,
                    graph_conv_type, graph_residual_connections, mode='train'):

    if mode == 'train':

        print(f"Training graph classification for {dataset} using GNN {graph_conv_type} "
              f"with residual connections {graph_residual_connections}")

        epoch_iterator(dataset, classes, graphs_dataset_prefix, graph_model_path, graph_conv_type,
                       graph_residual_connections)

    elif mode == 'predict':

        print(f"Predicting graph classification for {dataset} using GNN {graph_conv_type} residual connections "
              f"{graph_residual_connections}")

        inference(dataset, classes, graphs_dataset_prefix, graph_model_path,
                  graph_conv_type, graph_residual_connections)

    elif mode == 'embeddings':
        print(f"Extracting graph embeddings for {dataset} using GNN {graph_conv_type} residual connections "
              f"{graph_residual_connections}")

        get_graph_embeddings(dataset, classes, graphs_dataset_prefix, embeddings_prefix, graph_model_path,
                             graph_conv_type, graph_residual_connections)
