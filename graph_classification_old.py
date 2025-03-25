import os
import torch
from graph_models import GNN
from graph_dataset import GraphDataset
from torch.utils.data import ConcatDataset
from graph_concept_dataset import GraphConceptDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report


def save_checkpoint(state, checkpoint_prefix=None):
    torch.save(state, checkpoint_prefix)


def generate_classification_report(classes, model, loader, device):
    # print("Classification report function sanity check.")

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


def evaluate_model_performance(model, loader, device):
    # print("Evaluate function sanity check.")

    model = model.to(device)
    model.eval()

    correct_counter = 0

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data.y.to(device)).sum())

    return correct_counter / len(loader.dataset)


'''
def get_model_and_device(dataset, graph_model_path=None, graph_conv_type='gcn_conv', graph_residual_connections=False):

    try:
        model = GNN(dataset, hidden_channels=300, conv_type=graph_conv_type,
                    residual_connections=graph_residual_connections)

        checkpoint = torch.load(graph_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        last_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']

    except FileNotFoundError:
        model = GNN(dataset, hidden_channels=300, conv_type=graph_conv_type,
                    residual_connections=graph_residual_connections)

        last_epoch = 0
        best_test_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(model)

    return model, device, last_epoch, best_test_acc
'''


def get_model_and_device(dataset, num_classes, graph_model_path=None, graph_conv_type='gcn_conv',
                         graph_residual_connections=False, whitening=False):

    try:
        model = GNN(dataset, num_classes=num_classes, hidden_channels=300, conv_type=graph_conv_type,
                    residual_connections=graph_residual_connections, whitening=whitening)

        if whitening and '_whitened_' in graph_model_path:
            model.replace_norm_layers(300)

        checkpoint = torch.load(graph_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if whitening and '_whitened_' not in graph_model_path:
            model.replace_norm_layers(300)

        last_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']

    except FileNotFoundError:
        model = GNN(dataset, num_classes=num_classes, hidden_channels=300, conv_type=graph_conv_type,
                    residual_connections=graph_residual_connections)

        last_epoch = 0
        best_test_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(model)

    return model, device, last_epoch, best_test_acc


'''
def get_loaders(dataset, classes, graphs_dataset_prefix):

    train_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset)
    test_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset, test=True)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
'''


def get_loaders(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix=None, concepts=None,
                concept_gradient_importance_flag=False, concept_dot_product_flag=False):

    train_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset)
    test_dataset = GraphDataset(root=graphs_dataset_prefix, labels=classes, dataset=dataset, test=True)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    if concept_gradient_importance_flag:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    if graph_concepts_dataset_prefix and concepts:

        train_concept_datasets = {}
        train_concept_loaders = {}

        test_concept_datasets = {}
        test_concept_loaders = {}

        for concept in concepts:
            train_concept_datasets[concept] = GraphConceptDataset(root=f"{graph_concepts_dataset_prefix + '/' + concept}",
                                                                  label=concept, labels=classes)

            train_concept_loaders[concept] = DataLoader(train_concept_datasets[concept], batch_size=64, shuffle=True)

            test_concept_datasets[concept] = GraphConceptDataset(root=f"{graph_concepts_dataset_prefix + '/' + concept}",
                                                                 label=concept, labels=classes, test=True)

            test_concept_loaders[concept] = DataLoader(test_concept_datasets[concept], batch_size=1, shuffle=True)

        if concept_dot_product_flag:

            test_concept_dataset_temp = []

            for concept, test_concept_dataset in test_concept_datasets.items():
                test_concept_dataset_temp.append(test_concept_dataset)

            merged_test_concept_dataset = ConcatDataset(test_concept_dataset_temp)

            merged_test_concept_loader = DataLoader(merged_test_concept_dataset, batch_size=1, shuffle=True)

            return train_loader, merged_test_concept_loader

        else:
            return train_loader, test_loader, train_concept_loaders, test_concept_loaders

    else:
        return train_loader, test_loader


def get_optimizer_and_criterion(model, model_path=None):

    model.cuda()

    try:
        checkpoint = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(), 'best_test_acc': best_test_acc},
                            graph_model_path)

            print(f"New best Test Accuracy achieved on Epoch: {epoch:03d}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

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


def classify_graphs(dataset, classes, graphs_dataset_prefix, graph_model_path,
                    graph_conv_type, graph_residual_connections, mode='train'):

    if mode == 'train':

        print(f"Training graph classification for {dataset} using GNN {graph_conv_type} "
              f"with residual connections {graph_residual_connections}")

        epoch_iterator(dataset, classes, graphs_dataset_prefix, graph_model_path, graph_conv_type,
                       graph_residual_connections)

    elif mode == 'predict':

        print(f"Predicting graph classification for {dataset} using GNN {graph_conv_type} residual connections {graph_residual_connections}")

        inference(dataset, classes, graphs_dataset_prefix, graph_model_path,
                  graph_conv_type, graph_residual_connections)
