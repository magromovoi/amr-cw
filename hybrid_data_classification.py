import json
import torch
import shutil
import numpy as np
import pandas as pd
from utils import save_checkpoint
from torch.utils.data import DataLoader
from hybrid_dataset import HybridDataset
from hybrid_models import HybridModel
from sklearn.metrics import classification_report


def generate_hybrid_classification_report(classes, hybrid_model, hybrid_loader, device):

    y_pred = []
    y_true = []

    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()

    for data in hybrid_loader:
        out = hybrid_model(data['hybrid_embedding'].to(device))
        y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
        y_true.extend(data['target'].to(device).cpu().detach().numpy())

    cr = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cr_2 = classification_report(y_true, y_pred, target_names=classes)

    print(cr['macro avg']['f1-score'])
    #print(cr_2)


def get_hybrid_model_and_device(dataset, hybrid_model_path=None):

    try:
        hybrid_model = HybridModel(dataset)
        checkpoint = torch.load(hybrid_model_path)
        hybrid_model.load_state_dict(checkpoint['model_state_dict'])

        last_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']

    except FileNotFoundError:
        hybrid_model = HybridModel(dataset)

        last_epoch = 0
        best_test_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(hybrid_model)

    return hybrid_model, device, last_epoch, best_test_acc


def get_hybrid_loaders(dataset, embeddings_prefix, graph_conv_type, graph_residual_connections):

    if graph_residual_connections:
        graph_residual_connections = 'residual'
    else:
        graph_residual_connections = 'non_residual'

    hybrid_embeddings_file_path = embeddings_prefix + "/" + (f"{dataset}_BERT_text_{graph_conv_type}_"
                                                             f"{graph_residual_connections}_hybrid_embeddings.json")

    with open(hybrid_embeddings_file_path) as f:
        d = json.load(f)

    df = pd.DataFrame.from_dict(d)

    hybrids_train_df = df[df['split'] == 'train']
    train_dataset = HybridDataset(hybrids_train_df)
    hybrid_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    hybrids_test_df = df[df['split'] == 'test']
    test_dataset = HybridDataset(hybrids_test_df)
    hybrid_test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return hybrid_train_loader, hybrid_test_loader


def get_hybrid_optimizer_and_criterion(hybrid_model, hybrid_model_path=None):

    hybrid_model.cuda()

    try:
        checkpoint = torch.load(hybrid_model_path)
        optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=1e-05, weight_decay=5e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    except FileNotFoundError:
        optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=1e-05, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion


def hybrid_train(hybrid_model, hybrid_loader, optimizer, criterion, device):
    hybrid_model = hybrid_model.to(device)
    hybrid_model.train()

    for data in hybrid_loader:
        out = hybrid_model(data['hybrid_embedding'].to(device))
        loss = criterion(out, data['target'].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate_hybrid_model_performance(hybrid_model, hybrid_loader, device):

    model = hybrid_model.to(device)
    model.eval()

    correct_counter = 0

    for data in hybrid_loader:
        out = model(data['hybrid_embedding'].to(device))
        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data['target'].to(device)).sum())

    return correct_counter / len(hybrid_loader.dataset)


def hybrid_epoch_iterator(dataset, classes, embeddings_prefix, graph_conv_type, graph_residual_connections, 
                          hybrid_model_path):
    hybrid_train_loader, hybrid_test_loader = get_hybrid_loaders(dataset, embeddings_prefix, graph_conv_type,
                                                                 graph_residual_connections)

    hybrid_model, device, last_epoch, best_test_acc = get_hybrid_model_and_device(hybrid_train_loader.dataset,
                                                                                  hybrid_model_path)

    optimizer, criterion = get_hybrid_optimizer_and_criterion(hybrid_model, hybrid_model_path)

    if last_epoch != 0:
        train_acc = evaluate_hybrid_model_performance(hybrid_model, hybrid_train_loader, device)
        test_acc = evaluate_hybrid_model_performance(hybrid_model, hybrid_test_loader, device)
        print(f"Resuming training from Epoch: {last_epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    for epoch in range(last_epoch + 1, last_epoch + 10):
        hybrid_train(hybrid_model, hybrid_train_loader, optimizer, criterion, device)
        train_acc = evaluate_hybrid_model_performance(hybrid_model, hybrid_train_loader, device)
        test_acc = evaluate_hybrid_model_performance(hybrid_model, hybrid_test_loader, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        save_checkpoint({'epoch': epoch, 'model_state_dict': hybrid_model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'best_test_acc': test_acc},
                        hybrid_model_path)

    print("Post training classification report")
    # generate_hybrid_classification_report(classes, hybrid_model, hybrid_test_loader, device)
    print(f"Final test accuracy post training: {evaluate_hybrid_model_performance(hybrid_model, hybrid_test_loader, 
                                                                                  device)}")


def hybrid_inference(dataset, classes, embeddings_prefix, graph_conv_type, graph_residual_connections,
                     hybrid_model_path):

    hybrid_train_loader, hybrid_test_loader = get_hybrid_loaders(dataset, embeddings_prefix, graph_conv_type,
                                                                 graph_residual_connections)

    hybrid_model, device, last_epoch, best_test_acc = get_hybrid_model_and_device(hybrid_test_loader.dataset,
                                                                                  hybrid_model_path)

    print("Inference classification report")
    generate_hybrid_classification_report(classes, hybrid_model, hybrid_test_loader, device)


def get_hybrid_embeddings(dataset, embeddings_prefix, graph_conv_type, graph_residual_connections):

    if graph_residual_connections:
        graph_residual_connections = 'residual'
    else:
        graph_residual_connections = 'non_residual'

    text_embeddings_file_name = embeddings_prefix + "/" + f"{dataset}_BERT_text_embeddings.json"

    graph_embeddings_file_name = (embeddings_prefix + "/" +
                                  f"{dataset}_{graph_conv_type}_{graph_residual_connections}_graph_embeddings.json")

    with open(text_embeddings_file_name) as f:
        d_text = json.load(f)

    text_embeddings_df = pd.DataFrame.from_dict(d_text)

    with open(graph_embeddings_file_name) as f:
        d_graph = json.load(f)

    graph_embeddings_df = pd.DataFrame.from_dict(d_graph)

    temp_hybrid_embeddings_df = pd.merge(graph_embeddings_df, text_embeddings_df, on='custom_index')

    assert temp_hybrid_embeddings_df['label_x'].equals(temp_hybrid_embeddings_df['label_y']), \
        f"Something went wrong during obtaining the text and graph embeddings, label mismatch"

    assert temp_hybrid_embeddings_df['split_x'].equals(temp_hybrid_embeddings_df['split_y']), \
        f"Something went wrong during obtaining the text and graph embeddings, split mismatch"

    assert temp_hybrid_embeddings_df['target_x'].equals(temp_hybrid_embeddings_df['target_y']), \
        f"Something went wrong during obtaining the text and graph embeddings, target mismatch"

    d = {'custom_index': list(temp_hybrid_embeddings_df['custom_index']),
         'hybrid_embeddings': np.hstack((np.array(list(temp_hybrid_embeddings_df['text_embeddings'])),
                                         np.array(list(temp_hybrid_embeddings_df['graph_embeddings'])))).tolist(),
         'label': list(temp_hybrid_embeddings_df['label_x']),
         'split': list(temp_hybrid_embeddings_df['split_x']),
         'target': list(temp_hybrid_embeddings_df['target_x']),
         }

    hybrid_embeddings_file_name = (f"{dataset}_BERT_text_{graph_conv_type}_{graph_residual_connections}"
                                   f"_hybrid_embeddings.json")

    with open(hybrid_embeddings_file_name, 'w') as f:
        json.dump(d, f)

    hybrid_embeddings_file_path = embeddings_prefix + '/' + hybrid_embeddings_file_name
    shutil.move(hybrid_embeddings_file_name, hybrid_embeddings_file_path)


def classify_hybrid_data(dataset, classes, embeddings_prefix, hybrid_model_path, graph_conv_type='gcn_conv',
                         graph_residual_connections=False, mode='train'):

    if mode == 'embeddings':
        print(f"Extracting hybrid embeddings for {dataset} using BERT and GNN {graph_conv_type} residual connections "
              f"{graph_residual_connections}")

        get_hybrid_embeddings(dataset, embeddings_prefix, graph_conv_type, graph_residual_connections)

    elif mode == 'train':
        print(f"Training hybrid data classification for {dataset} using BERT and GNN {graph_conv_type} "
              f"residual connections {graph_residual_connections}")

        hybrid_epoch_iterator(dataset, classes, embeddings_prefix, graph_conv_type, graph_residual_connections,
                              hybrid_model_path)

    elif mode == 'predict':
        print(f"Predicting hybrid data classification for {dataset} using BERT and GNN {graph_conv_type} "
              f"residual connections {graph_residual_connections}")

        hybrid_inference(dataset, classes, embeddings_prefix, graph_conv_type, graph_residual_connections,
                         hybrid_model_path)
