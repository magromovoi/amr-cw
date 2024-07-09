import json
import torch
import shutil
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from hybrid_dataset import HybridDataset
from hybrid_models import HybridModel
from sklearn.metrics import classification_report


def generate_classification_report(classes, model, loader, device):

    y_pred = []
    y_true = []

    model = model.to(device)
    model.eval()

    for data in loader:
        out = model(data['hybrid_embeddings'].to(device))
        y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
        y_true.extend(data['targets'].to(device).cpu().detach().numpy())

    cr = classification_report(y_true, y_pred, target_names=classes)

    print(cr)

    '''
    for index, ele in enumerate(y_pred):
        print(f"{classes[y_true[index]]} predicted as {classes[y_pred[index]]}")
    '''


def get_model_and_device(dataset, model_path=None, pretrained=False):

    if model_path is None and pretrained is False:
        model = HybridModel(dataset)

    else:
        model = HybridModel(dataset)
        model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders(dataset, embeddings_prefix, shuffle=True):
    hybrid_embeddings_file_name = embeddings_prefix + "/" + f"{dataset}_hybrid_embeddings.json"

    with open(hybrid_embeddings_file_name) as f:
        d = json.load(f)

    df = pd.DataFrame.from_dict(d)

    hybrids_train_df = df[df['split'] == 'train']
    train_dataset = HybridDataset(hybrids_train_df)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=shuffle)

    hybrids_test_df = df[df['split'] == 'test']
    test_dataset = HybridDataset(hybrids_test_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=shuffle)

    return train_loader, test_loader


def get_optimizer_and_criterion(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion


def train(model, loader, optimizer, criterion, device):
    model = model.to(device)
    model.train()

    for data in loader:
        out = model(data['hybrid_embeddings'].to(device))
        loss = criterion(out, data['targets'].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, loader, device):

    model = model.to(device)
    model.eval()

    correct_counter = 0

    for data in loader:
        out = model(data['hybrid_embeddings'].to(device))
        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data['targets'].to(device)).sum())

    return correct_counter / len(loader.dataset)


def epoch_iterator(dataset, classes, embeddings_prefix, model_path):
    train_loader, test_loader = get_loaders(dataset, embeddings_prefix)
    model, device = get_model_and_device(train_loader.dataset)
    optimizer, criterion = get_optimizer_and_criterion(model)

    best_train_acc = 0
    early_stop_counter = 0

    for epoch in range(1, 1000):
        train(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            early_stop_counter = 0

        if early_stop_counter > 5 or train_acc > 0.975:
            break

        early_stop_counter += 1

    torch.save(model.state_dict(), model_path)

    print("Post training classification report")
    generate_classification_report(classes, model, test_loader, device)


def inference(dataset, classes, raw_texts_csv_prefix, model_path):
    train_loader, test_loader = get_loaders(dataset, raw_texts_csv_prefix)
    model, device = get_model_and_device(test_loader.dataset, model_path, pretrained=True)

    print("Inference classification report")
    generate_classification_report(classes, model, test_loader, device)


def construct_hybrid_embeddings(dataset, embeddings_prefix):
    graph_embeddings_file_name = embeddings_prefix + "/" + f"{dataset}_graph_embeddings.json"
    text_embeddings_file_name = embeddings_prefix + "/" + f"{dataset}_text_embeddings.json"

    with open(graph_embeddings_file_name) as f:
        d_graph = json.load(f)

    graph_embeddings_df = pd.DataFrame.from_dict(d_graph)

    with open(text_embeddings_file_name) as f:
        d_text = json.load(f)

    text_embeddings_df = pd.DataFrame.from_dict(d_text)

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

    embeddings_file_name = f"{dataset}_hybrid_embeddings.json"

    with open(embeddings_file_name, 'w') as f:
        json.dump(d, f)

    embeddings_file_path = embeddings_prefix + '/' + embeddings_file_name
    shutil.move(embeddings_file_name, embeddings_file_path)


def classify_hybrid_data(dataset, classes, embeddings_prefix, model_path, mode='training'):

    if mode == 'construct_hybrid_embeddings':
        construct_hybrid_embeddings(dataset, embeddings_prefix)

    elif mode == 'training':
        epoch_iterator(dataset, classes, embeddings_prefix, model_path)

    elif mode == 'inference':
        inference(dataset, classes, embeddings_prefix, model_path)
