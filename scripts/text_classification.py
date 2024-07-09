import json
import torch
import shutil
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from transformers import BertTokenizer
from text_models import BERTTextClassifier
from sklearn.metrics import classification_report


def generate_classification_report(classes, model, loader, device):

    y_pred = []
    y_true = []

    model = model.to(device)
    model.eval()

    for data in loader:
        out = model(data['input_ids'].to(device), data['attention_mask'].to(device))
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
        model = BERTTextClassifier(dataset)

    else:
        model = BERTTextClassifier(dataset)
        model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders(dataset, raw_texts_csv_prefix, shuffle=True):

    raw_texts_csv_prefix = raw_texts_csv_prefix + '/' + dataset + '_raw_texts.csv'
    df = pd.read_csv(raw_texts_csv_prefix)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    texts_train_df = df[df['split'] == 'train']
    train_dataset = TextDataset(texts_train_df, tokenizer, 512)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=shuffle)

    texts_test_df = df[df['split'] == 'test']
    test_dataset = TextDataset(texts_test_df, tokenizer, 512)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=shuffle)

    return train_loader, test_loader


def get_optimizer_and_criterion(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion


def train(model, loader, optimizer, criterion, device):
    model = model.to(device)
    model.train()

    for data in loader:
        out = model(data['input_ids'].to(device), data['attention_mask'].to(device))
        loss = criterion(out, data['targets'].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, loader, device):

    model = model.to(device)
    model.eval()

    correct_counter = 0

    for data in loader:
        out = model(data['input_ids'].to(device), data['attention_mask'].to(device))
        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data['targets'].to(device)).sum())

    return correct_counter / len(loader.dataset)


def epoch_iterator(dataset, classes, raw_texts_csv_prefix, model_path):
    train_loader, test_loader = get_loaders(dataset, raw_texts_csv_prefix)
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


def get_text_embeddings(dataset, raw_texts_csv_prefix, embeddings_prefix, model_path):
    d = {'custom_index': [], 'text_embeddings': [], 'label': [], 'split': [], 'target': []}

    texts_df = pd.read_csv(raw_texts_csv_prefix + '/' + dataset + '_raw_texts.csv')

    texts_train_df = texts_df[texts_df['split'] == 'train']
    texts_test_df = texts_df[texts_df['split'] == 'test']

    train_loader, test_loader = get_loaders(dataset, raw_texts_csv_prefix, shuffle=False)
    model, device = get_model_and_device(test_loader.dataset, model_path, pretrained=True)

    model = model.to(device)
    model.eval()

    for data in train_loader:
        text_embeddings = model.get_text_embeddings(data['input_ids'].to(device), data['attention_mask'].to(device))
        d['custom_index'].extend(np.array(texts_train_df.custom_index.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['text_embeddings'].extend(np.array(text_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend(np.array(texts_train_df.label.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['split'].extend(np.array(texts_train_df.split.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['target'].extend(np.array(texts_train_df.target.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())

    for data in test_loader:
        text_embeddings = model.get_text_embeddings(data['input_ids'].to(device), data['attention_mask'].to(device))
        d['custom_index'].extend(np.array(texts_test_df.custom_index.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['text_embeddings'].extend(np.array(text_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend(np.array(texts_test_df.label.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['split'].extend(np.array(texts_test_df.split.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())
        d['target'].extend(np.array(texts_test_df.target.iloc[data['idx'].to(device).cpu().detach().numpy()]).tolist())

    embeddings_file_name = f'{dataset}_text_embeddings.json'

    with open(embeddings_file_name, 'w') as f:
        json.dump(d, f)

    embeddings_file_path = embeddings_prefix + '/' + embeddings_file_name
    shutil.move(embeddings_file_name, embeddings_file_path)


def classify_texts(dataset, classes, raw_texts_csv_prefix, embeddings_prefix, model_path, mode='training'):

    if mode == 'training':
        epoch_iterator(dataset, classes, raw_texts_csv_prefix, model_path)

    elif mode == 'inference':
        inference(dataset, classes, raw_texts_csv_prefix, model_path)

    elif mode == 'get_text_embeddings':
        get_text_embeddings(dataset, raw_texts_csv_prefix, embeddings_prefix, model_path)
