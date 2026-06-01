import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
from utils import save_checkpoint
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from transformers import BertTokenizer
from text_model import BERTTextClassifier
from sklearn.metrics import classification_report


def generate_text_classification_report(classes, text_model, text_loader, device):

    y_pred = []
    y_true = []

    text_model = text_model.to(device)
    text_model.eval()

    for data in text_loader:
        out = text_model(input_ids=data['input_ids'].to(device), attention_mask=data['attention_mask'].to(device))

        y_pred.extend(out.argmax(dim=1).cpu().detach().numpy())
        y_true.extend(data['target'].to(device).cpu().detach().numpy())

    cr = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    print(cr['macro avg']['f1-score'])


def evaluate_text_model_performance(text_model, text_loader, device):
    # print("Evaluate function sanity check.")

    text_model = text_model.to(device)
    text_model.eval()

    correct_counter = 0

    for data in text_loader:
        out = text_model(input_ids=data['input_ids'].to(device), attention_mask=data['attention_mask'].to(device))

        prediction = out.argmax(dim=1)
        correct_counter += int((prediction == data['target'].to(device)).sum())

    return correct_counter / len(text_loader.dataset)


def get_text_model_and_device(dataset, text_model_path=None, embedding=False):

    try:
        text_model = BERTTextClassifier(dataset, embedding=embedding)
        checkpoint = torch.load(text_model_path)
        text_model.load_state_dict(checkpoint['model_state_dict'])

        last_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']

    except FileNotFoundError:
        text_model = BERTTextClassifier(dataset, embedding=embedding)

        last_epoch = 0
        best_test_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(text_model)

    return text_model, device, last_epoch, best_test_acc


def get_text_loaders(dataset, raw_texts_csv_prefix):

    raw_texts_csv_prefix = raw_texts_csv_prefix + '/' + dataset + '_raw_texts.csv'
    df = pd.read_csv(raw_texts_csv_prefix)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    texts_train_df = df[df['split'] == 'train']
    train_dataset = TextDataset(texts_train_df, tokenizer)
    text_train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    texts_test_df = df[df['split'] == 'test']
    test_dataset = TextDataset(texts_test_df, tokenizer)
    text_test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    return text_train_loader, text_test_loader


def get_text_optimizer_and_criterion(text_model, text_model_path=None):

    text_model.cuda()

    try:
        checkpoint = torch.load(text_model_path)
        optimizer = torch.optim.Adam(text_model.parameters(), lr=1e-05, weight_decay=5e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    except FileNotFoundError:
        optimizer = torch.optim.Adam(text_model.parameters(), lr=1e-05, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion


def text_train(text_model, text_loader, optimizer, criterion, device):
    text_model = text_model.to(device)
    text_model.train()

    for data in text_loader:
        out = text_model(input_ids=data['input_ids'].to(device), attention_mask=data['attention_mask'].to(device))

        loss = criterion(out, data['target'].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def text_epoch_iterator(dataset, classes, raw_texts_csv_prefix, text_model_path):
    text_train_loader, text_test_loader = get_text_loaders(dataset, raw_texts_csv_prefix)

    text_model, device, last_epoch, best_test_acc = get_text_model_and_device(text_train_loader.dataset,
                                                                              text_model_path)

    optimizer, criterion = get_text_optimizer_and_criterion(text_model, text_model_path)

    if last_epoch != 0:
        train_acc = evaluate_text_model_performance(text_model, text_train_loader, device)
        test_acc = evaluate_text_model_performance(text_model, text_test_loader, device)
        print(f"Resuming training from Epoch: {last_epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    for epoch in range(last_epoch + 1, last_epoch + 10):
        text_train(text_model, text_train_loader, optimizer, criterion, device)
        train_acc = evaluate_text_model_performance(text_model, text_train_loader, device)
        test_acc = evaluate_text_model_performance(text_model, text_test_loader, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        save_checkpoint({'epoch': epoch, 'model_state_dict': text_model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'best_test_acc': test_acc},
                        text_model_path)

        print(f"New best Test Accuracy achieved on Epoch: {epoch:03d}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    print("Post training classification report")
    generate_text_classification_report(classes, text_model, text_test_loader, device)
    print(f"Final test accuracy post training: {evaluate_text_model_performance(text_model, text_test_loader, device)}")


def text_inference(dataset, classes, raw_texts_csv_prefix, text_model_path):
    text_train_loader, text_test_loader = get_text_loaders(dataset, raw_texts_csv_prefix)

    text_model, device, last_epoch, best_test_acc = get_text_model_and_device(text_train_loader.dataset,
                                                                              text_model_path)

    print("Inference classification report")
    generate_text_classification_report(classes, text_model, text_test_loader, device)


def get_text_embeddings(dataset, raw_texts_csv_prefix, embeddings_prefix, text_model_path):
    d = {'custom_index': [], 'text_embeddings': [], 'label': [], 'split': [], 'target': []}

    texts_df = pd.read_csv(raw_texts_csv_prefix + '/' + dataset + '_raw_texts.csv')

    texts_train_df = texts_df[texts_df['split'] == 'train']
    texts_test_df = texts_df[texts_df['split'] == 'test']

    text_train_loader, text_test_loader = get_text_loaders(dataset, raw_texts_csv_prefix)

    text_model, device, last_epoch, best_test_acc = get_text_model_and_device(text_train_loader.dataset,
                                                                              text_model_path, embedding=True)

    text_model = text_model.to(device)
    text_model.eval()

    for train_data in text_train_loader:
        text_train_embeddings = text_model(input_ids=train_data['input_ids'].to(device),
                                           attention_mask=train_data['attention_mask'].to(device))

        d['custom_index'].extend(train_data['custom_index'])
        d['text_embeddings'].extend(np.array(text_train_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend(train_data['label'])
        d['split'].extend(train_data['split'])
        d['target'].extend(np.array(train_data['target'].to(device).cpu().detach().numpy()).tolist())

    for test_data in text_test_loader:
        text_test_embeddings = text_model(input_ids=test_data['input_ids'].to(device),
                                          attention_mask=test_data['attention_mask'].to(device))

        d['custom_index'].extend(test_data['custom_index'])
        d['text_embeddings'].extend(np.array(text_test_embeddings.to(device).cpu().detach().numpy()).tolist())
        d['label'].extend(test_data['label'])
        d['split'].extend(test_data['split'])
        d['target'].extend(np.array(test_data['target'].to(device).cpu().detach().numpy()).tolist())

    embeddings_file_name = f'{dataset}_BERT_text_embeddings.json'

    with open(embeddings_file_name, 'w') as f:
        json.dump(d, f)

    os.makedirs(embeddings_prefix, exist_ok=True)
    embeddings_file_path = embeddings_prefix + '/' + embeddings_file_name
    shutil.move(embeddings_file_name, embeddings_file_path)


def classify_texts(dataset, classes, raw_texts_csv_prefix, embeddings_prefix, text_model_path, mode='train'):

    if mode == 'train':
        print(f"Training text classification for {dataset} using BERT")

        text_epoch_iterator(dataset, classes, raw_texts_csv_prefix, text_model_path)

    elif mode == 'predict':
        print(f"Predicting text classification for {dataset} using BERT")

        text_inference(dataset, classes, raw_texts_csv_prefix, text_model_path)

    elif mode == 'embeddings':
        print(f"Extracting BERT-based document embeddings for {dataset}")

        get_text_embeddings(dataset, raw_texts_csv_prefix, embeddings_prefix, text_model_path)
