import torch
import pandas as pd
from transformers import BertTokenizer
from itertools import chain
from torch.utils.data import DataLoader
from text_models import BERTTextClassifier
from sklearn.metrics import classification_report
from ten_newsgroups_texts_dataset import TenNewsGroupsTextDataset


def get_model_and_device(dataset):
    model = BERTTextClassifier(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders(dataset, dataframe_file_name):

    dataframe_file_name = dataframe_file_name + '/' + dataset + '.csv'
    dataframe = pd.read_csv(dataframe_file_name)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_dataframe = dataframe[dataframe['split'] == 'train']
    train_dataset = TenNewsGroupsTextDataset(train_dataframe, tokenizer, 512)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataframe = dataframe[dataframe['split'] == 'test']
    test_dataset = TenNewsGroupsTextDataset(test_dataframe, tokenizer, 512)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

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


def epoch_iterator(dataset, classes, dataframe_file_name):
    train_loader, test_loader = get_loaders(dataset, dataframe_file_name)
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

        if early_stop_counter > 5 or train_acc > 0.96:
            break

        early_stop_counter += 1

    generate_classification_report(classes, model, test_loader, device)


def inference(dataset, classes, dataframe_file_name):
    train_loader, test_loader = get_loaders(dataset, dataframe_file_name)
    model, device = get_model_and_device(test_loader.dataset)

    generate_classification_report(classes, model, test_loader, device)


def generate_classification_report(classes, model, loader, device):

    y_pred = []
    y_true = []

    model = model.to(device)
    model.eval()

    for data in loader:
        out = model(data['input_ids'].to(device), data['attention_mask'].to(device))
        y_pred.append(out.argmax(dim=1).cpu().detach().numpy())
        y_true.append(data['targets'].to(device).cpu().detach().numpy())

    y_pred = list(chain.from_iterable(y_pred))
    y_true = list(chain.from_iterable(y_true))

    cr = classification_report(y_true, y_pred, target_names=classes)

    print(cr)

    '''
    for index, ele in enumerate(y_pred):
        print(f"{classes[y_true[index]]} predicted as {classes[y_pred[index]]}")
    '''


def classify_texts(dataset, classes, dataframe_file_name):
    epoch_iterator(dataset, classes, dataframe_file_name)
    # inference(dataset, classes, dataframe_file_name)
