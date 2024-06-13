import torch
from models import GCN_10
from itertools import chain
from torch_geometric.loader import DataLoader
from ten_newsgroups_dataset import NewsGroupDataset
from sklearn.metrics import classification_report


def get_model_and_device(dataset, pretrained=False):
    model = GCN_10(dataset, hidden_channels=64, pretrained=pretrained)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, device


def get_loaders():
    train_dataset = NewsGroupDataset(root='ten_newsgroups_data/')
    test_dataset = NewsGroupDataset(root='ten_newsgroups_data/', test=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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


def epoch_iterator(save_model=False):
    train_loader, test_loader = get_loaders()
    model, device = get_model_and_device(train_loader.dataset, pretrained=False)
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

        if early_stop_counter >= 15 or test_acc >= 0.84:
            break

        early_stop_counter += 1

    if save_model:
        model_path = 'ten_newsgroups_model.pt'
        torch.save(model.state_dict(), model_path)

    generate_classification_report(model, test_loader, device)


def inference():
    train_loader, test_loader = get_loaders()
    model, device = get_model_and_device(test_loader.dataset, pretrained=True)

    generate_classification_report(model, test_loader, device)


def generate_classification_report(model, loader, device):

    y_pred = []
    y_true = []

    classes = ["business", "entertainment", "food", "graphics", "historical",
               "medical", "politics", "space", "sport", "technologie"]

    model = model.to(device)
    model.eval()

    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        y_pred.append(out.argmax(dim=1).cpu().detach().numpy())
        y_true.append(data.y.to(device).cpu().detach().numpy())

    y_pred = list(chain.from_iterable(y_pred))
    y_true = list(chain.from_iterable(y_true))

    cr = classification_report(y_true, y_pred, target_names=classes)

    print(cr)

    for index, ele in enumerate(y_pred):
        print(f"{classes[y_true[index]]} predicted as {classes[y_pred[index]]}")


if __name__ == "__main__":
    # epoch_iterator(save_model=True)
    inference()
