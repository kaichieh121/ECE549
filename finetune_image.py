import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from L3Net import L3NetImageOnly
from sklearn.metrics import confusion_matrix, classification_report

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default="D:\\datasets\\imagenet")
    parser.add_argument('--manifest_dir', default="D:\\Projects\\ECE549\\manifest")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    return args


def train(dataloader, model, loss_fn, optimizer, loss_history = [], accuracy_history = []):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    cur_size = 0
    loss_avg = []
    accuracy_avg = []
    for batch, sample_batched in enumerate(dataloader):

        audio = sample_batched[0].to(device)
        y = sample_batched[1].to(device)

        # Compute prediction error
        pred = model(audio)
        loss = loss_fn(pred, y)

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        cur_size += pred.shape[0]
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 200 == 0:
            loss, current = loss.item(), (batch + 1) * len(audio)
            accuracy = 100*correct / cur_size
            loss_avg.append(loss)
            accuracy_avg.append(accuracy)
            print(f"loss: {loss:>7f} Accuracy: {(accuracy):>0.1f}% [{current:>5d}/{size:>5d}]")
    loss_history.append(np.mean(np.array(loss_avg)))
    accuracy_history.append(np.mean(np.array(accuracy_avg)))
    return loss_history, accuracy_history

def test(dataloader, model, loss_fn, test_loss_hist, test_accuracy_hist):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample_batched in dataloader:
            audio = sample_batched[0].to(device)
            y = sample_batched[1].to(device)
            pred = model(audio)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    test_loss_hist.append(test_loss)
    test_accuracy_hist.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss_hist, test_accuracy_hist, test_loss, correct

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluation(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    y_list = None
    pred_list = None
    with torch.no_grad():
        for sample_batched in dataloader:
            audio = sample_batched[0].to(device)
            pred = model(audio)
            if y_list is None:
                y_list = sample_batched[1]
                pred_list = pred.argmax(1).cpu()
            else:
                y_list = torch.cat((y_list, sample_batched[1]), dim=0)
                pred_list = torch.cat((pred_list,pred.argmax(1).cpu()), dim=0)

    cm = confusion_matrix(y_list, pred_list)
    plot_confusion_matrix(cm)
    plt.show()
    print(classification_report(y_list, pred_list))

    return

def main(args):
    data_dir = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    batch_size = args.batch_size
    epochs = args.epochs

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    full_dataset = torchvision.datasets.ImageFolder(
        data_dir / 'val',
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            normalize,
        ]))
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train_dataset = torchvision.datasets.ImageFolder(
    #     data_dir / 'train',
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(256),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    #
    # test_dataset = torchvision.datasets.ImageFolder(
    #     data_dir / 'val',
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    L3Net_weights = None
    L3Net_weights = torch.load(manifest_dir / 'best_model_by_accuracy.pth')

    model = L3NetImageOnly(L3Net_weights).to(device)
    # model.load_state_dict(torch.load(manifest_dir / 'imageonly_best_model_by_accuracy.pth'))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loss_hist = []
    test_loss_hist = []
    train_accuracy_hist = []
    test_accuracy_hist = []
    best_loss = float('inf')
    best_accuracy = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss_hist, train_accuracy_hist = train(train_dataloader, model, loss_fn, optimizer, train_loss_hist, train_accuracy_hist)
        test_loss_hist, test_accuracy_hist, test_loss, test_accuracy = test(test_dataloader, model, loss_fn, test_loss_hist, test_accuracy_hist)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print("Saving Best Model by accuracy...")
            torch.save(model.state_dict(), manifest_dir / 'imageonly_best_model_by_accuracy.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            print("Saving Best Model by loss...")
            torch.save(model.state_dict(), manifest_dir / 'imageonly_best_model_by_loss.pth')
    torch.save(model.state_dict(), manifest_dir / 'imageonly_last_model.pth')
    print("Done!")

    data = {'train_loss_hist': train_loss_hist,
            'test_loss_hist': test_loss_hist,
            'train_accuracy_hist': train_accuracy_hist,
            'test_accuracy_hist': test_accuracy_hist,
            }
    df = pd.DataFrame(data)
    df.to_csv(manifest_dir / 'imagenethistory.csv', index=False)

    model.load_state_dict(torch.load(manifest_dir / 'imageonly_best_model_by_accuracy.pth'))
    evaluation(train_dataloader, model)
    evaluation(test_dataloader, model)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
