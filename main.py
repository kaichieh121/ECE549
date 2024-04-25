import torch, torchaudio, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from CustomDataset import FlickrDataset
from L3Net import L3Net

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
    parser.add_argument('--data_dir', default="D:\\datasets\\flickr_soundnet")
    parser.add_argument('--output_dir', default="D:\\Projects\\ECE549\\manifest")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    return args

def create_dataset_csv(data_dir, output_dir, fold = 0):
    fold_dir_list = [x for x in data_dir.iterdir()]
    train_data = []
    test_data = []
    columns = ['audio_path', 'fig_path']
    for fold_dir in fold_dir_list:
        if fold_dir.name == f'{fold}':
            # test set
            audio_file_list = [x for x in fold_dir.iterdir() if '.wav' in x.name]
            for audio_file in audio_file_list:
                fig_file = fold_dir / f'{audio_file.name.split(".wav")[0]}.jpg'
                test_data.append([audio_file, fig_file])
        else:
            # training set
            audio_file_list = [x for x in fold_dir.iterdir() if '.wav' in x.name]
            for audio_file in audio_file_list:
                fig_file = fold_dir / f'{audio_file.name.split(".wav")[0]}.jpg'
                train_data.append([audio_file, fig_file])
    train_df = pd.DataFrame(train_data, columns=columns)
    test_df = pd.DataFrame(test_data, columns=columns)
    train_df.to_csv(output_dir / 'train.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)


def train(dataloader, model, loss_fn, optimizer, loss_history = [], accuracy_history = []):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    cur_size = 0
    loss_avg = []
    accuracy_avg = []
    for batch, sample_batched in enumerate(dataloader):

        audio = sample_batched[0].to(device)
        image = sample_batched[2].to(device)
        y = sample_batched[3].to(device)

        # Compute prediction error
        pred = model(audio, image)
        loss = loss_fn(pred, y)

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        cur_size += pred.shape[0]
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
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
            image = sample_batched[2].to(device)
            y = sample_batched[3].to(device)
            pred = model(audio, image)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    test_loss_hist.append(test_loss)
    test_accuracy_hist.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss_hist, test_accuracy_hist, test_loss, correct

def main(args):
    data_dir = Path(args.data_dir) / "Data"
    output_dir = Path(args.output_dir)
    batch_size = args.batch_size
    epochs = args.epochs

    audio_transform = nn.Sequential(
        torchaudio.transforms.Resample(22050, 48000),
        torchaudio.transforms.Spectrogram(n_fft=512)
    )

    image_transform = None


    create_dataset_csv(data_dir, output_dir)
    train_dataset = FlickrDataset(output_dir / 'train.csv', p=0.5, audio_transform=audio_transform, image_transform=image_transform)
    test_dataset = FlickrDataset(output_dir / 'test.csv', p=0.5, audio_transform=audio_transform, image_transform=image_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    model = L3Net().to(device)
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
            torch.save(model.state_dict(), output_dir / 'best_model_by_accuracy.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            print("Saving Best Model by loss...")
            torch.save(model.state_dict(), output_dir / 'best_model_by_loss.pth')
    torch.save(model.state_dict(), output_dir / 'last_model.pth')
    print("Done!")

    data = {'train_loss_hist': train_loss_hist,
            'test_loss_hist': test_loss_hist,
            'train_accuracy_hist': train_accuracy_hist,
            'test_accuracy_hist': test_accuracy_hist,
            }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'history.csv', index=False)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
