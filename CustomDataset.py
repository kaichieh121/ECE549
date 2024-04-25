import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class FlickrDataset(Dataset):
    def __init__(self, csv_file, audio_transform=None, image_transform=None, p = 1):
        self.pairs = pd.read_csv(csv_file)
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.p = p

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path = Path(self.pairs.at[idx, 'audio_path'])
        audio_name = audio_path.name.split('.wav')[0]
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0, sr:sr*2].unsqueeze(dim=0)
        if np.random.rand(1)[0] < self.p:
            random_img_idx = idx
        else:
            random_img_idx = np.random.randint(0, len(self.pairs)-1)
        image_path = Path(self.pairs.at[random_img_idx, 'fig_path'])
        image_name = image_path.name.split('.jpg')[0]
        image = read_image(image_path.__str__())
        image = image.type(torch.FloatTensor)
        label = 1 if audio_name == image_name else 0
        if self.audio_transform:
            audio = self.audio_transform(audio)
            audio = torch.log(audio + 1e-6)
        if self.image_transform:
            image = self.image_transform(image)
        return audio, sr, image, label


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, audio_transform=None):
        csv_file = data_dir / 'meta' / 'esc50.csv'
        self.audio_dir = data_dir / 'audio'
        self.labels = pd.read_csv(csv_file)
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.audio_dir / self.labels.at[idx, 'filename']
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0, sr:sr*2].unsqueeze(dim=0)
        label = self.labels.at[idx, 'target']
        if self.audio_transform:
            audio = self.audio_transform(audio)
            audio = torch.log(audio + 1e-6)
        return audio, label