import torch, torchaudio, torchvision
from torch import nn

def debug_sequential(model, input_tensor):
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(m, output.shape)
    return output

class L3Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.image_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(24),
        )


        self.audio_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((24, 16)),
        )

        self.fc1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, audio, image):
        # debug_sequential(self.audio_stack, audio)
        audio_feature = self.audio_stack(audio)
        audio_feature = self.flatten(audio_feature)
        # debug_sequential(self.image_stack, image)
        image_feature = self.image_stack(image)
        image_feature = self.flatten(image_feature)
        hidden_feature = torch.cat((image_feature, audio_feature), axis=1)
        hidden_feature = self.relu(self.fc1(hidden_feature))
        logits = self.fc2(hidden_feature)
        return logits


class L3NetAudioOnly(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.L3Net = L3Net()
        self.flatten = nn.Flatten()
        self.audio_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((24, 16)),
        )

        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 50)
        self.load_weights(weights)

    def load_weights(self, weights):
        if weights is not None:
            self.L3Net.load_state_dict(weights)
            self.audio_stack.load_state_dict(self.L3Net.audio_stack.state_dict())
        del(self.L3Net)
    def forward(self, audio):
        # debug_sequential(self.audio_stack, audio)
        audio_feature = self.audio_stack(audio)
        audio_feature = self.flatten(audio_feature)
        audio_feature = self.relu(self.fc1(audio_feature))
        logits = self.fc2(audio_feature)
        return logits


class L3NetImageOnly(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.flatten = nn.Flatten()
        self.L3Net = L3Net()
        self.image_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(24),
        )

        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1000)
        self.load_weights(weights)


    def load_weights(self, weights):
        if weights is not None:
            self.L3Net.load_state_dict(weights)
            self.image_stack.load_state_dict(self.L3Net.image_stack.state_dict())
        del(self.L3Net)

    def forward(self, image):
        # debug_sequential(self.image_stack, image)
        image_feature = self.image_stack(image)
        image_feature = self.flatten(image_feature)
        image_feature = self.flatten(image_feature)
        hidden_feature = self.relu(self.fc1(image_feature))
        logits = self.fc2(hidden_feature)
        return logits