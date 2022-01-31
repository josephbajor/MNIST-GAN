from cgi import test
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F


class ConvDescriminator(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, hparams.kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, hparams.kernel_size)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sig(x)
        return x


class BscGenerator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fc1 = nn.Linear(hparams.z_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, 28, 28)
        x = torch.unsqueeze(x, 1)
        return x


class ConditionalGenerator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, 28, 28)
        x = x.unsqueeze(1)
        return x
