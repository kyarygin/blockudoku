import torch.nn as nn
import torch
import numpy as np

from collections import Counter

class StateValueNetwork(nn.Module):
    def __init__(self, numChannels):
        super(StateValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.fc1 = nn.Linear(in_features=450, out_features=200)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=200, out_features=1)

        self.bce_loss = nn.BCELoss()

    def forward(self, x, y):
        print(x.size())

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        print(x.size())

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        print(x.size())

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        print(x.size())

        output = self.bce_loss(x, y)
        return output

if __name__ == '__main__':
    with open('train_data/X.npy', 'rb') as f:
        X = np.load(f)
    with open('train_data/y.npy', 'rb') as f:
        y = np.load(f)
    y = 1 * (y >= 3)
    print(Counter(y))
    Xt = torch.from_numpy(X[:100]).type(torch.float)
    yt = torch.from_numpy(y[:100]).type(torch.float)
    Xt = torch.unsqueeze(Xt, dim=1)
    yt = torch.unsqueeze(yt, dim=1)
    model = StateValueNetwork(numChannels=1)
    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.NLLLoss()

