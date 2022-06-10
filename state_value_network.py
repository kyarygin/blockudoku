import torch.nn as nn
import torch
import numpy as np

from collections import Counter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from typing import List

import os
from tqdm import tqdm

BATCH_SIZE = 4096
EPOCHS = 20
DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda')


def load_datasets(data_path: str='train_data', valid_share: float=0.2, seed: int=0) -> List[Dataset]:
    train_share = 1 - valid_share
    with open(os.path.join(data_path, 'X.npy'), 'rb') as f:
        X_np = np.load(f)
    with open(os.path.join(data_path, 't.npy'), 'rb') as f:
        t_np = np.load(f)
    t_np = 1 * (t_np >= 3)
    X = torch.from_numpy(X_np).type(torch.float).unsqueeze(dim=1)
    t = torch.from_numpy(t_np).type(torch.float).unsqueeze(dim=1)
    dataset = TensorDataset(X, t)
    n_train_samples = round(X.size(0) * train_share)
    n_valid_samples = round(X.size(0) * valid_share)
    train_dataset, valid_dataset = random_split(
        dataset,
        [n_train_samples, n_valid_samples],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, valid_dataset

class StateValueNetwork(nn.Module):
    def __init__(self):
        super(StateValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.linear1 = nn.Linear(in_features=450, out_features=200)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=200, out_features=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.sigm(x)

        return x

if __name__ == '__main__':
    train_dataset, valid_dataset = load_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = StateValueNetwork().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss = nn.BCELoss()

    EPOCHS = 10
    for epoch in range(EPOCHS):
        train_loss = []
        valid_loss = []

        model.train()
        for X_batch, t_batch in tqdm(train_dataloader):
            X_batch, t_batch = X_batch.to(DEVICE), t_batch.to(DEVICE)
            y_batch = model(X_batch)
            batch_loss = loss(y_batch, t_batch)
            batch_size = X_batch.size(0)
            train_loss.append((batch_loss.item(), batch_size))

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        model.eval()
        for X_batch, t_batch in tqdm(valid_dataloader):
            X_batch, t_batch = X_batch.to(DEVICE), t_batch.to(DEVICE)
            y_batch = model(X_batch)
            batch_loss = loss(y_batch, t_batch)
            batch_size = X_batch.size(0)
            valid_loss.append((batch_loss.item(), batch_size))

        mean_train_loss = sum(loss*size for loss, size in train_loss) / sum(size for _, size in train_loss)
        mean_valid_loss = sum(loss*size for loss, size in valid_loss) / sum(size for _, size in valid_loss)
        print(mean_train_loss, mean_valid_loss)

    torch.save(model.state_dict(), f'models/de_novo.valid_{mean_valid_loss:.3f}.model')


