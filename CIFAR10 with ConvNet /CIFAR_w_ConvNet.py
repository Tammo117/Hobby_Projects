#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:30:03 2022

@author: tammodukker
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
from torchsummary import summary

# %% call CIFAR dataset
train_dataset = datasets.CIFAR10(
    'data', train=True, transform=ToTensor(), download=True)
test_dataset = datasets.CIFAR10(
    'data', train=False, transform=ToTensor(), download=True)

# %% Plot CIFAR
num = 10
images = train_dataset.data[:num]
labels = train_dataset.targets[:num]

print(images.shape)

num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
for i in range(num):
    ax = axes[i//num_col, i % num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

# %% Hyperparameters
batch_size = 64
epochs = 1

in_channels = 3
hidden_channels = [64, 128, 256, 512]
out_features = 10

# %% Create data loaders.
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %% ResNetBlock


class PreActResNetBlock(nn.Module):

    def __init__(self, c):
        """
        Inputs:
            c - Number of input channels
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        z = self.net(x)
        out = x + z
        return out

# %% Build a Convolutional NN


class ConvNet(nn.Module):
    """
   # Convolutional Neural Network that resembles the VGG Network

    #Args:
     #   in_channels: number of features of the input image ("depth of image")
       # hidden_channels: number of hidden features ("depth of convolved images")
      #  out_features: number of features in output layer
    """

    def __init__(self, in_channels, hidden_channels, out_features):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            # Convolutional layer 0
            nn.Conv2d(
                in_channels, hidden_channels[0], kernel_size=3, padding=1, stride=1),
            PreActResNetBlock(hidden_channels[0]),

            # Convolutional layer 1
            nn.Conv2d(hidden_channels[0], hidden_channels[1],
                      kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Convolutional layer 2
            PreActResNetBlock(hidden_channels[1]),
            PreActResNetBlock(hidden_channels[1]),
            nn.Conv2d(hidden_channels[1], hidden_channels[2],
                      kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Convolutional layer 3
            PreActResNetBlock(hidden_channels[2]),
            PreActResNetBlock(hidden_channels[2]),
            nn.Conv2d(hidden_channels[2], hidden_channels[3],
                      kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 4
            PreActResNetBlock(hidden_channels[3]),
            PreActResNetBlock(hidden_channels[3]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 5
            PreActResNetBlock(hidden_channels[3]),
            PreActResNetBlock(hidden_channels[3]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Linear layer
            nn.Flatten(),
            nn.Linear(hidden_channels[3], out_features)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


model = ConvNet(in_channels, hidden_channels, out_features)
summary(model, (3, 32, 32), device='cpu')
# print(model)

# %% Cost function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

# %% Create a train loop


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        # Forward Pass
        pred = model(X)
        # get cost
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print cost every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# %% Create evaluation


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


# %% Train
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    correct = test(test_dataloader, model, loss_fn)
print("Done!")

# %% Plot prediction of 10 images in last batch

picked = np.random.randint(64, 1064, size=10)
X_image = torch.from_numpy(test_dataset.data[picked])

X_image = torch.empty((64, 3, 32, 32))
labels = torch.empty((64))
for batch, (X, y) in enumerate(test_dataloader):
    X_batch = X
    y_batch = y
    X_image = torch.cat((X_image, X), dim=0)
    labels = torch.cat((labels, y), dim=0)

X_image = X_image[picked]
labels = labels[picked]

y_hat = model(X_image).argmax(1)
images = X_image.permute((0, 2, 3, 1))

num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
for i, ind in enumerate(picked):
    ax = axes[i//num_col, i % num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Pred: {}'.format(y_hat[i])+" label: {}".format(labels[i]))
plt.tight_layout()
plt.show()
plt.title(f"Test Accuracy: {(100*correct):>0.1f}%")
plt.savefig('ConvNet_prediction.png')

# save the model
torch.save(model, "ConvNet_model.pt")
