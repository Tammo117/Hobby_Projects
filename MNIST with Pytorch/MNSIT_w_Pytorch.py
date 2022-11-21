#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:39:32 2022

@author: tammodukker
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# for mac
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



# call Mnist dataset
train_dataset = datasets.MNIST(
    'data', train=True, transform=ToTensor(), download=True)
test_dataset = datasets.MNIST(
    'data', train=False, transform=ToTensor(), download=True)

# %% Plot MNIST
num = 10
images = train_dataset.data[:num]
labels = train_dataset.targets[:num]

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
epochs = 3
layer_dims = [train_dataset.data.shape[1]*train_dataset.data.shape[2],
              512, 300, 200, len(train_dataset.classes)]

# %% Create data loaders.
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %%  Build a NN


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]),
            nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]),
            nn.ReLU(),
            nn.Linear(layer_dims[3], layer_dims[-1]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
print(model)

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


# %% Train
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# %% Plot prediction of 10 images

picked = np.random.randint(0, test_dataset.data.shape[0], size=10)
X_image = test_dataset.data.float()[picked]
y_hat = model(X_image).argmax(1).numpy()

images = test_dataset.data[picked]
labels = test_dataset.targets[picked]

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

# %%
