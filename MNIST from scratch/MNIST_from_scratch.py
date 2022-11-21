#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:26:14 2022

@author: tammodukker
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# call fucntions that train & build neural network
from functions_build_NN import *

# %% call Minst dataset
import sklearn
from sklearn.datasets import fetch_openml
X, y_fetch = fetch_openml("mnist_784", version=1, return_X_y=True)

#%%
# Plot MNIST
X_image = X.to_numpy().reshape((X.shape[0], 28, 28))
num = 10
images = X_image[:num]
labels = y_fetch[:num]

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

# %% Manipulate Data: Normalize, one hot encode the labels and split into training, validation & test set

# Normalize X
X = X.to_numpy() / np.max(X.to_numpy())

# one hot encode the labels
y = np.array(list(map(int, y_fetch.T)))
Y = np.zeros((10, X.shape[0]))
for i in range(10):
    Y[i, np.where(y == i)] = 1

# split the data
split = [0, int(X.shape[0]*0.8), int(X.shape[0]-1000), int(X.shape[0])]
X_train, Y_train = X[split[0]:split[1], :].T, Y[:, split[0]:split[1]]
X_val, Y_val = X[split[1]:split[2], :].T, Y[:, split[1]:split[2]]
X_test, Y_test = X[split[2]:split[3], :].T, Y[:, split[2]:split[3]]

# %% Train the neural network

# Hyperparameters
epochs = 2
batch_size = 64
layer_dims = [X.shape[1], 512, 300, 200, 10]

# Call training function
parameters, costs, cost_val = NN_model_train(
    X_train, Y_train, X_val, Y_val, layer_dims, epochs=epochs, seed=1009, print_cost=True)

# %% Test accuracy

y_hat = predict(parameters, X_test, layer_dims)
labels = Y_test.argmax(0)

test_accuracy = np.sum(np.where(y_hat == labels, 1, 0))/Y_test.shape[1]
print("test set accuracy is: ", test_accuracy*100, "%")

# Plot prediction of 10 images
picked = np.random.randint(0, X_test.shape[1], size=10)
X_picked = X_test[:, picked]
y_hat = predict(parameters, X_picked, layer_dims)

X_image = X_test[:, picked].T
images = X_image.reshape((X_image.shape[0], 28, 28))
labels = Y_test[:, picked].argmax(0)

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

# %% Plot costs
num_batches = len(costs)
plt.figure()
plt.plot(np.linspace(0, num_batches, num_batches),
         costs, label="training cost")
plt.plot(np.linspace(0, num_batches, num_batches//100+1),
         cost_val, label="validation cost every 100 batches")
plt.xlabel("Number of batches passed")
plt.ylabel("Cost")
plt.legend()
