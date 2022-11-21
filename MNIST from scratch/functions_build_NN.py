#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:51:23 2022

@author: tammodukker
"""
import numpy as np
from dnn_utils import *
import math

#%% Create random mini batches

def random_mini_batches(X,Y,mini_batch_size=64,seed=1):
    """
    Creates mini batches of size mini batches from training data X and training labels Y

    Parameters
    ----------
    X : training data of size features x number of examples
    Y : associated training labels of size features x number of examples
    mini_batch_size : minibatch size
        DESCRIPTION. The default is 64.

    Returns
    -------
    mini_batches : dictionary containing the minibatches
    num_mini_batches : number of minibatches

    """

    np.random.seed(seed)
    
    m = X.shape[1]
    mini_batches = {}
    
    # shuffle the datasets 
    permutation = list(np.random.permutation(m))
    shuffled_X, shuffled_Y = X[:,permutation], Y[:,permutation]
    
    inc = mini_batch_size
    
    # Partition
    # Cases with complete mini-batch sizes
    num_complete_minibatches = math.floor(m/inc)
    
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*inc:inc*(k+1)]
        mini_batch_Y = shuffled_Y[:,k*inc:inc*(k+1)]
        mini_batches["X_batch"+str(k)] = mini_batch_X
        mini_batches["Y_batch"+str(k)] = mini_batch_Y
        
    
     # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)    
    if m % mini_batch_size != 0:
      
        mini_batch_X = shuffled_X[:,(m-(m%mini_batch_size)):]
        mini_batch_Y = shuffled_Y[:,(m-(m%mini_batch_size)):]
        
        mini_batches["X_batch"+str(num_complete_minibatches)] = mini_batch_X
        mini_batches["Y_batch"+str(num_complete_minibatches)] = mini_batch_Y
    
    num_mini_batches = math.ceil(m/ mini_batch_size)   
    
    return mini_batches, num_mini_batches
    
#%% Initialize the parameters

def initialize_parameters(layer_dims):
    """
    arguments:
    layer_dims : list containing how many neurons are in each layer
    
    Returns:
    parameters: a dictionary containing the initialized parameters
    
    """
    L = len(layer_dims)-1
    parameters = {}
    for l in range(1,L+1):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.1
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters

#%% Now perform feed forward

def feed_forward(X_train, parameters, layer_dims):
    """
    Parameters
    ----------
    X : Array of (n_x,m) of the training dataset
    parameters : dictionary containing Wi and bi initialized parameters
   layer_dims : list containing how many neurons are in each layer
    
    

    Returns
    AL: output at final node
    store: an array where intermediate activations are stored
    -------
    """
    L = len(layer_dims)-1
    m = X_train.shape[1]
    
    A_prev = X_train
    store = {}
    store["A0"] = X_train
    
    # Implement Relu feedforward on hidden layers
    for l in range(1,L):
        store["Z"+str(l)] = np.dot(parameters["W"+str(l)],A_prev)+parameters["b"+str(l)]
        assert store["Z"+str(l)].shape == (layer_dims[l],m)
       
        store["A"+str(l)] = relu(store["Z"+str(l)])
        A_prev = store["A"+str(l)]
  
    # Implement Softmax on last layer
    store["Z"+str(L)] = np.dot(parameters["W"+str(L)],store["A"+str(L-1)])+parameters["b"+str(L)]
    AL = softmax(store["Z"+str(L)])
         
    return AL,store

#%% Compute cross-entropy cost

def compute_cost(AL, Y):
    """
    computes cross entropy loss
    
    Arguments:
    AL -- probability vector after softmax corresponding to your label predictions, shape (C, number of examples)
    Y -- true "label" vector of one hot encoded training labels shape (C, number of examples)

    Returns:
    cost -- cross-entropy cost

    """
    m = Y.shape[1]
    
    cost = 1/m*np.sum(Y*-np.log(AL))
    cost = np.squeeze(cost) 
    return cost


#%% Now we perform backwards propagation

def back_propagation(AL,Y_train,store,L,parameters):
    """
    Performs backward propagation
    
    Arguments:
    AL -- probability vector corresponding to your label predictions
    Y -- true "label" vector of training data -1 = no heart disease, 1 = hear disease
    store -- storage dictionary containing: Zi and activation Ai of each layer
    L -- number of layers 
    parameters -- a dictionary containing the parameters
    
    returns: 
        grads -- dictionary of gradients
    """
    m = Y_train.shape[1]
   
    grads = {}
        
    # derivative wrt AL of cross entropy
    #grads["dC_dAL"] = -1/m*(Y_train/AL - (1-Y_train)/(1-AL))
    
    # dC_dZL = y_hat - y from softmax
    grads["dC_dZ"+str(L)] = AL-Y_train
    
    assert grads["dC_dZ"+str(L)].shape == store["Z"+str(L)].shape
    
    # Find derivatives wrt W3 and b3
    grads["dC_dW"+str(L)] =  np.dot(grads["dC_dZ"+str(L)],store["A"+str(L-1)].T)
    grads["dC_db"+str(L)] = np.sum(grads["dC_dZ"+str(L)],axis = 1, keepdims=True)
    
    assert grads["dC_dW"+str(L)].shape == parameters["W"+str(L)].shape
    assert grads["dC_db"+str(L)].shape == parameters["b"+str(L)].shape
    
    # Now backpropagate past Relu layers
    
    for l in reversed(range(1,L)):
        # derivative wrt activation
        grads["dC_dA"+str(l)] = np.dot(parameters["W"+str(l+1)].T,grads["dC_dZ"+str(l+1)])
        
        # derivative wrt z : dC_dZl = dC_dAl*dAl_dZl(=derivative Relu)
        grads["dC_dZ"+str(l)] = grads["dC_dA"+str(l)]*derivative_relu(store["Z"+str(l)])
        
        assert grads["dC_dZ"+str(l)].shape == store["Z"+str(l)].shape
        
        # Find derivative wrt Wl & bl
        grads["dC_dW"+str(l)] = np.dot(grads["dC_dZ"+str(l)],store["A"+str(l-1)].T)
        grads["dC_db"+str(l)] = np.sum(grads["dC_dZ"+str(l)],axis = 1, keepdims=True)
        
        assert grads["dC_dW"+str(l)].shape == parameters["W"+str(l)].shape
        assert grads["dC_db"+str(l)].shape == parameters["b"+str(l)].shape
        
    return grads

#%% Update Parameters

def update_parameters(params, grads, learning_rate, L):
    
    """
    Update Wi and bi using gradient descent

    Parameters
    ----------
    params : python dictionary
        dictionary containing Wi and bi obtained from forward pass
    grads : python dictionary
        dictionary contraining the gradients obtained from backward pass
    learning_rate : float
        learning rate for gradient descent

    Returns
    -------
    Parameters: python dictionary
        dictionary containing updated values of parameters from gradient decsent

    """
    parameters = params.copy()
    for l in range(1,L+1):
        parameters["W"+str(l)] = params["W"+str(l)]-learning_rate*grads['dC_dW'+str(l)]
        parameters["b"+str(l)] = params["b"+str(l)]-learning_rate*grads["dC_db"+str(l)]
        
    return parameters

     
#%% Brining it all together in one function that builds and trains a neural network

def NN_model_train(X_train, Y_train, X_val, Y_val, layer_dims, learning_rate = 0.0075, epochs = 3, mini_batch_size = 64, seed = 1009, print_cost=False):
    """
    Build and train a neural network 

    Parameters
    ----------
    X_train : array of training data of size(n_x,m)
    Y_train : array of true label data of size(1,m)
    layers_dims : list of number of neurons per layer
    learning_rate :  float setting the learning rate for gradient descent
    epochs : Tnumber of times gradient descent is performed
    print_cost : print the cost or not

    Returns
    -------
    parameters -- a dictionary with the trained parameter with which predictions can be made
    costs -- list of cost

    """
    assert X_train.shape[0] == layer_dims[0], "mismatch between number of input neurons and features"
    assert Y_train.shape[0] == layer_dims[-1], "mismatche between number of output neurons and true label vector"
    
    # number of layers
    L = len(layer_dims)-1
    # number of training examples
    m = X_train.shape[1]
    # list to keep track of costs
    costs = []
    costs_val = []
    
    # Setting seed of how mini batches are fixed. Model is very sensitive to this value somehow
    
    
    #initialize parameters
    parameters = initialize_parameters(layer_dims)
    
    # loop (gradient descent)
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        
        # define the random minibatches
        seed+=1
        mini_batches, num_mini_batches = random_mini_batches(X_train,Y_train,mini_batch_size,seed=seed)
        
        # minibatch gradient descent
        for k in range(num_mini_batches):
            
            # select minibatch
            mini_batch_X = mini_batches["X_batch"+str(k)]
            mini_batch_Y = mini_batches["Y_batch"+str(k)]
        
            # feedforward
            AL, store = feed_forward(mini_batch_X, parameters, layer_dims)
            
            # compute cost
            cost = compute_cost(AL, mini_batch_Y)
            costs.append(cost)
            
            # backpropagation
            grads = back_propagation(AL,mini_batch_Y,store,L,parameters)
            # update parameters
            parameters = update_parameters(parameters, grads, learning_rate, L)
            
            
            # Print cost and test on validation set every 100 batches
            if print_cost == True:
                if k % 100 == 0:
                    loss, current = cost, k * mini_batch_size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{m:>5d}]")
                    
                    # test on validation set
                    cost_val, acc_val = validate(parameters, X_val, Y_val, layer_dims)
                    costs_val.append(cost_val)
    print("Done!")        
    return parameters, costs, costs_val

#%% Predict

def predict(parameters, X_test, layer_dims):
    """
    function that with trained parameters predicts associated labels of data

    Parameters
    ----------
    parameters : trained parameters from NN_model train
    X_test : the dataset of which one would like to predict the labels
    threshold: value between 0 and 1 which determines cutoff point when label 1 is assigned and otherwise 0
    layer_dims :  list of layer dimensions in neural network

    Returns
    -------
    y_hat = predictions of labels

    """
    AL_test,store_test = feed_forward(X_test, parameters, layer_dims)
    Y_hat = AL_test.argmax(0)
    
    return Y_hat

#%% Validate

def validate(parameters, X_val, Y_val, layer_dims):
    AL_val, store_val = feed_forward(X_val, parameters, layer_dims)
    cost_val = compute_cost(AL_val, Y_val)
    
    Y_hat_val = AL_val.argmax(0)
    val_accuracy = np.sum(np.where(Y_hat_val==Y_val.argmax(0),1,0))/Y_val.shape[1]
    
    return cost_val, val_accuracy