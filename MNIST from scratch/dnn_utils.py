#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:54:16 2022

@author: tammodukker
"""
import numpy as np


#%% Activation functions
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    Arguments:
    Z -- numpy array of any shape
    Returns:
    A -- output of sigmoid(z), same shape as Z
    """
    A = 1/(1+np.exp(-Z))
    
    assert A.shape == Z.shape

    return A

def relu(Z): 
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)
    
    assert A.shape == Z.shape
    
    return A

def derivative_sigmoid(Z):
    """
    Calculate derivative of sigmoid unit
    
    Z -- input of that layer
    da_dz -- derivative of activation wrt input (Z)
    
    """
    da_dz = sigmoid(Z)*(1-sigmoid(Z))
    
    assert da_dz.shape == Z.shape
    
    return da_dz

def derivative_relu(Z):
    """
    Calculate derivative of relu unit
    
    Z-- input of that layer
    da_dz -- derivative of activation wrt to input(Z)

    """
    da_dZ = np.where(Z<0,0,1) 
    
    assert da_dZ.shape == Z.shape
        
    return da_dZ

def softmax(Z):
    """
    Calculate numerically stable softmax from input vector Z

    Parameters
    ----------
    Z : input of last neuron layer of shape (classes x num examples)

    Returns
    -------
    AL : probablility of each class of shape (classes x num examples)

    """
    Z_norm = Z-np.max(Z,axis=0)
    t = np.exp(Z_norm)
    AL = t/np.sum(t,axis=0)
    
    return AL