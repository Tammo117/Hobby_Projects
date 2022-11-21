#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:18:07 2022

@author: tammodukker
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:20:16 2022

@author: tammodukker
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
from torchsummary import summary
from torch.nn.functional import one_hot
from torch.autograd import Variable

#%% Get data
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

#%% prepare the dataset of input to output pairs encoded as integers

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_sequences = len(dataX)
print("Total sequences: ", n_sequences)


#%% Hyperparameters
batch_size = 64
epochs = 2

#%%reshape X to be [sequences, time steps, features]

X = np.reshape(dataX, (n_sequences, seq_length))

# Turn X into tensor
X = torch.LongTensor(X)

# one hot encode X
X_oh = one_hot(X,-1)
# one hot encode the output variable
dataY = torch.LongTensor(dataY)
y_oh = one_hot(dataY,-1)

# creat batches
X_batch = {}
Y_batch = {}
l = 0
for i in range(0,X_oh.shape[0],batch_size):
    X_batch["batch"+str(l)] = X[i:i+batch_size]
    Y_batch["batch"+str(l)] = dataY[i:i+batch_size]
    l+=1

#%% Build LSTM

class LSTM(nn.Module):
    """
 
    """
    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2):
        super(LSTM, self).__init__()
        
        # defintions
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.num_hidden = lstm_num_hidden
        self.num_layers = lstm_num_layers
        
        # attempt with embedding. 
        self.Embedding = nn.Embedding(num_embeddings=vocabulary_size, 
                                      embedding_dim=vocabulary_size)
        
        # layers
        self.lstm = nn.LSTM(
            input_size = vocabulary_size,
            hidden_size = lstm_num_hidden,
            num_layers = lstm_num_layers,
            batch_first = True
            )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(lstm_num_hidden*seq_length, vocabulary_size)
        
    def forward(self, x):
        
        #h_0 = Variable(torch.zeros((self.num_layers, x.shape[0], self.num_hidden), dtype = torch.long)) # intital hidden state of shape [num_layers, batch_size, hidden_size]
        #c_0 = Variable(torch.zeros((self.num_layers, x.shape[0], self.num_hidden), dtype = torch.long))
        
        x = self.Embedding(x)
    
        # input of size [batch, seq_length, embedding_dim]
        lstm_out, (h_n, c_n)  = self.lstm(x)
        # lstm out is now of shape [64, 100, 256] now. Force it to be [64, 100*256] with flatten
        flatten_out = self.flatten(lstm_out)
        logits = self.classifier(flatten_out) # Now should be size [64, 45] no?
         
        return logits
    
model = LSTM(batch_size = batch_size, seq_length = seq_length, vocabulary_size = n_vocab)

#print(model)
#summary(model, (batch_size, seq_length, n_vocab), device='cpu')

#%% Cost function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)  

#%% Create a train loop

def train(X_batch, Y_batch, model, loss_fn, optimizer):
    size = len(dataX)
    model.train()
    for i in range(l):
        # loop over batches
        X = X_batch["batch"+str(i)] # shape [64,100]
        Y = Y_batch["batch"+str(i)] # shape [64,45]
                
        # Forward Pass
        pred = model(X)
        #get cost
        loss = loss_fn(pred,Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Use gradient clippping to circumvent exploding gradients problem
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        # Print cost every 100 batches
        if i % 100 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
#%% Train

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(X_batch, Y_batch, model, loss_fn, optimizer)
print("Done!")

#%%Save
torch.save(model, 'model_2.pth')

#%% Generate new sequences

model = torch.load('model_2.pth') 

# take a random 100 charachter part in the book
start = np.random.randint(0, len(dataX)-1)
pattern_start = dataX[start]

# generate an additional 500 characters 
pattern = pattern_start.copy()
for i in range(500):
    x = np.reshape(pattern[i:], (1, seq_length))
    x = torch.LongTensor(x)
    pred = model.forward(x)
    index = torch.argmax(pred, axis = -1).item()
    result = int_to_char[index]
    pattern.append(index)
    
print("<seed:>","\"", ''.join([int_to_char[value] for value in pattern_start]), "\"")  
print("<generated:>","\"", ''.join([int_to_char[value] for value in pattern[seq_length:]]), "\"")  

	 
            