# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:59:56 2019

@author: David Abitbol, Renze Li, Feijiao Yuan, Zhihao Yang
"""

import torch
import torch.autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np

class MLP(nn.Module):
    def __init__(self, size_list, dropout = False, dropoutProb = 0.1, batchNorm = False):
        super(MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.ReLU())
            
            if batchNorm:
                layers.append(nn.BatchNorm1d(size_list[i+1]))
                
            if dropout:
                layers.append(nn.Dropout(p = dropoutProb))
            
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        
        # Unpack the list
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RNN(nn.Module):
    def __init__(self, size_list):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=size_list[0],
            hidden_size=size_list[1],         # rnn hidden unit
            num_layers=len(size_list)-2,           # number of rnn layer
            )

        self.out = nn.Linear(size_list[-2], size_list[-1])

        

    def forward(self, x):
        
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
    

def train_epoch(model, optimizer, X_train, y_train, criterion):
    model.train()
    
    start_time = time.time()

    optimizer.zero_grad()  

    outputs = model(X_train)
    
    loss = criterion(outputs, y_train)
    
    running_loss = loss.item()

    loss.backward()
    
    optimizer.step()
    
    end_time = time.time()
    
    #print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss




# Optimizer can be 'SGD', 'RMSprop', 'ADAM', 
def nnTrain(X_train, y_train, size_list, dropout = False, dropoutProb = 0.1, batchNorm = False, 
                optimizer = 'SGD', lr = 0.01, n_epochs = 100, LSTM = False):   

    X_train = torch.autograd.Variable(torch.Tensor(X_train.values.astype(float)))
    y_train = torch.autograd.Variable(torch.Tensor(y_train.values.astype(float)))
    
    if LSTM == True:
        X_train = X_train.view(-1, X_train.shape[0], X_train.shape[1])
        model = RNN(size_list)
    else:
        model = MLP(size_list, dropout, dropoutProb, batchNorm)
    
    
    criterion = nn.MSELoss()
    
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = lr)
    if optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr = lr)
    if optimizer == 'ADAM':
        optimizer = optim.ADAM(model.parameters(), lr = lr)
    
    Train_loss = []
    
    for i in range(n_epochs):
        train_loss = train_epoch(model, optimizer, X_train, y_train, criterion)
        Train_loss.append(train_loss)
    
    return model, Train_loss


def nnTest(model, X_test, y_test):
    X_test = torch.autograd.Variable(torch.Tensor(X_test.values.astype(float)))
    y_test = torch.autograd.Variable(torch.Tensor(y_test.values.astype(float)))
    criterion = nn.MSELoss()
    
    if hasattr(model, 'rnn'):
        X_test = X_test.view(-1, X_test.shape[0], X_test.shape[1])
    
    with torch.no_grad():
        model.eval()        

        outputs = model(X_test)

        loss = criterion(outputs, y_test).detach()
        running_loss = loss.item()
        
        return running_loss
    
def nnPredict(model, X_test):
    X_test = torch.autograd.Variable(torch.Tensor(X_test.values.astype(float)))
    
    if hasattr(model, 'rnn'):
        X_test = X_test.view(-1, X_test.shape[0], X_test.shape[1])
    
    with torch.no_grad():
        model.eval()        
        outputs = model(X_test)
    return np.array(outputs).flatten()
    