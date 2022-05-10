import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import numpy as np 

import matplotlib.pyplot as plt 
import time
import pylab as pl
from IPython import display

import wandb


# -------------------------------------------------------------------------------------- #

# SGD optimizer
path_lr_1_sgd = 'xiaolinzzz/Assignment 3/21oj0lgc'        # learning rate --> 1
path_lr_01_sgd = 'xiaolinzzz/Assignment 3/1qa9b4bb'       # learning rate --> 0.1
path_lr_001_sgd = 'xiaolinzzz/Assignment 3/1q6yco86'      # learning rate --> 0.01
path_lr_0001_sgd = 'xiaolinzzz/Assignment 3/32cyco4x'     # learning rate --> 0.001


# ADAM optimizer
path_lr_1_adam = 'xiaolinzzz/Assignment 3/26hm95zu'       # learning rate --> 1
path_lr_01_adam = 'xiaolinzzz/Assignment 3/1uqd5ec2'      # learning rate --> 0.1
path_lr_001_adam = 'xiaolinzzz/Assignment 3/xay5e4fz'     # learning rate --> 0.01
path_lr_0001_adam = 'xiaolinzzz/Assignment 3/2cwj3bvj'    # learning rate --> 0.001


# -------------------------------------------------------------------------------------- #


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

##Define a test function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    

def get_score(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct


def get_loss(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss


def sgd_optimizer(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    wandb.log({'optimizer': 'SGD'})
    wandb.log({'lr': lr})
    
    return loss_fn, optimizer
    
def adam_optimizer(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    wandb.log({'optimizer': 'Adam'})
    wandb.log({'lr': lr})
    
    return loss_fn, optimizer




# -------------------------------------------------------------------------------------- #
