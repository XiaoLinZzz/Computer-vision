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


 
    



def sgd_optimizer(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # wandb.log({'optimizer': 'SGD'})
    # wandb.log({'lr': lr})
    
    return loss_fn, optimizer
    
def adam_optimizer(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # wandb.log({'optimizer': 'Adam'})
    # wandb.log({'lr': lr})
    
    return loss_fn, optimizer




# -------------------------------------------------------------------------------------- #


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')