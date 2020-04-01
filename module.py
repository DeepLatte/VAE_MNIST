import torch
import torch.nn as nn

def init_linear(in_dim, out_dim, init=True):
    mod = nn.Linear(in_dim, out_dim)
    if init:
        nn.init.xavier_normal_(mod.weight)
    
    return mod

