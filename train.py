import itertools
import math
import time
import tqdm
import os

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

from vae import VAE, VAELoss
from params import param



def train(data_loader):
    graph = VAE(in_dim=param.img_size ** 2, hidden_dim=300, out_dim=20).to(DEVICE)
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=param.lr)
    vae_loss = VAELoss()
    epoch = 0

    while epoch < param.stop_epoch:
        for batch_data in data_loader:
            optimizer.zero_grad()
            images, labels = batch_data

            input_img = images.to(DEVICE).view(-1, param.img_size ** 2)
            output, mu, sigma = graph(input_img)

            loss = vae_loss(output, input_img, mu, sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(graph.parameters(), 2.0)
            optimizer.step()

        epoch += 1
    
        print('[TRAIN] EPOCH : {} / LOSS : {}'.format(epoch, loss))

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([          
            transforms.ToTensor(),                     
            transforms.Normalize(mean=(0,), std=(1,))
    ])

    train_dataset = dsets.MNIST(root='./data/', train=True, download=True, transform=transform) #전체 training dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True) # dataset 중에서 샘플링하는 data loader입니다

    train(train_loader)

    