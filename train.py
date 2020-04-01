import itertools
import math
import time
from tqdm import tqdm
import os

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from vae import VAE, VAELoss
from params import param
from utils import plot_reproduce

def model_save(graph, epoch, optimizer, loss, network_path, tag):
    model_path = os.path.join(network_path, 'trained')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': graph.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(model_path, 'trained_{}.pth'.format(tag)))


def train(data_loader, network_path, writer):
    graph = VAE(in_dim=param.img_size ** 2, hidden_dim=500, out_dim=param.out_dim).to(DEVICE)
    graph.train()
    optimizer = torch.optim.Adam(graph.parameters(), lr=param.lr)
    vae_loss = VAELoss()
    epoch = 0
    global_step = 0
    val_loss_min = 999999.

    for epoch in tqdm(range(param.stop_epoch)):
        temp_val_loss = 0
        for i, batch_data in enumerate(data_loader):
            if i < (8 * len(data_loader)//10):
                graph.train()
                optimizer.zero_grad()
                images, labels = batch_data

                input_img = images.to(DEVICE).view(-1, param.img_size ** 2)
                output, mu, sigma = graph(input_img)

                loss, mrg_likelihood, kl_div = vae_loss(output, input_img, mu, sigma)
                writer.add_scalar('train/loss', loss, global_step=global_step)
                writer.add_scalar('train/mrg_likelihood', mrg_likelihood, global_step=global_step)
                writer.add_scalar('train/kl_div', kl_div, global_step=global_step)

                loss.backward()
                nn.utils.clip_grad_norm_(graph.parameters(), 2.0)
                optimizer.step()
                global_step += 1
                if global_step % 10000 == 0:
                    model_save(graph, epoch, optimizer, loss, network_path, global_step)
                    plot_reproduce(input_img[:100], output[:100], 10, global_step)
                print('[TRAIN] G_S : {} / LOSS : {} / Marg : {} / kl : {}'.format(global_step, loss, mrg_likelihood, kl_div))
            else:
                graph.eval()
                with torch.no_grad():
                    images, labels = batch_data

                    val_input_img = images.to(DEVICE).view(-1, param.img_size ** 2)
                    val_output, mu, sigma = graph(val_input_img)
                    loss, mrg_likelihood, kl_div = vae_loss(val_output, val_input_img, mu, sigma)
                    writer.add_scalar('test/loss', loss, global_step=global_step)
                    writer.add_scalar('test/mrg_likelihood', mrg_likelihood, global_step=global_step)
                    writer.add_scalar('test/kl_div', kl_div, global_step=global_step)
                    temp_val_loss += loss
        
        temp_val_loss /= (len(data_loader) - (8 * len(data_loader)//10))

        if temp_val_loss < val_loss_min:
            val_loss_min = temp_val_loss
            print("val_model_saved")
            model_save(graph, epoch, optimizer, temp_val_loss, network_path, "val")
            plot_reproduce(val_input_img[:100], val_output[:100], 10, "{}".format(epoch))
        epoch += 1

        

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_path = os.path.abspath('../saved')

    transform = transforms.Compose([          
            transforms.ToTensor(),                     
            transforms.Normalize(mean=(0,), std=(1,))
    ])
    writer = SummaryWriter(network_path)
    if not os.path.exists(network_path):
        os.mkdir(network_path)
    train_dataset = dsets.MNIST(root='../MNISTdata/', train=True, download=True, transform=transform) #전체 training dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False) # dataset 중에서 샘플링하는 data loader입니다

    train(train_loader, network_path, writer)

    