import os

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

from vae import VAE
from utils import sampleOut, fashion_scatter
from params import param

def test(data_loader, model_path):
    graph = modelLoad(model_path)
    fashion_tsne = []
    label_list = []
    for batch in tqdm(data_loader):
        images, labels = batch
        images = images.to(DEVICE).view(-1, param.img_size ** 2)
        _, mu, sigma = graph(images)
        
        z = sampleOut(np.array(mu.data.cpu()), np.array(sigma.data.cpu()))
        assert len(z) == len(labels)
        fashion_tsne.extend(TSNE(random_state=123).fit_transform(z))
        label_list.extend(np.array(labels.data.cpu()))
    
    fashion_tsne = np.stack(fashion_tsne, axis=0)
    label_list = np.stack(label_list, axis=0)
    fashion_scatter(fashion_tsne, label_list)
    
    exit()

def modelLoad(model_path):
    graph_path = os.path.join(model_path, 'trained.pth')
    graph = VAE(in_dim=param.img_size ** 2, hidden_dim=300, out_dim=2).to(DEVICE)

    graph_ckpt = torch.load(graph_path)
    graph.load_state_dict(graph_ckpt['model_state_dict'])

    return graph

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([          
            transforms.ToTensor(),                     
            transforms.Normalize(mean=(0,), std=(1,))
    ])

    model_path = os.path.abspath('./saved/trained')

    test_dataset = dsets.MNIST(root='../MNISTdata/', train=False, download=True, transform=transform) #전체 training dataset
    train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True) # dataset 중에서 샘플링하는 data loader입니다

    test(train_loader, model_path)