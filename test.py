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
from utils import sampleOut, fashion_scatter, plot_reproduce, latent_plot
from params import param

def test(data_loader, model_path):
    graph = modelLoad(model_path)
    graph.eval()
    with torch.no_grad():
        fashion_tsne = []
        label_list = []
        for i, batch in tqdm(enumerate(data_loader)):
            images, labels = batch
            images = images.to(DEVICE).view(-1, param.img_size ** 2)
            out_img, mu, sigma = graph(images)
            
            z = sampleOut(np.array(mu.data.cpu()), np.array(sigma.data.cpu()))
            assert len(z) == len(labels)
            fashion_tsne.extend(z)
            # fashion_tsne.extend(TSNE(random_state=123).fit_transform(z)) # when z size is not 2
            label_list.extend(np.array(labels.data.cpu()))
            plot_reproduce(images, out_img, 20, 'test_{}'.format(i))
            if i > 10:
                break
        fashion_tsne = np.stack(fashion_tsne, axis=0)
        label_list = np.stack(label_list, axis=0)
        fashion_scatter(fashion_tsne, label_list)
    

def latent_test(data_loader, model_path):
    graph = modelLoad(model_path)
    graph.eval()

    row_mesh = np.linspace(-2,2, 20)
    col_mesh = np.linspace(-2,2, 20)
    row, col = np.meshgrid(row_mesh, col_mesh)

    with torch.no_grad():
        out_img_list = []
        for i, batch in tqdm(enumerate(data_loader)):
            out_img_list = []
            images, _ = batch
            images = images.to(DEVICE).view(-1, param.img_size ** 2)
            _, mu, sigma = graph(images)
            
            # z = sampleOut(np.array(mu.data.cpu()), np.array(sigma.data.cpu()))
            # z = z[0]
            # row_z = row + z[0]
            # col_z = col + z[1]
            row_z = row
            col_z = col

            for r, c in zip(row_z, col_z):
                r_and_c = torch.from_numpy(np.stack([r,c], axis=1)).float().to(DEVICE)
                out_img = graph.infer_z(r_and_c)
                out_img_list.append(out_img)

            out_img_list = torch.cat(out_img_list, dim=0)
            latent_plot(out_img_list, 20, i)
        
        exit()

def modelLoad(model_path):
    graph_path = os.path.join(model_path, 'trained_val.pth')
    graph = VAE(in_dim=param.img_size ** 2, hidden_dim=500, out_dim=param.out_dim).to(DEVICE)

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

    model_path = os.path.abspath('../saved/trained')

    test_dataset = dsets.MNIST(root='../MNISTdata/', train=False, download=True, transform=transform) #전체 training dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=400, shuffle=True) # dataset 중에서 샘플링하는 data loader입니다

    # test(test_loader, model_path)
    latent_test(test_loader, model_path)