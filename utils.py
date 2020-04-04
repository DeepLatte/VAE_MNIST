import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os 

from params import param

def sampleOut(mu, sigma):
    z_list = []
    for m, s in zip(mu, sigma):
        z = m + s * np.random.normal(0, 1, len(m))
        z_list.append(z)
    
    return z_list


# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.savefig(os.path.abspath('../saved/res.png'))
    return f, ax, sc, txts


def plot_reproduce(input, output, numb_in_row, i):

    width = numb_in_row * param.img_size
    height = width
    paper_input = np.zeros((height, width))
    paper_output = np.zeros((height, width))
    start_h = 0
    pivot_h = start_h
    pivot_w = 0
    input = input.view(-1, param.img_size, param.img_size)
    output = output.view(-1, param.img_size, param.img_size)
    for in_img, out_img in zip(input, output):
        paper_input[pivot_h:pivot_h + param.img_size, pivot_w : pivot_w + param.img_size] = np.array(in_img.data.cpu())
        paper_output[pivot_h:pivot_h + param.img_size, pivot_w : pivot_w + param.img_size] = np.array(out_img.data.cpu())
        pivot_w += param.img_size
        if pivot_w >= width:
            pivot_w = 0
            pivot_h += param.img_size

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    bx = fig.add_subplot(1,2,2)
    
    ax.imshow(paper_input)
    bx.imshow(paper_output)
    plt.savefig(os.path.abspath('../saved/res_{}.png'.format(i)))
    plt.close()

def latent_plot(output, numb_in_row, i):
    width = numb_in_row * param.img_size
    height = width
    paper_output = np.zeros((height, width))
    start_h = 0
    pivot_h = start_h
    pivot_w = 0
    output = output.view(-1, param.img_size, param.img_size)
    for out_img in output:
        paper_output[pivot_h:pivot_h + param.img_size, pivot_w : pivot_w + param.img_size] = np.array(out_img.data.cpu())
        pivot_w += param.img_size
        if pivot_w >= width:
            pivot_w = 0
            pivot_h += param.img_size

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.imshow(paper_output)
    plt.savefig(os.path.abspath('../saved/res_latent_{}.png'.format(i)))
    plt.close()