import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from hypersindy.net import Net
from hypersindy.trainer import Trainer
from hypersindy.library import Library
from hypersindy.utils import init_weights, make_folder, set_random_seed
from hypersindy.equations import get_equations
from scripts.generate_lorenz import simulation


"""
Generates test Lorenz samples from appendix.
"""

def plot_samples(samples, num_samples=4, dpi=300, figsize=None, filename=None):
    '''Plot samples.
        Plotting code partially adopted from:
        https://www.tutorialspoint.com/how-to-hide-axes-but-keep-axis-labels-in-3d-plot-with-matplotlib
    '''

    sns.set()
    if figsize is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = plt.figure(dpi=dpi)
    count = 0
    for i in range(len(samples)):
        sample_idx = 0
        for j in range(num_samples):
            ax = fig.add_subplot(3, num_samples, count + 1, projection='3d')
            curr_sample = samples[i][sample_idx]
            while np.any(np.isnan(curr_sample)):
                sample_idx += 1
                curr_sample = samples[i][sample_idx]
            ax.plot(curr_sample[:, 0], curr_sample[:, 1], curr_sample[:,2], color='red')

            ax.grid(False)
            color_tuple = (1.0, 1.0, 1.0, 0.0)
            ax.xaxis.set_pane_color(color_tuple)
            ax.yaxis.set_pane_color(color_tuple)
            ax.zaxis.set_pane_color(color_tuple)
            ax.xaxis.line.set_color(color_tuple)
            ax.yaxis.line.set_color(color_tuple)
            ax.zaxis.line.set_color(color_tuple)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            sample_idx += 1
            count += 1
            
    fig.subplots_adjust(hspace=-0.7, wspace=-0.125)    

    if filename is not None:
        plt.savefig(filename + ".png", dpi=300)
    plt.show()
    plt.close()

def get_samples(init_cond, steps, dt, s, r, b, scale, num_samples, x_dim):
    '''Sample test trajectories.
    '''
    samples = np.zeros([num_samples, steps, x_dim])
    for i in range(num_samples):
        samples[i] = simulation(init_cond, steps, dt, s, r, b, scale)[0]
    return samples

def main():
    # settings
    SEED = 123456
    set_random_seed(SEED)
    dt = 0.01
    x_dim = 3
    timesteps = 10000
    x0 = (-1, 2, 0.5)
    scales = [1, 5, 10]

    # sample gt trajectories
    samples = [get_samples(x0, timesteps, dt, 10, 28, 8.0 / 3, scale, 20, x_dim) for scale in scales]

    # plot
    plot_samples(samples, 6, 300, (20, 20), "../results/app_lorenz_true")

if __name__ == "__main__":
    main()