import sys
sys.path.append("../")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import numpy as np

from HyperSINDy import Net
from baseline import Trainer
from library_utils import Library
from other import init_weights, set_random_seed

from exp_utils import get_equations, log_equations

import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

"""
Generates plots for figure 4a (lorenz-96).
"""

def load_model(device, x_dim, poly_order, include_constant,
               noise_dim, hidden_dim, stat_size, batch_size,
               num_hidden, batch_norm, cp_path):

    torch.cuda.set_device(device=device)
    device = torch.cuda.current_device()

    library = Library(n=x_dim, poly_order=poly_order, include_constant=include_constant)

    net = Net(library, noise_dim=noise_dim, hidden_dim=hidden_dim,
              statistic_batch_size=stat_size,
              num_hidden=num_hidden, batch_norm=batch_norm).to(device)

    cp = torch.load(cp_path, map_location="cuda:" + str(device)) 
    net.load_state_dict(cp['model'])
    net.to(device)
    net = net.eval()
    
    return net, library, device

def sample_trajectory(net, library, device, x0, batch_size=10, dt=1e-2, ts=10000, seed=0):
    set_random_seed(seed)
    zc = torch.from_numpy(x0).type(torch.FloatTensor).to(device).unsqueeze(0)
    zc = zc.expand(batch_size, -1)
    zs = []
    for i in range(ts):
        coefs = net.get_masked_coefficients(batch_size=batch_size, device=device)        
        lib = net.library.transform(zc).unsqueeze(1)
        zc = zc + torch.bmm(lib, coefs).squeeze(1) * dt
        zs.append(zc)
    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1)
    return zs.detach().cpu().numpy()

def plot_both(spatial, t, sample, test, figsize, fname, cmap):
    sns.set()
    vmin = min(np.min(sample), np.min(test))
    vmax = max(np.max(sample), np.max(test))
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=500)
    mesh1 = axes[0].pcolormesh(t, spatial, test, cmap=cmap, vmin=vmin, vmax=vmax)
    mesh2 = axes[1].pcolormesh(t, spatial, sample, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.subplots_adjust(right=0.8)
    colorbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(mesh1, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=20)
    axes[0].set_xticks([])
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(fname)
    plt.show()
    plt.close()

def plot_single(spatial, t, traj, figsize, fname, cmap, vmin, vmax):
    sns.set()
    fig = plt.figure(figsize=figsize, dpi=500)
    mesh = plt.pcolormesh(t, spatial, traj, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(fname)
    plt.show()
    plt.close()

def main():
    # settings
    SEED = 5281998
    data_folder = "../data/"
    model = "HyperSINDy"
    dt = 0.01
    hidden_dim = 128
    stat_size = 250
    num_hidden = 5
    x_dim = 10
    adam_reg = 1e-2
    gamma_factor = 0.999
    poly_order = 3
    include_constant = True
    device = 2
    batch_norm = False
    noise_dim = 20
    timesteps = 2500
    cp_path = "../runs/lorenz96/cp_1.pt"

    # load hypersindy model
    net, library, device = load_model(device, x_dim, poly_order, include_constant,
                                    noise_dim, hidden_dim, stat_size, stat_size,
                                    num_hidden, batch_norm, cp_path)

    # load test data
    x_test = np.load("../data/lorenz96/scale-10.0/x_test_1.npy")
    x0_test = x_test[0]

    # sample hypersindy trajectories
    samples = sample_trajectory(net, library, device, x0_test, batch_size=5, dt=dt, ts=timesteps, seed=SEED)

    # use 3rd sample (see notebooks/fig4a.ipynb)
    idx = 3
    sample = samples[idx]

    # necessary for plotting the heatmap
    t = np.array([np.linspace(0, timesteps * dt, timesteps) for _ in range(x_dim)]).T
    spatial = np.array([np.arange(1, x_dim + 1, 1) for _ in range(timesteps)])

    # plot test and hypersindy on same figure
    plot_both(spatial, t, sample, x_test[0:timesteps], (25, 5), "../results/fig4.png", "Blues")

    # plot test and hypersindy on separate figures but using same color scale
    vmin = min(np.min(sample), np.min(x_test))
    vmax = max(np.max(sample), np.max(x_test))
    plot_single(spatial, t, x_test[0:timesteps], (20, 3.75), "../results/fig4_test.png", "Blues", vmin, vmax)
    plot_single(spatial, t, sample, (20, 3.75), "../results/fig4_sample.png", "Blues", vmin, vmax)

if __name__ == "__main__":
    main()