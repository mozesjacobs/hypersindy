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

"""
Generates HyperSINDy Lorenz-96 samples from appendix.
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
    """Sample HyperSINDy trajectories.
    """
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

def plot(spatial, t, samples, num_samples, dpi, figsize, fname, cmap):
    """Plot num_samples sample heatmaps.
    """
    sns.set()
    fig, axes = plt.subplots(num_samples, 1, figsize=figsize, dpi=dpi)
    sample_idx = 0
    for i in range(num_samples):
        curr_sample = samples[sample_idx]
        while np.any(np.isnan(curr_sample)):
            sample_idx += 1
            curr_sample = samples[sample_idx]
        mesh = axes[i].pcolormesh(t, spatial, curr_sample, cmap=cmap)
        sample_idx += 1
        if i != num_samples - 1:
            axes[i].set_xticks([])
    if fname is not None:
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
    runs = "../runs/lorenz96"
    timesteps = 2500

    # load hypersindy model
    net, library, device = load_model(device, x_dim, poly_order, include_constant,
                                    noise_dim, hidden_dim, stat_size, stat_size,
                                    num_hidden, batch_norm, runs + "/cp_1.pt")

    # load test data and get test initial condition
    x_test = np.load("../data/lorenz96/scale-10.0/x_test_1.npy")
    x0_test = x_test[0]

    # sample trajectories
    samples = sample_trajectory(net, library, device, x0_test, batch_size=25, dt=dt, ts=timesteps, seed=SEED)

    # necessary for heatmap
    t = np.array([np.linspace(0, timesteps * dt, timesteps) for _ in range(x_dim)]).T
    spatial = np.array([np.arange(1, x_dim + 1, 1) for _ in range(timesteps)])
    
    # plot samples
    plot(spatial, t, samples, 5, 300, (20, 10), "../results/app_lorenz96.png", "Blues")

if __name__ == "__main__":
    main()