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
from other import init_weights, make_folder, set_random_seed

from exp_utils import get_equations, log_equations

import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

"""
Generates plots for figure 2b (rossler).
"""

def load_test_data(data_folder):
    x1 = np.array([np.load(data_folder + "rossler/scale-1.0/x_test_0.npy")])
    x2 = np.array([np.load(data_folder + "rossler/scale-5.0/x_test_0.npy")])
    x3 = np.array([np.load(data_folder + "rossler/scale-10.0/x_test_0.npy")])
    return [x1, x2, x3]

def load_model(device, x_dim, poly_order, include_constant,
               noise_dim, num_hidden, hidden_dim, stat_size, batch_size,
               cp_path):

    torch.cuda.set_device(device=device)
    device = torch.cuda.current_device()

    library = Library(n=x_dim, poly_order=poly_order, include_constant=include_constant)

    net = Net(library, noise_dim=noise_dim, hidden_dim=hidden_dim,
              statistic_batch_size=stat_size, num_hidden=num_hidden).to(device)
    net.get_masked_coefficients(batch_size=batch_size, device=device)

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

def mean_trajectory(net, library, device, x0, dt=1e-2, ts=10000, seed=0):
    set_random_seed(seed)
    coefs = net.get_masked_coefficients(device=device).mean(0)
    zc = torch.from_numpy(x0).type(torch.FloatTensor).to(device).unsqueeze(0)
    zs = []
    for i in range(ts):
        lib = net.library.transform(zc)
        zc = zc + torch.matmul(lib, coefs) * dt
        zs.append(zc)
    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1)
    return zs.detach().cpu().numpy().squeeze(0)

def plot_samples(x, samples, means, dpi=300, figsize=None, filename=None):
    '''Plot test, mean, and sample trajectories.
        Plotting code partially adopted from:
        https://www.tutorialspoint.com/how-to-hide-axes-but-keep-axis-labels-in-3d-plot-with-matplotlib
    '''

    sns.set()
    for i in range(len(samples)):
        if figsize is not None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        else:
            fig = plt.figure(dpi=dpi)
        
        axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
        axes[0].plot(x[i][0][:, 0], x[i][0][:, 1], x[i][0][:,2], color='red')
        axes[1].plot(means[i][:, 0], means[i][:, 1], means[i][:,2], color='purple')
        sample_idx = 0
        curr_sample = samples[i][sample_idx]
        while np.any(np.isnan(curr_sample)):
            sample_idx += 1
            curr_sample = samples[i][sample_idx]
        axes[2].plot(curr_sample[:, 0], curr_sample[:, 1], curr_sample[:,2], color='blue')
        
        color_tuple = (1.0, 1.0, 1.0, 0.0)
        for ax in axes:
            ax.grid(False)
            ax.xaxis.set_pane_color(color_tuple)
            ax.yaxis.set_pane_color(color_tuple)
            ax.zaxis.set_pane_color(color_tuple)
            ax.xaxis.line.set_color(color_tuple)
            ax.yaxis.line.set_color(color_tuple)
            ax.zaxis.line.set_color(color_tuple)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
        fig.subplots_adjust(hspace=-0.1, wspace=-0.15)
            
        if filename is not None:
            plt.savefig(filename + str(i) + ".png", dpi=300)
        plt.show()
        plt.close()

def main():
    # parameters
    seed = 5281998
    device = 3
    data_folder = "../data/"
    model = "HyperSINDy"
    dt = 0.01
    hidden_dim = 64
    stat_size = 250
    num_hidden = 5
    noise_dim = 6
    x_dim = 3
    adam_reg = 1e-2
    gamma_factor = 0.999
    poly_order = 3
    include_constant = True
    noise_dim = 6
    batch_size = 500
    net1_path = "../runs/rossler/cp_1.pt"
    net2_path = "../runs/rossler/cp_2.pt"
    net3_path = "../runs/rossler/cp_3.pt"

    # load models
    net1, library, device = load_model(device, x_dim, poly_order,
                                       include_constant, noise_dim, num_hidden,
                                       hidden_dim, stat_size, batch_size, net1_path)
    net2, library, device = load_model(device, x_dim, poly_order,
                                       include_constant, noise_dim, num_hidden,
                                       hidden_dim, stat_size, batch_size, net2_path)
    net3, library, device = load_model(device, x_dim, poly_order,
                                       include_constant, noise_dim, num_hidden,
                                       hidden_dim, stat_size, batch_size, net3_path)
    nets = [net1, net2, net3]

    # load test data and initial condition
    x_test = load_test_data(data_folder)
    x0_test = x_test[0][0][0]

    # sample trajectories
    samples = [sample_trajectory(nets[i], library, device, x0_test, 10, seed=seed) for i in range(len(nets))]

    # trajectory using mean of sampled equations
    means = [mean_trajectory(nets[i], library, device, x0_test, seed=seed) for i in range(len(nets))]

    # plot
    plot_samples(x_test, samples, means, 300, (20, 20), "../results/fig2b_")

    # output sampled equations
    with open("../results/fig2b.txt", "w") as f:
        for net in nets:
            curr_eqs = get_equations(net, library, "HyperSINDy", device, True, seed=seed)
            for eq in curr_eqs:
                print(eq, file=f)
            print(file=f)

if __name__ == "__main__":
    main()