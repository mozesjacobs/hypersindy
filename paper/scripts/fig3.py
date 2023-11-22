import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import pysindy as ps
import torch
import torch.nn as nn

import sys
sys.path.append("../")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


from HyperSINDy import Net
from baseline import Trainer
from library_utils import Library
from other import init_weights, make_folder, set_random_seed
from exp_utils import get_equations, log_equations

"""
Generates plots for figure 3 (lotka-volterra).
"""

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

def fourth_order_diff(x, dt):
    """Gets the derivatives of the data.

    Gets the derivative of x with respect to time using fourth order
    differentiation.
    The code for this function was taken from:
    https://github.com/urban-fasel/EnsembleSINDy

    Args:
        x: The data (np.ndarray of shape (timesteps x x_dim)) to
            differentiate.
        dt: The amount of time between two adjacent data points (i.e.,
            the time between x[0] and x[1], or x[1] and x[2]).

    Returns:
        A np.ndarray of the derivatives of x, with shape (timesteps x x_dim)
    """
    dx = np.zeros([x.shape[0], x.shape[1]])
    dx[0] = (-11.0 / 6) * x[0] + 3 * x[1] - 1.5 * x[2] + x[3] / 3
    dx[1] = (-11.0 / 6) * x[1] + 3 * x[2] - 1.5 * x[3] + x[4] / 3
    dx[2:-2] = (-1.0 / 12) * x[4:] + (2.0 / 3) * x[3:-1] - (2.0 / 3) * x[1:-3] + (1.0 / 12) * x[:-4]
    dx[-2] = (11.0 / 6) * x[-2] - 3.0 * x[-3] + 1.5 * x[-4] - x[-5] / 3.0
    dx[-1] = (11.0 / 6) * x[-1] - 3.0 * x[-2] + 1.5 * x[-3] - x[-4] / 3.0
    return dx / dt 

def fit_esindy(x_train, x_dot, dt, degree, include_bias, threshold, thresholder, max_iter, n_models, seed):
    set_random_seed(seed)
    feature_names = ['x', 'y', 'z']
    ps_library = ps.PolynomialLibrary(degree=degree, include_bias=include_bias)
    optimizer = ps.SR3(
        threshold=threshold, thresholder=thresholder, max_iter=max_iter, tol=1e-1
    )
    model = ps.SINDy(feature_names=feature_names, feature_library=ps_library, optimizer=optimizer)
    model.fit(x_train, x_dot=x_dot, t=dt, ensemble=True, quiet=True, n_models=n_models)
    return model

def main():
    # settings
    SEED = 5281998
    set_random_seed(SEED)
    device = 2
    noise_dim = 4
    batch_size = 250
    data_folder = "../data/"
    model = "HyperSINDy"
    dt = 0.005
    hidden_dim = 64
    stat_size = 250
    num_hidden = 5
    x_dim = 2
    adam_reg = 1e-2
    gamma_factor = 0.999
    poly_order = 3
    include_constant = True
    runs = "runs/"
    t = None

    # load hypersindy model
    cp_path = "../runs/lotkavolterra/cp_1.pt"
    params = np.load(data_folder + "lotkavolterra/scale-5.0/x_params.npy")
    net, library, device = load_model(device, x_dim, poly_order, include_constant,
                                    noise_dim, num_hidden, hidden_dim, stat_size, batch_size,
                                    cp_path)
    
    # load train data and use same derivative calculation as hypersindy used
    x_train = np.load('../data/lotkavolterra/scale-5.0/x_train.npy')
    x_dot = fourth_order_diff(x_train, dt)

    # get esindy coefficients
    esindy = fit_esindy(x_train, x_dot, dt, 2, True, 2.5, "l0", 2000, 1000, SEED)
    ensemble_coefs = np.array(esindy.coef_list)
    ensemble_coefs = np.transpose(ensemble_coefs, (0, 2, 1))
    e1 = ensemble_coefs[:,1,0]
    e2 = ensemble_coefs[:,4,0]
    e3 = ensemble_coefs[:,2,1]
    e4 = ensemble_coefs[:,4,1]
    es = [e1, e2, e3, e4]
    terms = ['x', 'xy', 'y', 'xy']

    # get hypersindy coefficients
    coefs = net.get_masked_coefficients(batch_size=1000, device=device)
    c1 = coefs[:,1,0]
    c2 = coefs[:,4,0]
    c3 = coefs[:,2,1]
    c4 = coefs[:,4,1]
    cs = [c1, c2, c3, c4]
    cs = [c.detach().cpu().numpy() for c in cs]
    terms = ['x', 'xy', 'y', 'xy']

    # plot histograms
    sns.set()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    for i in range(4):    
        true_coefs = params[:,i]
        if i == 1 or i == 2:
            true_coefs = params[:,i] * -1
        sns.histplot(true_coefs, ax=axes[i], stat='percent', color='red', bins=25)
        sns.histplot(cs[i], ax=axes[i], stat='percent', color='blue', bins=25)
        sns.histplot(es[i], ax=axes[i], stat='percent', color='yellow', bins=25)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    plt.savefig("../results/fig3.png")
    plt.show()

if __name__ == "__main__":
    main()