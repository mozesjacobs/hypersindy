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
from Datasets import SyntheticDataset
from other import init_weights, set_random_seed

"""
Train HyperSINDy on lorenz-96
"""

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Template")
    parser.add_argument('-S1', '--s1', default=0, type=int, help="Exp set 1.")
    parser.add_argument('-D1', '--d1', default=1, type=int, help="Device for exp 1.")
    return parser.parse_args()

def pipeline(library, trainset, epochs, lr,
             lmda_init, lmda_max, lmda_max_epoch, lmda_spike, lmda_spike_epoch,
             beta_init, beta_max, beta_max_epoch, beta_spike, beta_spike_epoch,
             adam_reg, gamma_factor, batch_size,
             thresh_interval, eval_interval, hard_thresh,
             run_name, runs, noise_dim, hidden_dim, stat_size, device,
             num_hidden, batch_norm):
    print(run_name)

    torch.cuda.set_device(device=device)
    device = torch.cuda.current_device()
    net = Net(library, noise_dim=noise_dim, hidden_dim=hidden_dim,
              statistic_batch_size=stat_size, num_hidden=num_hidden,
              batch_norm=batch_norm).to(device)
    net.apply(init_weights)

    trainer = Trainer(net, library, runs + run_name, runs + "cp_" + run_name + ".pt",
                      beta_init=beta_init, beta_max=beta_max, beta_max_epoch=beta_max_epoch, 
                      beta_spike=beta_spike, beta_spike_epoch=beta_spike_epoch,
                      lmda_init=lmda_init, lmda_max=lmda_max, lmda_max_epoch=lmda_max_epoch,
                      lmda_spike=lmda_spike, lmda_spike_epoch=lmda_spike_epoch,
                      learning_rate=lr, adam_reg=adam_reg, gamma_factor=gamma_factor,
                      epochs=epochs, batch_size=batch_size, device=device,
                      hard_threshold=hard_thresh, threshold_interval=thresh_interval,
                      eval_interval=eval_interval)
    trainer.train(trainset)

def load_data(library, data_folder, dataset, t, dt, model):
    x = np.load(data_folder + dataset + "/x_train.npy")
    if t:
        t = np.load(data_folder + dataset + "/x_ts.npy")
    else:
        t = None
    return SyntheticDataset(x=x, t=t, library=library, dataset=dataset, dt=dt, model=model)

def main():
    # Parse exp args
    args = parse_args()

    # Globals
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
    runs = "../runs/lorenz96/"
    library = Library(n=x_dim, poly_order=poly_order, include_constant=include_constant)
    t = False
    seed = 0

    # Individual experiments
    if args.s1 == 1:
        set_random_seed(seed)
        run_name = "1"
        device = args.d1
        trainset = load_data(library, data_folder, "lorenz96/scale-10.0", t, dt, model)
        pipeline(library=library, trainset=trainset, epochs=999, lr=5e-3, batch_size=250,
            lmda_init=1e-2, lmda_max=1e-2, lmda_max_epoch=1, lmda_spike=1e1, lmda_spike_epoch=400,
            beta_init=0.01, beta_max=10.0, beta_max_epoch=100, beta_spike=None, beta_spike_epoch=None,
            noise_dim=20, thresh_interval=100, hard_thresh=0.05, batch_norm=False, eval_interval=50,
            adam_reg=adam_reg, gamma_factor=gamma_factor, runs=runs, run_name=run_name, device=device,
            hidden_dim=hidden_dim, stat_size=stat_size, num_hidden=num_hidden)
        
        
if __name__ == "__main__":
    main()