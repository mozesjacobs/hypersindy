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
Generates equations for figure 4b (lorenz-96).
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
    
    return net, library, device


def reformat(eqs, eq_starts, filename=None):
    eq_ct = 0
    for eq in eqs:
        if eq == "MEAN":
            continue
        if eq == "STD":
            eq_ct = 0
            continue
            
        curr_eq_start = eq_starts[eq_ct]
        
        eq = eq.split(" ")
        
        result = ""
        for i in range(len(eq)):
            curr_term = eq[i]
            if curr_term[0:2] == "dx":
                result += "\dot{x}_{" + curr_term[2:] + "}"
            elif curr_term == "=":
                result += " = "
            elif "x" not in curr_term and curr_term != "+":
                result += curr_term + " "
            elif curr_term == "+":
                if eq[i + 1][0] == "-":
                    result += "- "
                    next_term = eq[i + 1][1:]
                else:
                    result += "+ "
                    next_term = eq[i + 1]
                next_term = next_term.split("x")
                coef = next_term[0]
                result += coef
                for j in range(1, len(next_term)):
                    result += "x_{" + next_term[j] + "}"
                result += " "
                    
        #print(result)
        if filename is not None:
            print(result, file=filename)
    #print()
    if filename is not None:
        print(file=filename)  
        eq_ct += 1


def main():
    # settings
    seed = 5281998
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

    # load hypersindy model
    net1, library, device = load_model(device, x_dim, poly_order, include_constant,
                                    noise_dim, hidden_dim, stat_size, stat_size,
                                    num_hidden, batch_norm, runs + "/cp_1.pt")

    # get equations
    eq1 = get_equations(net1, library, model, device, seed=seed)

    # start of each equations and term names
    eq_starts = ["dx" + str(i + 1) for i in range(x_dim)]
    terms = np.array(['x' + str(i + 1) for i in range(x_dim)])

    # output reformatted equations (reformats to latex-friendly version  for paper figure)
    with open("../results/lorenz96.txt", "w") as f:
        reformat(eq1, eq_starts, f)        

if __name__ == "__main__":
    main()