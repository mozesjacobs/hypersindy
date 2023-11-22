import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from hypersindy.utils import *


def build_equation(lib, coef, eq, round_eq):
    """Builds an equations.

    Builds an equation using the given coefficient and library terms and then
    appends the result to the given equation.

    Args:
        lib: A list of strings of each term in the SINDy library. Should 
            be what is returned from "equation_sindy_library" in
            model_utils.py.
        coef: The coefficients (numpy array of shape (library_dim,)) of
            each term in the library 
        eq: A string of the start of the equation to be created. For example,
            if eq = "dx = ", then appends the result to the right side of that
            string.
        round_eq: If True, rounds the coefficients to 2 significant figures.

    Returns:
        A string of the created equation.
    """
    for i in range(len(coef)):
        if coef[i] != 0:
            curr_coef = coef[i]
            if round_eq:
                rounded_coef = np.round(curr_coef, 2) 
                if rounded_coef == 0:
                    rounded_coef = np.format_float_scientific(curr_coef, 1)
                rounded_coef = str(rounded_coef)
            else:
                rounded_coef = str(curr_coef)
            if i == len(coef) - 1:
                eq += rounded_coef + lib[i]
            else:
                eq += rounded_coef + lib[i] + ' + '
    if eq[-2] == '+':
        eq = eq[:-3]
    return eq


def get_equations(net, library, device, round_eq=True, seed=None):
    """Gets the equations learned by the network.

    Gets a list of the mean and STD equations learned by the network.

    Args:
        net: The network (torch.nn.Module) to get the equations for.
        library: The SINDy library object (from src.utils.library_utils).
        device: The cpu or gpu device to get the equations with. To use cpu,
            device must be "cpu". To use, specify which gpu as an integer
            (i.e.: 0 or 1 or 2 or 3).
        round_eq: If True, rounds the coefficients to 2 significant figures.
            Default: True.
        seed: The seed to use for reproducible randomization through
            set_random_seed from other.py. The default is None.

    Returns:
        Returns the equations as a list of strings in the format:
            ["MEAN",
                equation_1,
                equation_2,
                ...,
                equation_n,
                "STD",
                equation_1,
                equation_2,
                ...,
                equation_n]
        where n = x_dim. 
    """
    if seed is not None:
        set_random_seed(seed)
    starts = ["dx = ", "dy = ", "dz = "]
    if library.x_dim > 3:
        starts = ['dx' + str(i + 1) + " = " for i in range(library.x_dim)]
    equations = {"mean" : [],
                 "std" : []}
    coeffs = net.get_masked_coefficients(device=device).detach().cpu().numpy()
    mean_coefs, std_coefs = coeffs.mean(0), coeffs.std(0)
    feature_names = library.get_feature_names()
    for i in range(library.x_dim):
        equations["mean"].append(build_equation(feature_names, mean_coefs[:,i], starts[i], round_eq))
        equations["std"].append(build_equation(feature_names, std_coefs[:,i], starts[i], round_eq))
    return equations