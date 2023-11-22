import sys
sys.path.append("../")
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import halfnorm

from other import set_random_seed, make_folder

"""
Generates Lotka-Volterra data.
"""

# The simulation code here was adopted from:
# https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

def derivative(x, y, a, b, c, d, t, scale):
    # sample coefficients
    a = halfnorm.rvs(a, scale)
    b = halfnorm.rvs(b, scale)
    c = halfnorm.rvs(c, scale)
    d = halfnorm.rvs(d, scale)

    # derivative
    x_dot = a * x - b * x * y
    y_dot = c * x * y - d * y
    
    return np.array([x_dot, y_dot]), np.array([a, b, c, d])

def simulation(init_conds, steps, dt, a, b, c, d, scale):
    # Need one more for the initial values
    x = np.zeros([steps, len(init_conds)])
    dx = np.zeros([steps, len(init_conds)])
    ts = np.zeros([steps,])
    params = np.zeros([steps, 4])

    # Set initial values
    x[0] = init_conds

    # Step through "time", calculating the partial derivatives at the current
    # point and using them to estimate the next point
    for i in range(steps):
        x_dot, curr_params = derivative(x[i][0], x[i][1], a, b, c, d, i * dt, scale)
        dx[i] = x_dot
        ts[i] = i * dt
        params[i] = curr_params
        if i == steps - 1:
            break
        x[i + 1] = x[i] + x_dot * dt
    return x, dx, ts, params

def pipeline(folder, init_conds=(4, 2), steps=10000, dt=5e-3, a=1.0, b=1.0, c=1.0, d=1.0, scale=10.0):
    x_train, x_dot_train_measured, ts, params = simulation(init_conds, steps, dt, a, b, c, d, scale)

    make_folder(folder)
    if folder[-1] != "/":
        folder += "/"
    np.save(folder + "x_train", x_train)
    np.save(folder + "x_dot", x_dot_train_measured)
    np.save(folder + "x_ts", ts)
    np.save(folder + "x_params", params)

def main():
    set_random_seed(1000)
    pipeline("../data/lotkavolterra/scale-0.0", scale=0.0)
    set_random_seed(1000)
    pipeline("../data/lotkavolterra/scale-1.0", scale=1.0)
    set_random_seed(1000)
    pipeline("../data/lotkavolterra/scale-2.5", scale=2.5)
    set_random_seed(1000)
    pipeline("../data/lotkavolterra/scale-5.0", scale=5.0)
    


if __name__ == "__main__":
    main()