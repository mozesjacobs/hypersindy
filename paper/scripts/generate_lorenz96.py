import sys
sys.path.append("../")
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import os

from other import set_random_seed, make_folder

"""
Generates Lorenz-96 data.
"""

# The simulation code here was taken from:
# https://en.wikipedia.org/wiki/Lorenz_96_model

def L96(x, F, N, scale):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    params = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        F_curr = np.random.normal(F, scale)
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F_curr
        params[i] = F_curr
    return d, params

def simulation(steps, dt, N, F, scale, x0=None):
    if x0 is None:
        x0 = F * np.ones(N)  # Initial state (equilibrium)
        x0[0] += 0.01  # Add small perturbation to the first variable
    timesteps = 10000
    dt = 0.01
    x = np.zeros([steps, N])
    dx = np.zeros([steps, N])
    ts = np.zeros([steps, N])
    params = np.zeros([steps, N])

    x[0] = x0
    for i in range(steps):
        x_dot, curr_params = L96(x[i], F, N, scale)
        dx[i] = x_dot
        ts[i] = i * dt
        params[i] = curr_params
        if i == steps - 1:
            break
        x[i + 1] = x[i] + x_dot * dt
    return x,  dx, ts, params

def pipeline(folder, steps=10000, dt=1e-2, N=10, F=8, scale=10.0, train=True, end='', x0=None):
    x_train, x_dot_train_measured, ts, params = simulation(steps, dt, N, F, scale, x0=x0)

    make_folder(folder)
    if folder[-1] != "/":
        folder += "/"
    if train:
        np.save(folder + "x_train" + end, x_train)
        np.save(folder + "x_dot" + end, x_dot_train_measured)
        np.save(folder + "x_ts" + end, ts)
        np.save(folder + "x_params" + end, params)
    else:
        np.save(folder + "x_test" + end, x_train)
        np.save(folder + "x_dot_test" + end, x_dot_train_measured)
        np.save(folder + "x_ts_test" + end, ts)
        np.save(folder + "x_params_test" + end, params)

def main():
    N = 10  # Number of variables
    F = 8  # Forcing

    set_random_seed(1000)
    pipeline("../data/lorenz96/scale-0.0/", scale=0.0)
    set_random_seed(1000)
    pipeline("../data/lorenz96/scale-5.0/", scale=5.0)
    set_random_seed(1000)
    pipeline("../data/lorenz96/scale-10.0/", scale=10.0)
    set_random_seed(1000)
    pipeline("../data/lorenz96/scale-20.0/", scale=20.0)

    # generate test trajectories
    x0_test = F * np.ones(N) + np.random.normal(0, 1, N)
    #print(x0_test)
    # x0_test:
    # [7.79695817, 8.68167566, 8.39110433, 5.9644262, 9.86240947, 9.54466122, 7.47865947, 6.87000308, 6.87716224, 8.73656858]
    set_random_seed(0)
    pipeline("../data/lorenz96/scale-10.0/", scale=10.0, train=False, end="_1", x0=x0_test)
    set_random_seed(0)
    pipeline("../data/lorenz96/scale-20.0/", scale=20.0, train=False, end="_1", x0=x0_test)

if __name__ == "__main__":
    main()