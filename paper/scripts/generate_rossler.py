import sys
sys.path.append("../")
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from other import set_random_seed, make_folder

"""
Generates Rossler data.
"""

# The simulation code here was adopted from:
# https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

def derivative(x, y, z, a, b, c, t, scale):
    # sample coefficients
    a = np.random.normal(a, min(1.0, scale))
    b = np.random.normal(b, min(1.0, scale))
    c = np.random.normal(c, scale)

    # derivative
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    
    return np.array([x_dot, y_dot, z_dot]), np.array([a, b, c])

def simulation(init_conds, steps, dt, s, r, b, scale):
    # Need one more for the initial values
    x = np.zeros([steps, len(init_conds)])
    dx = np.zeros([steps, len(init_conds)])
    ts = np.zeros([steps,])
    params = np.zeros([steps, 3])

    # Set initial values
    x[0] = init_conds

    # Step through "time", calculating the partial derivatives at the current
    # point and using them to estimate the next point
    for i in range(steps):
        x_dot, curr_params = derivative(x[i][0], x[i][1], x[i][2], s, r, b, i * dt, scale)
        dx[i] = x_dot
        ts[i] = i * dt
        params[i] = curr_params
        if i == steps - 1:
            break
        x[i + 1] = x[i] + x_dot * dt
    return x, dx, ts, params

def pipeline(folder, init_conds=(0., 1., 1.05), steps=10000, dt=1e-2, s=0.2, r=0.2, b=5.7, scale=10.0, train=True, end=''):
    x_train, x_dot_train_measured, ts, params = simulation(init_conds, steps, dt, s, r, b, scale)

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
    set_random_seed(1000)
    pipeline("../data/rossler/scale-0.0", scale=0.0)
    set_random_seed(1000)
    pipeline("../data/rossler/scale-1.0", scale=1.0)
    set_random_seed(1000)
    pipeline("../data/rossler/scale-2.5", scale=2.5)
    set_random_seed(1000)
    pipeline("../data/rossler/scale-5.0", scale=5.0)
    set_random_seed(1000)
    pipeline("../data/rossler/scale-10.0", scale=10.0)

    # generate test trajectories
    x0 = (-1, 2, 0.5)
    test_seeds = [0, 1]
    for i in range(2):
        set_random_seed(test_seeds[i])
        pipeline("../data/rossler/scale-1.0", x0, scale=1.0, train=False, end="_" + str(i))
        set_random_seed(test_seeds[i])
        pipeline("../data/rossler/scale-2.5", x0, scale=2.5, train=False, end="_" + str(i))
        set_random_seed(test_seeds[i])
        pipeline("../data/rossler/scale-5.0", x0, scale=5.0, train=False, end="_" + str(i))
        set_random_seed(test_seeds[i])
        pipeline("../data/rossler/scale-10.0", x0, scale=10.0, train=False, end="_" + str(i))
    


if __name__ == "__main__":
    main()