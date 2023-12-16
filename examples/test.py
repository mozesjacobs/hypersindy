import numpy as np
import torch
import sys
from hypersindy import HyperSINDy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.from_numpy(np.load("paper/data/lorenz/scale-1.0/x_train.npy")).to(device)

#"""
net = HyperSINDy()
net = net.fit(x, dt=1e-2, device=device,
    beta=10, beta_warmup_epoch=100, beta_spike=None, beta_spike_epoch=None,
    lmda_init=1e-2, lmda_spike=None, lmda_spike_epoch=None,
    checkpoint_interval=50, eval_interval=50,
    learning_rate=5e-3, hard_threshold=0.05, threshold_interval=100,
    epochs=499, batch_size=250, run_path="runs/1")

net.save("runs/cp1.pt")
#"""

net = HyperSINDy().load("runs/cp1.pt", device)
#net.print()
print(net.coefs(5))