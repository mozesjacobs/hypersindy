import math
import torch
import torch.nn as nn
from torch.autograd import Variable

# This code was taken from:
# https://github.com/moskomule/l0.pytorch/blob/master/l0module.py


class L0Norm(nn.Module):

    """A HyperSINDy model.

    The HyperSINDy model that uses a variational encoder and a hypernetwork
    to learn governing equations.

    Attributes:

        self.library_dim = The number (int) of terms in the SINDy library.
        self.x_dim = The spatial dimension (int) of the data.
        self.loc = The location of q(s) (the log alpha value described in the HyperSINDy manuscript).
        self.temp = The scale of q(s) (the beta value described in the HyperSINDy manuscript).
        self.gamma = The lower bound of "stretched" s (gamma < 0).
        self.zeta = The upper bound of "stretched" s (zeta > 1).
        self.gamma_zeta_ratio = log(-gamma / zeta).
        self.sig = A nn.Sigmoid() object.
        self.hard_threshold_mask: A torch.Tensor of shape (library_dim x x_dim) 
            used to permanently zero out SINDy coefficients.
    """

    def __init__(self, library_dim, x_dim, 
                 loc_mean=0, loc_sdev=0.01,
                 beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """Initalizes the _L0Norm.

        Initializes the _L0Norm object.

        Args:
            loc_mean: The mean of the normal distribution which generates
                initial location parameters. Default: 0.
            loc_sdev: The standard deviation of the normal distribution which
                generates initial location parameters. Default: 0.01.
            beta: The initial temperature parameter. Default: 2  / 3
            gamma: The lower bound of "stretched" s (gamma < 0). Default: -0.1.
            zeta: The upper bound of "stretched" s (zeta > 1). Default: 1.1.
            fix_temp: If True, fixes beta. If False, makes beta a trainable
                parameter. Default: True.

        Returns:
            A _L0Norm().
        """

        super(L0Norm, self).__init__()
        self.library_dim = library_dim
        self.x_dim = x_dim
        self.loc = nn.Parameter(torch.zeros([self.library_dim, self.x_dim]).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.sig = nn.Sigmoid()
        self.hard_threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.x_dim),
            requires_grad=False)


    def get_mask(self, batch_size, device, eval_mask=False):
        """Returns the l0 mask.

        Scenario 1:
        In training (or if eval_mask is False), samples batch_size masks
        with the reparameterization trick and returns the l0 penalty.
        Scenario 2:
        If testing (or if eval_mask is True), returns the mask without
        sampling and 0 for the penalty.

        Args:
            n: The torch.Tensor of shape (-1, noise_dim) to feed into the
                hypernetwork. If None, samples from a N(0, 1) distribution.
            batch_size: If n is None, samples a vector of shape 
                (batch_size, self.noise_dim) from a N(0, 1) distribution.
            device: If n is None, the sampled noise vector uses this 
                cpu or gpu device. To use cpu, device must be "cpu". To use
                gpu, specify which gpu to use as an integer (i.e.: 0 or 1 or 2
                or 3). 
            l0_mask: The torch.Tensor to use to mask the coefficients.
                Should be the result of calling _L0Norm._get_mask().
                If l0_mask is None (default), then makes a call _get_mask().

        Returns:
            A tuple of (mask, penalty) where:
            Scenario 1:
                mask is a torch.tensor of shape (batch_size, library_dim, x_dim)
                and penalty is a torch.tensor of shape(library_dim, x_dim)
            Scenario 2:
                mask is a torch.tensor of shape (library_dim, x_dim) and
                penalty = 0.
        """
        if eval_mask or not self.training:
            s = self.sig(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        else:
            u = torch.zeros([batch_size, self.library_dim, self.x_dim], device=device).uniform_()
            s = self.sig((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = self.sig(self.loc - self.temp * self.gamma_zeta_ratio)
            penalty = penalty * self.hard_threshold_mask
        return hard_sigmoid(s) * self.hard_threshold_mask, penalty

def hard_sigmoid(x):
    """
        Returns the hard sigmoid of x.
    """
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))