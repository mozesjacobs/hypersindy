import torch
from torch.utils.data import Dataset
import numpy as np

class DynamicDataset(Dataset):
    """A dataset to run experiments with.

    Creates a dataset of torch tensors. Tensors must be loaded from numpy
    array files or passed in as numpy arrays.


    Attributes:
        self.x: The state as a torch.Tensor of shape (timesteps x x_dim).
        self.x_dot: The time derivative of the state as a torch.tensor of
            shape (timesteps x x_dim). The derivative is calculated using
            fourth order finite differences.
        self.x_dot_standard: A standardized version of self.x_dot:
            self.x_dot_standard = (self.x_dot - self.x_dot.mean(0)) / self.x_dot.std(0)
        self.x_lib: The SINDy library form of self.x, as a torch.Tensor of
            of shape (timesteps x library_dim).
    """

    def __init__(self, x, x_lib, dt=0.01):
        """Initializes the DynamicDataset.

        Initializes the DynamicDataset using the given parameters.

        Args:
            x: A Numpy array of the data to use
            library: The SINDy library object (from library_utils)
                used to transform the data.
            dt: The time between adjacent states (e.g. between x[0] and x[1], x[1] and x[2]).
                The default is 0.01.
            
            fpath: The fll path to the data file of x. The default is None. I

        Returns:
            A DynamicDataset.
        """
        self.x = x
        self.x_lib = x_lib
        self.x_dot = self.fourth_order_diff(self.x, dt)
        self.x_dot_standard = (self.x_dot - self.x_dot.mean(0)) / self.x_dot.std(0)

    def __len__(self):
        """The length of the dataset.

        Gets the length of the dataset (in timesteps).

        Args:
            None

        Returns:
            The length of the dataset along dimension 0.
        """
        return len(self.x)
    
    def __getitem__(self, idx):
        """Gets the item.

        Gets the item at the current index.

        Args:
            idx: The integer index to access the data.

        Returns:
            If t was NOT given during construction of the dataset:
                A tuple of (tensor_a, tensor_b, tensor_c, tensor_d)
                where tensor_a is the state, tensor_b is the library,
                tensor_c is the derivative, and tensor_c is the standardized
                derivative.
            If t was given during construction:
                A tuple of (tensor_a, tensor_b, tensor_c, tensor_d, tensor_e)
                where tensors a, b, c, and d are the same as above, and tensor_e
                is the associated timepoints.
        """
        return self.x[idx], self.x_lib[idx], self.x_dot[idx], self.x_dot_standard[idx]

    def fourth_order_diff(self, x, dt):
        """Gets the derivatives of the data.

        Gets the derivative of x with respect to time using fourth order
        differentiation.
        The code for this function was taken from:
        https://github.com/urban-fasel/EnsembleSINDy

        Args:
            x: The data (torch.Tensor of shape (timesteps x x_dim)) to
                differentiate.
            dt: The amount of time between two adjacent data points (i.e.,
                the time between x[0] and x[1], or x[1] and x[2]).

        Returns:
            A torch.tensor of the derivatives of x.
        """
        dx = torch.zeros(x.size())
        dx[0] = (-11.0 / 6) * x[0] + 3 * x[1] - 1.5 * x[2] + x[3] / 3
        dx[1] = (-11.0 / 6) * x[1] + 3 * x[2] - 1.5 * x[3] + x[4] / 3
        dx[2:-2] = (-1.0 / 12) * x[4:] + (2.0 / 3) * x[3:-1] - (2.0 / 3) * x[1:-3] + (1.0 / 12) * x[:-4]
        dx[-2] = (11.0 / 6) * x[-2] - 3.0 * x[-3] + 1.5 * x[-4] - x[-5] / 3.0
        dx[-1] = (11.0 / 6) * x[-1] - 3.0 * x[-2] + 1.5 * x[-3] - x[-4] / 3.0
        return dx / dt