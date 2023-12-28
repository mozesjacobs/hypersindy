import torch
import torch.nn as nn
import numpy as np

class HyperNet(nn.Module):
    """A HyperNetwork.

    A HyperNetwork used to generate the parameters of some other function.

    Attributes:
        self.in_dim: The (int) size of the vector input to the hypernetwork.
            This vector should have dimension (batch x self.in_dim), where
            batch is the batch dimension.
        self.out_shape: The tuple (int_1, int_2, ..., int_n) shape of the
            vector output by the hypernetwork, excluding the batch dimension.
            Including the batch_dimension, this vector will have shape
            (batch, int_1, int_2, ..., int_n).
        self.layers: A list of layers in the hypernetwork. Includes nn.Linear and
            activation functions.
    """
    def __init__(self, in_dim, out_shape, hidden_dims=[8, 16, 32], bias=True, activation=nn.ELU()):
        """Initializes the HyperNet.

        Initializes the HyperNet with the given features.

        Args:
            in_dim: The (int) size of the vector input to the hypernetwork.
                This vector should have dimension (batch x self.in_dim), where
                batch is the batch dimension. 
            out_shape: The tuple (int_1, int_2, ..., int_n) shape of the
                vector output by the hypernetwork, excluding the batch dimension.
                Including the batch_dimension, this vector will have shape
                (batch, int_1, int_2, ..., int_n).
            hidden_dims: A list of integers denoting the size of the hidden
                layers. The network will have structure:
                    nn.Linear(in_dim x hidden_dims[0]),
                    activation,
                    nn.Linear(hidden_dims[1], hidden_dims[2]),
                    activation,
                    ...
                    nn.Linear(hidden_dims[-2], hidden_dims[-1]).
                The default is [8, 16, 32].
            bias: Iff True (bool), uses bias in the nn.Linear in the network.
                The default is True.
            activation: The activation function to use in the network. The
                default is nn.ELU().

        """
        super(HyperNet, self).__init__()

        self.in_dim = in_dim
        self.out_shape = out_shape

        layers = []
        in_features = self.in_dim
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            layers.append(activation)
            in_features = out_features
        layers.append(nn.Linear(in_features, np.prod(self.out_shape), bias=bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, n):
        """Runs the forward pass.

        Runs the forward pass by feeding the given noise vector to the
        hypernet.

        Args:
            n: The noise vector to feed to the hypernet. Should be a
                torch.Tensor of shape (batch_size x self.in_dim).
        
        Returns:
            A torch.Tensor of shape (batch size x int_1 x int_2, ..., int_z),
            where batch_size is n.size(0) and int_1 x int_2, ..., int_z is
            the unraveled tuple of ints, self.out_shape.
        """
        return self.layers(n).reshape(n.size(0), *self.out_shape)