import torch
import torch.nn as nn
from hypersindy.hypernet import HyperNet
from hypersindy.l0 import L0Norm


class Net(nn.Module):
    """A HyperSINDy model.

    The HyperSINDy model that uses a variational encoder and a hypernetwork
    to learn governing equations.

    Attributes:

        self.library: The SINDy library object (from library_utils)
            used to transform the data.
        self.library_dim: The number (int) of terms in the SINDy library.
        self.x_dim: The spatial dimension (int) of the data.
        self.statistic_batch_size: An integer indicating the default number of
            samples to draw when sampling coefficients if not specified.
        self.hidden_dim: An int of the size of the Linear layers in
            the hypernetwork.
        self.z_dim: An int of the size of the latent vector (z) to be fed
            into the hypernetwork.
        self.model: A string denoting the name of the model ("HyperSINDy").
            This attribute is an artifact of old code and can be ignored (will
            be removed).
        self.hypernet: A HyperNet (from HyperNet) that generates coefficients.
        self.l0: The learned mask used to zero out coefficients (from L0Module).
        self.encoder: A neural network used to encode pairs of (x, x_dot).
        self.fc1: A nn.Linear layer used to calculate the mean of
            q(z | x, x_dot).
        self.fc2: A nn.Linear layer used to calculate the log variance of 
            q(z | x, x_dot).
    """
    def __init__(self, library, z_dim=6, hidden_dim=64,
                 stat_batch_size=500, num_hidden=4):
        """Initalizes the network.

        Initializes the HyperSINDy network.

        Args:
            args: The argparser object return by parse_args() in the file
                cmd_line.py.
            hyperparams: The argparser object returned by parse_hyperparams() in 
                the file cmd_line.py
            library: The SINDy library object (from src.utils.library_utils)
                used to transform the data.

        Returns:
            A Net().
        """
        super(Net, self).__init__()
        
        self.library = library
        self.library_dim = library.get_library_size()
        self.x_dim = self.library.x_dim
        self.statistic_batch_size = stat_batch_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.model = "HyperSINDy"

        self.hypernet = HyperNet(self.z_dim, (self.library_dim * self.x_dim, ),
            [self.hidden_dim for _ in range(num_hidden)])

        self.l0 =  L0Norm(self.library_dim, self.x_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim + self.x_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.z_dim)
        )

        # mu and logvar
        self.fc1 = nn.Linear(self.z_dim, self.z_dim)
        self.fc2 = nn.Linear(self.z_dim, self.z_dim)
        
        
    def forward(self, x, x_dot, x_lib=None, device=0):
        """Runs the forward pass.

        Runs the forward pass, encoding the given state and derivative pairs,
        sampling from the latent space, generating coefficients, and predicting
        the derivative.

        Args:
            x: A torch.tensor of shape (batch_size x x_dim) for the state of
                the system.
            x_dot: A torch.tensor of shape (batch_size x x_dim) for the
                derivative of x.
            x_lib: The sindy_library form of x. Should be a torch.tensor of
                shape (batch_size x library_dim). Default is None. If None,
                creates a SINDy library out of x.
            device: The cpu or gpu device to do calculations with. To use cpu,
                device must be "cpu". To use gpu, specify which gpu to use as
                an integer (i.e.: 0 or 1 or 2 or 3). 
            t: A torch.tensor of shape (batch_size, ) denoting the time
                corresponding to each state, derivative pair. Default is None.
                Can leave as None (artifact of old code that will be removed).
        
        Returns:
            A tuple of (tensor_a, tensor_b, tensor_c), where tensor_a is the
            calculated derivative as a torch.Tensor, tensor_b is the l0 penalty,
            and tensor_c is the KL divergence.
            The shape of tensor_a is: (batch_size x x_dim).
            The shape of tensor_b is: ().
            The shape of tensor_c is: (batch_size, ).
        """
        x = x.type(torch.FloatTensor).to(device)
        x_dot = x_dot.type(torch.FloatTensor).to(device)

        # encode and get kl
        n, mu, logvar = self.qz(x, x_dot)
        kl = self.kl(mu, logvar)
        
        # sample l0 mask and get l0 penalty
        l0_mask, pen = self.l0.get_mask(x.size(0), device)

        # get coefficients
        coeffs = self.get_masked_coefficients(n=n, device=device, l0_mask=l0_mask)

        # transform library
        if x_lib is not None:
            x_lib = x_lib.type(torch.FloatTensor).to(device)
        x_lib = self.library.transform(x, x_lib)

        # calculate derivative and return
        return self.dx(x_lib, coeffs), pen.sum(), kl

    def qz(self, x, x_dot, sample=True):
        """Returns q(z | x, x_dot).

        Calculates q(z | x, x_dot) and samples from the distribution.

        Args:
            x: A torch.tensor of shape (batch_size x x_dim) for the state of
                the system.
            x_dot: A torch.tensor of shape (batch_size x x_dim) for the
                derivative of x.
            sample: If True (default), uses logvar in the sampling. If False,
                then returns just the mean.

        Returns:
            A tuple of (n, mu, logvar), where each is a torch.tensor of shape
            (batch_size x z_dim). n is a sample from q(z | x, x_dot),
            mu is the mean of q(z | x, x_dot). logvar is the log variance of
            q(z | x, x_dot).
        """
        e = self.encode(x, x_dot)
        mu, logvar = self.posterior(e)
        n = self.reparameterize(mu, logvar, sample=sample)
        return n, mu, logvar

    def encode(self, x, x_dot):
        """Encodes the given vectors.

        Encodes the given vectors into one that can be used to calculate q.

        Args:
            x: A torch.tensor of shape (batch_size x x_dim) for the state of
                the system.
            x_dot: A torch.tensor of shape (batch_size x x_dim) for the
                derivative of x.

        Returns:
            A torch.tensor of shape (batch_size x z_dim).
        """
        return self.encoder(torch.cat((x, x_dot), dim=1))

    def posterior(self, encoded):
        """Returns the mean and log variance of q.

        Converts the given encoded vector into the mean and log variance of
        the q distribution.

        Args:
            encoded: A torch.tensor of shape (batch_size x z_dim) returned
                by self.encode().

        Returns:
            A tuple of (mu, logvar) where mu and logvar are both torch.tensor
            of shape (batch_size x z_dim) representing the mean and log
            variance of q, respectively.
        """
        return self.fc1(encoded), self.fc2(encoded)

    def reparameterize(self, mu, logvar, sample=True):
        """Reparameterizes mu and logvar.

        Reparameterizes mu and logvar to return a sample from q.

        Args:
            mu: A torch.tensor of shape (batch_size x z_dim) for the mean
                of q.
            logvar: A torch.tensor of shape (batch_size x z_dim) for the log
                of the variance of q.
            sample: If True (default), uses logvar in the sampling. If False,
                then returns just the mean.

        Returns:
            The z sample as a torch.tensor of shape (batch_size x z_dim).
        """
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(logvar)
            return mu + std * eps
        return mu

    def kl(self, mu, logvar):
        """Calculate the KL divergence.

        Calculates the KL divergence between q(mu, logvar) and the
        standard normal distribution with diagonal covariance.
        Assumes q is a normal distribution.

        Args:
            mu: A torch.tensor of shape (batch_size x z_dim) for the mean
                of q.
            logvar: A torch.tensor of shape (batch_size x z_dim) for the log
                of the variance of q.

        Returns:
            The calculated KL divergence as a torch.tensor of shape [batch_size].
        """
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    
    def dx(self, library, coefs):
        """Calculate the derivative.

        Given the library terms and the SINDy coefficients, calculate the
        derivative.

        Args:
            library: The SINDy library terms as a torch.Tensor of shape
                (batch_size x library_dim).
            coefs: The SINDy coefficients as a torch.Tensor of shape
                (batch_size x library_dim x x_dim).

        Returns:
            The calculated derivatives as a torch.Tensor of shape
            (batch_size x x_dim).
        """
        return torch.bmm(library.unsqueeze(1), coefs).squeeze(1)

    def sample_coeffs(self, n=None, batch_size=None, device=0):
        """Samples coefficients.

        Samples coefficients from the hypernetwork.

        Args:
            n: The torch.Tensor of shape (-1, z_dim) to feed into the
                hypernetwork. If None, samples from a N(0, 1) distribution.
            batch_size: If n is None, samples a vector of shape 
                (batch_size x self.z_dim) from a N(0, 1) distribution.
            device: If n is None, the sampled noise vector uses this 
                cpu or gpu device. To use cpu, device must be "cpu". To use
                gpu, specify which gpu to use as an integer (i.e.: 0 or 1 or 2
                or 3). 

        Returns:
            Sampled SINDy coefficients as a torch.Tensor of shape
            (n.size(0) x self.z_dim), or if n is None, of shape
            (batch_size x self.z_dim).
        """
        if batch_size is None:
            batch_size = self.statistic_batch_size
        if n is None:
            n = torch.randn([batch_size, self.z_dim], device=device)
        return self.hypernet(n)

    def get_masked_coefficients(self, n=None, batch_size=None, device=0, l0_mask=None):
        """Samples thresholded coefficients (masked coefficients).

        Samples coefficients from the hypernetwork and thresholds them using
        the l0 mask and the hard threshold mask.

        Args:
            n: The torch.Tensor of shape (-1 x z_dim) to feed into the
                hypernetwork. If None, samples from a N(0, 1) distribution.
            batch_size: If n is None, samples a vector of shape 
                (batch_size x self.z_dim) from a N(0, 1) distribution.
            device: If n is None, the sampled noise vector uses this 
                cpu or gpu device. To use cpu, device must be "cpu". To use
                gpu, specify which gpu to use as an integer (i.e.: 0 or 1 or 2
                or 3). 
            l0_mask: The torch.Tensor to use to mask the coefficients.
                Should be the result of calling L0Norm.get_mask().
                If l0_mask is None (default), then makes a call get_mask().

        Returns:
            Sampled SINDy coefficients as a torch.Tensor of shape
            (n.size(0) x self.z_dim), or if n is None, of shape
            (batch_size x self.z_dim).
        """
        coefs = self.sample_coeffs(n, batch_size, device)
        coefs = coefs.reshape(-1, self.library_dim, self.x_dim)
        if l0_mask is None:
            l0_mask, _ = self.l0.get_mask(coefs.size(0), device)
        return coefs * l0_mask

    def update_threshold_mask(self, threshold, device):
        """Permanently thresholds coefficients.

        Samples coefficients using self.get_masked_coefficients. Gets the
        indices of any SINDy coefficients with absolute value less than the
        given threshold. Sets self.threshold_mask[indices] = 0.

        Args:
            threshold: The threshold (float) to use.
            device: The cpu or gpu device to sample coefficients with. To use
                cpu, device must be "cpu". To use gpu, specify which gpu to
                use as an integer (i.e.: 0 or 1 or 2 or 3). 
        
        Returns:
            None
        """
        if threshold is not None:
            coefs = self.get_masked_coefficients(device=device)
            self.l0.hard_threshold_mask[torch.abs(coefs.mean(0)) < threshold] = 0