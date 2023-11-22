import sys
import torch

from hypersindy.net import Net
from hypersindy.library import Library
from hypersindy.dataset import DynamicDataset
from hypersindy.trainer import Trainer
from hypersindy.equations import get_equations
from hypersindy.utils import set_random_seed


class HyperSINDy:
    """A HyperSINDy model.

    The HyperSINDy model that can be fit on data to discover a distribution of
    governing equations.

    Attributes:

        self.library: The SINDy library object (Theta from the manuscript)
            used to transform the state, x.
        self.net: The PyTorch model containing a variational encoder and
            hypernetwork
        self.dt: The time between adjacent state observations
            (e.g. between x_t and x_t+1)
        self.device: The cpu or gpu device to run HyperSINDy with

    """
    def __init__(self, x_dim=3, z_dim=6, poly_order=3, include_constant=True,
                 hidden_dim=64, stat_batch_size=250, num_hidden=5):
        """Initalizes the network.

        Initializes the HyperSINDy network.

        Parameters:
            x_dim: The spatial dimension (int) of the data.
            z_dim: An int of the size of the latent vector (z) to be fed
                into the hypernetwork. Recommended: 2 times x_dim
            poly_order: 
            include_constant:
            hidden_dim:
            stat_batch_size:
            num_hidden:

        Returns:
            A HyperSINDy().
        """
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.poly_order = poly_order
        self.include_constant = include_constant
        self.hidden_dim = hidden_dim
        self.stat_batch_size = stat_batch_size
        self.num_hidden = num_hidden

    def fit(self, x, dt, device,
            beta, beta_warmup_epoch, beta_spike, beta_spike_epoch,
            lmda_init, lmda_spike, lmda_spike_epoch,
            checkpoint_interval=50, eval_interval=50,
            learning_rate=5e-3, hard_threshold=0.05, threshold_interval=100,
            epochs=499, batch_size=250, run_path=None):
        """Trains the HyperSINDy model.

        Trains the HyperSINDy model on the given data using the given
        parameters. If .fit() was called previously, calling it again resets
        the model (as though .fit() was not called previously) and trains
        again.

        Args:
            x: A torch.tensor of shape (batch_size x x_dim) for the state of
                the system.
            x_dot: A torch.tensor of shape (batch_size x x_dim) for the
                derivative of x.
            device: The cpu or gpu device to fit the HyperSINDy model with.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        
        # Set device
        self.set_device(device)

        # Build / reset model
        self.__reset(device, dt)

        # Prepare dataset
        trainset = self.__prep_dataset(x, dt)

        # Prepare trainer
        trainer = self.__prep_trainer(run_path,
            learning_rate, beta, beta_warmup_epoch, beta_spike, beta_spike_epoch,
            hard_threshold, threshold_interval, epochs, batch_size, lmda_init,
            lmda_spike, lmda_spike_epoch, device, checkpoint_interval,
            eval_interval)

        # Train
        trainer.train(trainset)

        return self
    
    def print(self, fname=None, round=True, seed=None):
        """Prints the learned equations.

        Prints the mean and standard deviation of the equations learned by
        the fitted HyperSINDy model. Note that fit() must be called
        before print().

        Args:
            fname: The name of the file to print the equations to. The default
                is None, in which case print is directed to the system standard
                output.
            round: Iff True (default), rounds the coefficients to two decimal
                places.
            seed: The random seed to use before printing. The default is None,
                in which case the seed is not manually specified.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        eqs = self.equations(self.net, self.library, self.device, round, seed)
        orig = sys.stdout
        if fname is not None:
            sys.stdout = open(fname, "w")
        for eq in eqs: print(eq)
        sys.stdout = orig
        return self
    
    def simulate(self, x0, batch_size, ts=10000, seed=None, dt=None):
        """Generates sample trajectories.

        Generates a batch of sample trajectories from the given initial
        condition.

        Args:
            x0: The initial condition (torch.Tensor of shape (x_dim)).
            batch_size: The number of trajectories to simulate.
            ts: The number (int) of timesteps to simulate, including the
                provided initial condition. The default is 10000.
            seed: The random seed (int) to use. The default is None.
            dt: The time between adjacent state observations. The default
                is None, in which case self.dt is used.
        
        Returns:
            The sampled trajectories as numpy array of shape
            (batch_size, ts, x_dim).
        """
        if seed is not None:
            set_random_seed(seed)
        if dt is None:
            dt = self.dt
        xt = x0.type(torch.FloatTensor).to(self.device)
        xt = xt.unsqueeze(0).expand(batch_size, -1)
        trajectories = [xt]
        for i in range(ts - 1):
            xt = xt + self.derivative(xt) * dt
            trajectories.append(xt)
        trajectories = torch.transpose(torch.stack(trajectories, dim=0), 0, 1)
        return trajectories.detach().cpu().numpy()
    
    # get equations
    def equations(self, round=True, seed=None):
        """Gets the equations.

        Returns a list of the mean and standard deviation of the equations
        learned by HyperSINDy.

        Args:
            round: Iff True (default), rounds the coefficients to two decimal
                places.
            seed: The random seed to use before printing. The default is None,
                in which case the seed is not manually specified.
        
        Returns:
            A dictionary with keys {'mean', 'std'}, where the values are the
            learned equations.
        """
        return get_equations(self.net, self.library, self.device, round, seed)
    
    def coefs(self, batch_size=None, z=None):
        """Samples coefficients.

        Samples coefficients learned by the HyperSINDy model.

        Args:
            batch_size: The number (int) of sets of coefficients to return. The
                default is None, in which case self.stat_batch_size is used.
                One of batch_size or z must be specified.
            z: The latent vector (torch.Tensor of shape
                (num_samples, self.z_dim)) to generate coefficients with. The
                default is None. One of batch_size or z must be specified. If
                both batch_size and z are given, z is used instead.
            
        Returns:
            The sampled coefficients as a torch.Tensor of shape
            (batch_size, library_dim, x_dim).
        """
        if batch_size is None:
            batch_size = self.stat_batch_size
        return self.net.get_masked_coefficients(self, z, batch_size,
            self.device)
    
    def transform(self, x):
        """Creates theta(x).
        
        Transforms the given state matrix with the theta library.

        Args:
            x: A torch.tensor of shape (batch_size, x_dim) to transform into
                the theta library.

        Returns:
            Theta(x) as a torch.tensor of shape (batch_size, library_dim).
        """
        return self.library.transform(x)

    def derivative(self, x):
        """Predicts the derivative.

        Predicts the derivative of the given state.

        Args:
            x: A torch.tensor of shape (batch_size, x_dim) to predict the
                derivative.
        
        Returns:
            The predicted derivatives as a torch.tenor of shape
            (batch_size, x_dim).
        """
        batch_size = x.size(0)
        coefs = self.coefs(batch_size)
        theta_x = self.transform(x)
        return torch.bmm(theta_x, coefs).squeeze(1)
    
    def save(self):
        pass
    
    def load(self):
        pass

    def set_device(self, device):
        self.device = device
        return self

    def __reset(self, device='cpu', dt=0.01):
        self.library = Library(self.x_dim, self.poly_order,
                               self.include_constant)
        self.net = Net(self.library, self.z_dim, self.hidden_dim,
                       self.stat_batch_size, self.num_hidden).to(device)
        self.dt = dt
        return self
    
    def __prep_dataset(self, x, dt):
        return DynamicDataset(x, self.transform(x), dt)
    
    def __prep_trainer(self, run_path, optim, learning_rate, beta_max,
        beta_max_epoch, beta_spike, beta_spike_epoch,
        hard_threshold, threshold_interval, epochs, batch_size, lmda_init,
        lmda_spike, lmda_spike_epoch, device, checkpoint_interval,
        eval_interval):

        # Hard-coded defaults - these work well in general
        beta_init = 0.01
        lmda_max = lmda_init
        lmda_max_epoch = 1
        optim = "AdamW"
        clip = 1.0
        adam_reg = 1e-5
        gamma_factor = 0.999
        amsgrad =  True

        # Tensorboard and checkpoint paths
        tb_path = run_path
        cp_path = run_path + ".pt"

        # Build Trainer
        trainer = Trainer(self.net, self.library, tb_path, cp_path, optim,
                          learning_rate, adam_reg, amsgrad,
                          gamma_factor, beta_init, beta_max,
                          beta_max_epoch, beta_spike, beta_spike_epoch,
                          hard_threshold, threshold_interval,
                          epochs, batch_size,
                          lmda_init, lmda_max, lmda_max_epoch,
                          lmda_spike, lmda_spike_epoch,
                          clip, device, checkpoint_interval,
                          eval_interval)
        return trainer