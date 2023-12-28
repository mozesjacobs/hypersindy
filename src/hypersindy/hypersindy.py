import sys
import pickle
import torch

from hypersindy.net import Net
from hypersindy.library import Library
from hypersindy.dataset import DynamicDataset
from hypersindy.trainer import Trainer
from hypersindy.equations import get_equations
from hypersindy.utils import set_random_seed, init_weights


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
        """Initalizes HyperSINDy.

        Initializes the HyperSINDy model.

        Args:
            x_dim: The spatial dimension (int) of the data.
            z_dim: An int of the size of the latent vector (z) to be fed
                into the hypernetwork. Recommended: 2 times x_dim
            poly_order: The maximum order of the polynomial library (int)
            include_constant: Iff True, includes a constant term in the library
            hidden_dim: The dimension (int) of the hidden layers of the
                hypernet and the encoder
            stat_batch_size: The number (int) of samples to generate when
                generating coefficients (e.g. for printing equations or
                calculating mean / std)
            num_hidden: The number (int) of hidden layers in the hypernetwork

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
            epochs=499, batch_size=250, clip=1.0, gamma_factor=0.999,
            adam_reg=1e-5, run_path=None, num_workers=1):
        """Trains the HyperSINDy model.

        Trains the HyperSINDy model on the given data using the given
        parameters. If .fit() was called previously, calling it again resets
        the model (as though .fit() was not called previously) and trains
        again.

        Args:
            x: A torch.tensor of shape (batch_size x x_dim) for the state of
                the system.
            dt: The time (float) between adjacent state observations
                (e.g. between x_t and x_t+1).
            device: The cpu or gpu device to fit the HyperSINDy model with.
            beta: The weight (float) of the KL divergence term.
            beta_warmup_epoch: The number (int) of epochs to warm up to reach
                beta.
            beta_spike: The beta value (int) to use to later in training.
                Default: None (disables spiking)
            beta_spike_epoch: The epoch (int) at which to spike to.
                Default: None (disables spiking)
            checkpoint_interval: The epoch interval (int) to save check-points
                of the model during training.
            eval_interval: The epoch interval (int) to evaluate the model
                during training.
            learning_rate: The learning rate (float)
            hard_threshold: The value (float) to use when permanently setting
                terms to zero during training.
            hard_threshold_interval: The epoch interval (int) to permanently
                threshold terms.
            epochs: The number (int) of epochs to train for.
            batch_size: The size (int) of a batch in training.
            clip: The value to use for gradient clipping.
                Default: 1.0. Use None to disable.
            gamma_factor: The learning rate decay factor.
                Default: 0.999. Use None to disable.
            adam_reg: The regularization for the adam optimizer.
                Default: 1e-5.
            run_path: The folder (str) to store tensorboard logs during
                training.
            num_workers: The number (int) of workers to use for the dataloader.
                Default: 1.
        
        Returns:
            self: The fitted HyperSINDy model.
        """

        # Build / reset model
        self.__reset(device, dt)

        # Set device
        self.set_device(device)

        # Prepare dataset
        trainset = self.__prep_dataset(x, dt)

        # Prepare trainer
        trainer = self.__prep_trainer(run_path,
            learning_rate, beta, beta_warmup_epoch, beta_spike, beta_spike_epoch,
            hard_threshold, threshold_interval, epochs, batch_size, lmda_init,
            lmda_spike, lmda_spike_epoch, device, checkpoint_interval,
            eval_interval, clip, gamma_factor, adam_reg, num_workers)

        # Train
        trainer.train(trainset)

        # Put model in eval mode
        self.net = self.net.eval()

        return self
    
    def print(self, fname=None, round_digits=True, seed=None):
        """Prints the learned equations.

        Prints the mean and standard deviation of the equations learned by
        the fitted HyperSINDy model. Note that fit() must be called
        before print().

        Args:
            fname: The name of the file to print the equations to. The default
                is None, in which case print is directed to the system standard
                output. If fname doesn't end with .txt, appends .txt to it.
            round_digits: Iff True (default), rounds the coefficients to two
                decimal places.
            seed: The random seed to use before printing. The default is None,
                in which case the seed is not manually specified.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        eqs = self.equations(round_digits, seed)
        orig = sys.stdout
        if fname is not None:
            fname = self.__fix_fpath(fname, ".txt")
            sys.stdout = open(fname, "w")
        for stat in eqs:
            print(stat)
            for eq in eqs[stat]:
                print(eq)
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

    def simulate_mean(self, x0, ts=10000, seed=None, dt=None):
        """Generate a trajectory with the mean equation.

        Generates a trajectory using the mean of the discovered equations.

        Args:
            x0: The initial condition (torch.Tensor of shape (x_dim)).
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
        xt = xt.unsqueeze(0)
        trajectories = [xt]
        coefs = self.coefs().mean(0)
        for i in range(ts - 1):
            theta_x = self.transform(xt)
            xt = xt + torch.matmul(theta_x, coefs) * dt
            trajectories.append(xt)
        trajectories = torch.transpose(torch.stack(trajectories, dim=0), 0, 1)
        return trajectories.squeeze().detach().cpu().numpy()

    
    def equations(self, round_digits=True, seed=None):
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
        return get_equations(self.net, self.library, self.device, round_digits, seed)
    
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
        return self.net.get_masked_coefficients(z, batch_size,
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
        theta_x = self.transform(x).unsqueeze(1)
        return torch.bmm(theta_x, coefs).squeeze(1)
    
    def save(self, fpath):
        """Saves the model.

        Saves the state dictionary of the learned network to the given fpath.

        Args:
            fpath: The name of the file to save the state dictionary to. Should
                end in .pt. If fpath doesn't end with .pt, appends .pt to it.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        fpath = self.__fix_fpath(fpath, ".pt")
        torch.save(self.net.state_dict(), fpath)
        return self
    
    def load(self, fpath, device='cpu'):
        """Loads the model.

        Loads the state dictionary of the learned network in the given fpath.
        Loads the network in eval mode.

        Args:
            fpath: The name of the file to load the state dictionary from.
                Should end in .pt. If fpath does not end in .pt, appends .pt
                to it.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        fpath = self.__fix_fpath(fpath, ".pt")
        self.__reset()
        self.net.load_state_dict(torch.load(fpath))
        self.net = self.net.eval()
        self.set_device(device)
        return self

    def to(self, device):
        """Sets the device.

        Wrapped for set_device. Sets the device to use the model on.

        Args:
            device: The cpu or gpu device to use.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        return self.set_device(device)

    def set_device(self, device):
        """Sets the device.

        Sets the device to use the model on.

        Args:
            device: The cpu or gpu device to use.
        
        Returns:
            self: The fitted HyperSINDy model.
        """
        self.device = device
        self.net = self.net.to(device)
        return self

    def __reset(self, device='cpu', dt=0.01):
        """
        Resets the model, as though training never occurred.
        """
        self.library = Library(self.x_dim, self.poly_order,
                               self.include_constant)
        self.net = Net(self.library, self.z_dim, self.hidden_dim,
                       self.stat_batch_size, self.num_hidden)
        self.net.apply(init_weights)
        self.net = self.net.train()
        self.set_device(device)
        self.dt = dt
        return self
    
    def __prep_dataset(self, x, dt):
        """
        Returns a DynamicDataset of x with the given dt.
        """
        return DynamicDataset(x, self.transform(x), dt)
    
    def __prep_trainer(self, run_path, learning_rate, beta_max,
        beta_max_epoch, beta_spike, beta_spike_epoch,
        hard_threshold, threshold_interval, epochs, batch_size, lmda_init,
        lmda_spike, lmda_spike_epoch, device, checkpoint_interval,
        eval_interval, clip, gamma_factor, adam_reg, num_workers):
        """
        Returns a Trainer.
        """

        # Hard-coded defaults - these work well in general
        beta_init = 0.01
        lmda_max = lmda_init
        lmda_max_epoch = 1
        optim = "AdamW"
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
                          eval_interval, num_workers)
        return trainer

    def __fix_fpath(self, fname, end):
        """
        If fname does not end with end, returns fname + end. If fname ends with
        end, returns fname.
        """
        if not fname.endswith(end):
            fname += end
        return fname