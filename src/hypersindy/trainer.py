import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from hypersindy.equations import get_equations
from hypersindy.utils import save_model, make_optim

class Trainer():

    """Trainer object.

    Trainer object used to rain the HyperSINDy network.

    Attributes:
        self.net: The HyperSINDy object to train.
        self.library: The feature library (library_utils.py).
        self.gamma_factor: The learning rate decay factor.
        self.optim_weight_decay: The weight decay to use in the optimizer (different from lmda, which refers
            to the weight of the l0 regularization).
        self.optim: The torch.optim optimizer to train the network with.
        self.scheduler: The torch.optim.lr_scheduler to adjust learning rate with.
        self.board: The tensorboard SummaryWriter to log training progress.
        self.cp_path: A string for the path to save checkpoints.
        self.beta: The beta value (float) determining the strength of the KL divergence in the loss.
        self.beta_max: The max beta value to reach during beta warmup (but beta spike can be greater).
        self.beta_max_epoch: The number of epochs to do beta warmup.
        self.beta_inc: The amount (float) to increase beta each epoch during beta warmup.
        self.beta_spike: The value (float) to spike beta to (can be greater than beta_max).
        self.beta_spike_epoch: The epoch (int) at which to spike beta.
        self.lmda: The lambda value (float) determining the strength of the l0 term in the loss.
        self.lmda_max: The maximum lambda value to reach during lambda warmup (but lmda_spike can be greater).
        self.lmda_max_epoch: The number of epochs to do lambda warmup.
        self.lmda_inc: The amount (float) to increase lmda each epoch during lmda warmup.
        self.lmda_spike: The value (float) to spike lmda to (can be greater than lmda_max).
        self.lmda_spike_epoch: The epoch (int) at which to spike lmda.
        self.hard_threshold: The value (float) to permanently threshold the coefficients.
        self.threshold_interval: The epoch interval (int) to permanently threshold the coefficients.
        self.epochs: The number (int) of epochs to train for.
        self.batch_size: The number (int) of examples in a batch.
        self.clip: The value (float) to use for gradient clipping.
        self.model: The name (str) of the network being trained.
        self.device: The GPU devuce (0, 1, 2, 3) or cpu ("cpu") to train with.
        self.checkpoint_interval: The epoch interval (int) to save checkpoints.
        self.eval_interval: The epoch interval (int) to do evaluations.

    """

    def __init__(self, net, library, tb_path, cp_path,
                 optimizer="AdamW", learning_rate=5e-3, adam_reg=1e-5,
                 amsgrad=True, gamma_factor=0.999,
                 beta_init=0.01, beta_max=10.0, beta_max_epoch=100,
                 beta_spike=None, beta_spike_epoch=None,
                 hard_threshold=0.05, threshold_interval=100,
                 epochs=499, batch_size=250,
                 lmda_init=1e-2, lmda_max=1e-2, lmda_max_epoch=1,
                 lmda_spike=None, lmda_spike_epoch=None,
                 clip=1.0, device=0, checkpoint_interval=50,
                 eval_interval=50):
        """Initializes the Trainer.

        Initializes the Trainer.

        Args:
            net: The HyperSINDy network to train.
            library: The library object (from library_utils) to use.
            tb_path: The path (str) to create the tensorboard SummaryWriter with. 
            cp_path: The path (str) to save checkpoints at.
            optimizer: The torch.optim optimizer to train the network with. Default: AdamW
            learning_rate: The learning rate to use. Default: 5e-3
            adam_reg: The weight decay to use in the optimizer (different from
                lmda, which refers to the weight of the l0 regularization). Default: 1e-5
            amsgrad: Iff True, uses amsgrad in the optimizer. Default: True.
            gamma_factor: The per epoch learning rate decay factor (float). Default: 0.999.
            beta_init: The initial beta value (float) to warm up from.  Default: 0.01.
            beta_max: The beta value (float) to warm up to. Default: 1.0.
            beta_max_epoch: The number (int) of epochs to do beta warmup.
            beta_spike: The beta value to spike to (float). Default: 1.0.
            beta_spike_epoch: The epoch (int) at which to spike the beta value.
            hard_threshold: The value (float) to permanently threshold the coefficients. Default: 0.05
            threshold_interval: The epoch interval (int) to permanently threshold the coefficients. Default: 100.
            epochs: The number (int) of epochs to train for. Default: 999.
            batch_size: The number (int) of examples in a batch. Default: 250.
            lmda_init: The initial lmda value (float) to warm up from.  Default: 0.01.
            lmda_max: The lmda value (float) to warm up to. Default: None.
            lmda_max_epoch: The number (int) of epochs to do lmda warmup. Default: None.
            lmda_spike: The lmda value to spike to (float). Default: 1e-2.
            lmda_spike_epoch: The epoch (int) at which to spike the lmda value. Default: -1.
            clip: The value (float) to use for gradient clipping. Default: 1.0. 
            device: The GPU devuce (0, 1, 2, 3) or cpu ("cpu") to train with. Default: 0.
            checkpoint_interval: The epoch interval (int) to save checkpoints. Default: 50.
            eval_interval: The epoch interval (int) to do evaluations. Default: 50.

        Returns:
            A Trainer().
        """

        # objects
        self.net = net
        self.library = library
        self.gamma_factor = gamma_factor
        self.optim_weight_decay = adam_reg
        self.optim = make_optim(net, optimizer, learning_rate, adam_reg, 
                                amsgrad)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optim, gamma=gamma_factor)
        self.board = SummaryWriter(tb_path, purge_step=True)
        self.cp_path = cp_path

        # beta
        self.beta = beta_init
        self.beta_max = beta_max
        self.beta_max_epoch = beta_max_epoch
        if beta_max is None:
            self.beta_max = self.beta
        if beta_max_epoch is None:
            self.beta_max_epoch = 0
            self.beta_inc = 0
        else:
            self.beta_inc = (1.0 * self.beta_max) / self.beta_max_epoch
        self.beta_spike = beta_spike
        self.beta_spike_epoch = beta_spike_epoch

        # weight decay
        self.lmda = lmda_init
        self.lmda_max = lmda_max
        self.lmda_max_epoch = lmda_max_epoch
        if lmda_max is None:
            self.lmda_max = self.lmda
        if lmda_max_epoch is None:
            self.lmda_max_epoch = 0
            self.lmda_inc = 0
        else:
            self.lmda_inc = (1.0 * self.lmda_max) / self.lmda_max_epoch
        self.lmda_spike = lmda_spike
        self.lmda_spike_epoch = lmda_spike_epoch

        # hard thresholding
        self.hard_threshold = hard_threshold
        self.threshold_interval = threshold_interval

        # settings
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip = clip
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        
    def train(self, trainset):
        """Trains the network for on the dataset.

        Trains the network on the given dataset.

        Args:
            trainset: The torch dataset that the net will be trained with.
            
        Returns:
            None.
        """
        trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=1, drop_last=True)
        for epoch in range(self.epochs):
            # one train step
            recons, klds, regs = self.train_epoch(trainloader)

            # log losses
            self.log_losses(recons, klds, regs, epoch)

            # check if we need to exit
            if str(recons) == "nan" or str(klds) == "nan" or str(regs) == "nan":
                print("NAN Loss. Exiting.")
                break

            # threshold
            self.update_threshold_mask(epoch)

            if (epoch % self.checkpoint_interval == 0) and (epoch != 0):
                save_model(self.cp_path, self.net, self.library,
                           self.optim, self.scheduler, epoch)

            # eval
            if (epoch % self.eval_interval == 0) and (epoch != 0):
                self.eval_model(epoch)

            self.scheduler.step()
            self.update_beta()
            self.update_lmda()

            if epoch + 1 == self.beta_spike_epoch:
                self.beta_max = self.beta_spike
                self.beta = self.beta_spike

            if epoch + 1 == self.lmda_spike_epoch:
                self.lmda_max = self.lmda_spike
                self.lmda = self.lmda_spike
        
        save_model(self.cp_path, self.net, self.library,
                   self.optim, self.scheduler, epoch)
        self.eval_model(epoch)

        self.board.flush()
        self.board.close()

        return self


    def train_epoch(self, trainloader):
        """Trains the network for one epoch.

        Trains the network for one pass over the given dataloader using the given
        parameters.

        Args:
            trainloader: The torch dataloader that net will be trained with.
            
        Returns:
            A tuple (float_a, float_b, float_c) where float_a is the sum of the
            derivative calculation loss over all the batches in the dataloader,
            float_b is the sum of the KL divergence term over all the batches
            in the dataloader, and float_c is the sum of the L0 regularization
            term over all the batches in the dataloader.
        """

        # train mode
        self.net = self.net.train()

         # go through trainloader once
        recons, klds, regs = 0, 0, 0
        for i, batch in enumerate(trainloader):
            x, x_lib, x_dot, x_dot_standard = batch
            # device
            x = x.type(torch.FloatTensor).to(self.device)
            x_lib = x_lib.type(torch.FloatTensor).to(self.device)
            x_dot = x_dot.type(torch.FloatTensor).to(self.device)
            x_dot_standard = x_dot_standard.type(torch.FloatTensor).to(self.device)
            
            # one gradient step
            recon, kld, reg = self.train_batch(x, x_lib, x_dot, x_dot_standard)
            recons += recon
            klds += kld
            regs += reg
        return recons / len(trainloader), klds / len(trainloader), regs / len(trainloader)


    def train_batch(self, x, x_lib, x_dot, x_dot_standard):
        """Trains the hypernetwork on one batch of data.

        Feeds the hypernetwork the given data batch, calculates the loss, and
        performs one gradient step.

        Args:
            x: The raw state (torch.Tensor) to evaluate the model on. Should have
                shape (batch_size x x_dim).
            x_lib: The result (torch.Tensor) of calling sindy_library from
                model_utils.py on x. Should have shape (batch_size x library_dim).
            x_dot: The corresponding derivatives (torch.Tensor) of x. Should have
                shape (batch_size x x_dim).
            x_dot_standard: The standardized version of x_dot (torch.Tensor).
                Should have shape (batch_size x x_dim).
            
        Returns:
            A tuple (float_a, float_b, float_c) where float_a is the the derivative
            calculation loss over the given data batch, float_b is the KL
            divergence term in the loss function, and float_c is the L0
            regularzation term in the loss function.
        """
        x_dot_pred, reg, vae_kl = self.net(x, x_dot_standard, x_lib, self.device)
        recon = ((x_dot_pred - x_dot) ** 2).sum(1).mean()
        kld = vae_kl.mean() * self.beta
        reg = reg * self.lmda
        loss = recon + reg + kld
        self.optim.zero_grad()
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.optim.step()
        return recon.item(), kld.item(), reg.item()

    def update_threshold_mask(self, epoch):
        """Updates the threshold mask based on coefficient values.

        If coefficients are less than the given threshold, sets the corresponding
        value in the given networks threshold mask to 0. For HyperSINDy, samples
        a batch of coefficients and uses the mean over the batch as the
        the coefficients to be judged.

        Args:
            epoch: The current epoch (int) during training. If 
                epoch % threshold_timer != 0, or epoch == 0, will not threshold.

        Returns:
            None.
        """
        with torch.no_grad():
            if (epoch % self.threshold_interval == 0) and (epoch != 0):
                self.net.update_threshold_mask(self.hard_threshold, self.device)

    def log_losses(self, recon, kld, reg, epoch):
        """Logs the losses in tensorboard.

        Updates the given reconstruction and regularization loss in the given
        SummaryWriter.

        Args:
            recon: A float representing the error between the predicted and ground
                truth derivatives.
            kld: A float representing the KL divergence.
            reg: A float representing the L0 loss.
            epoch: The current epoch (int) during training.
                
        Returns:
            None.
        """
        self.board.add_scalar("Loss/dynamics", recon, epoch)
        self.board.add_scalar("Loss/kld", kld, epoch)
        self.board.add_scalar("Loss/l0", reg, epoch)

    def update_beta(self):
        """Updates the beta value.

        Increases beta by the given increment. If incrementing beta would make it
        exceed the given max, sets beta equal to the max.

        Args:
                
        Returns:
            None.
        """
        self.beta += self.beta_inc
        if self.beta > self.beta_max:
            self.beta = self.beta_max

    def update_lmda(self):
        """Updates the lambda value.

        Increases the lmda by the given increment. If incrementing lmda would
        make it exceed the given max, sets lmda equal to the max.

        Args:
                
        Returns:
            None.
        """
        self.lmda += self.lmda_inc
        if self.lmda > self.lmda_max:
            self.lmda = self.lmda_max

    def eval_model(self, epoch):
        """Evaluates the network.

            Logs the current discovered equations in tensorboard, as well as
            information about the L0 mask.

        Args:
            epoch: The current epoch (int) during training.
                
        Returns:
            None
        """
        self.net = self.net.eval()

        # get learned equations
        equations = get_equations(self.net, self.library, self.device)

        # log learned equations
        self.log_equations(equations, epoch)

        # show some mask info
        self.net = self.net.train()
        masks, pen = self.net.l0.get_mask(batch_size=250, device=self.device)
        res = ''
        for curr in pen:
            res += str(curr) + "  \n"
        self.board.add_text(tag="L0 Mask Train/Pen", text_string=res, global_step=epoch, walltime=None)

        self.net = self.net.eval()
        mask, pen = self.net.l0.get_mask(batch_size=250, device=self.device)
        mask = mask.detach().cpu().numpy()
        res = ''
        for curr in mask:
            res += str(curr) + "  \n"
        self.board.add_text(tag="L0 Mask Eval/Mask", text_string=res, global_step=epoch, walltime=None)

        self.net = self.net.train()

    def log_equations(self, equations, epoch):
        """Logs the equations into Tensorboard.

        Logs the given equations to the given Tensorboard.

        Args:
            equations: The equations as a dictionary with keys "mean" : [str], "std" : [str] to log (result of
                get_equations).
            board: The tensorboard to use.
            epoch: An int for the current epoch in training.

        Returns:
            None.
        """
        eq_mean = equations['mean'][0]
        eq_std = equations['std'][0]
        for i in range(1, len(equations['mean'])):
            eq_mean += "  \n" + str(equations['mean'][i])
            eq_std += "  \n" + str(equations['std'][i])
        self.board.add_text(tag="Equations/mean", text_string=eq_mean, global_step=epoch, walltime=None)
        self.board.add_text(tag="Equations/std", text_string=eq_std, global_step=epoch, walltime=None)