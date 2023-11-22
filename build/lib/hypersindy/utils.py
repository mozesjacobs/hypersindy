import os
import json
import torch
import numpy as np
import random
import torch.nn as nn

def make_folder(name):
    """Creates a folder.

    Creates the given folder, unless it already exists.

    Args:
        name: A str denoting the path to the folder to create.
    
    Returns:
        None
    """
    if not os.path.isdir(name):
        os.makedirs(name)
        
def set_random_seed(seed):
    """Sets the random seed.

    Sets the random seeds for reproducibility.
    The code for this function was taken from:
    https://github.com/pawni/BayesByHypernet_Pytorch/blob/master/train.ipynb

    Args:
        seed: the random seed (integer)
    
    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def save_model(cp_path, net, library, optim, scheduler, epoch):
    """Saves the current checkpoint.

    Saves the current network, optimizer, scheduler, and epoch as a .pt file
    at the specified path. Overwrites the file if one exists at the given path.

    Args:
        cp_path: The string (relative) path to the checkpoint to save.
        net: The network (nn.Module) to save.
        library: The library (nn.Module) to save.
        optim: The optimizer (torch.optim) to save.
        scheduler: The torch.optim.lr_scheduler to save.
        epoch: The current epoch in training to save.

    Returns:
        None
    """
    checkpoint = {'epoch': epoch,
                  'model': net.state_dict(),
                  'library': library,
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, cp_path)

def save_args(args, args_path):
    """Saves the arguments.

    Saves the given args as a json file at the given path. Overwrites the file
    at the given path if it already exists.

    Args:
        args: The argparser object return by parse_args() in the file
            cmd_line.py.
        args_path: The path to the json file to save args as.

    Returns:
        None
    """
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def init_weights(layer):
    """Initializes the weights.

    Initializes the weights of the layer. For Linear laters, uses xavier
    uniform initialization. For LayerNorm and BatchNorm1d layers, only 
    initializes the bias terms with the value 0.01. 

    Args:
        layer: The layer (nn.Linear, nn.LayerNorm, nn.BatchNorm1d) to
        initialize.
    
    Returns:
        None
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
    elif isinstance(layer, nn.LayerNorm):
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.BatchNorm1d):
        layer.bias.data.fill_(0.01)

def make_optim(net, optimizer, lr, weight_decay, amsgrad):
    """Creates an optimizer.

    Creates a PyTorch optimizer for the network with the settings
    specified by args.

    Args:
        net: The network to create the optimizer for.
        optimizer: A string denoting which optimizer to create.
            Options: {Adam, AdamW}. Prints an error and exits of optimizer
            is not one of the aforementioned options.
        lr: The learning rate (float) for the optimizer.
        weight_decay: The weight decay (float) for the optimizer.
        amsgrad: IFF True (bool) uses amsgrad in the optimizer.
    
    Returns:
        The optimizer (a torch.optim object).
    """
    if optimizer == "Adam":
        optim = torch.optim.Adam(net.parameters(), lr=lr,
            weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer == "AdamW":
        optim = torch.optim.AdamW(net.parameters(), lr=lr,
            weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer == "SGD":
        optim = torch.optim.SGD(net.parameters(), lr=lr,
            weight_decay=weight_decay, momentum=0.9, nesterov=True)
    else:
        print("ERROR: args.optimizer must be Adam, AdamW, or SGD, not " + optimizer)
        exit()
    return optim


def load_checkpoint(cp_path, net, library, optim, scheduler, device):
    """Loads the last checkpoint.

    Loads the latest checkpoint at cp_path into the latest epoch and the given
    network, optimizer, and scheduler.

    Args:
        cp_path: The string (relative) path to the checkpoint to load.
        net: The network (nn.Module) to load into.
        optim: The optimizer (torch.optim) to load into.
        scheduler: The torch.optim.lr_scheduler to load into.
        device: The cpu or gpu device to load the checkpoint (and network)
            onto. For cpu, device must be "cpu". For gpu, the device must be
            an integer corresponding to the gpu to be used (i.e.: 0 or 1 or 2
            or 3).

    Returns:
        A tuple (Net, Optim, Scheduler, Initial_e). Net is the nn.Module that
        was loaded from the checkpoint. Optim is the torch.optim that was
        loaded from the checkpoint. Scheduler is the torch.optim.lr_scheduler
        that was loaded from the checkpoint. Initial_e is an integer describing
        which epoch in training was loaded from the checkpoint.
    """
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    optim.load_state_dict(checkpoint['optimizer'])
    library.load_state_dict(checkpoint['library'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    initial_e = checkpoint['epoch']
    return net, library, optim, scheduler, initial_e