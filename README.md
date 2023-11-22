# HyperSINDy
This repository is the official implementation of HyperSINDy, first introduced in [HyperSINDy: Deep Generative Modeling of Nonlinear Stochastic Governing Equations](). 


## Requirements
All the requirements are contained in the environment.yml file\\
To install the requirements, run:
```
conda env create -f environment.yml
```

Then, activate the conda environment:
```
conda activate hypersindy
```

Or, with some conda environment that can run Python 3.9, you can manually install
the dependencies:
```
conda install scipy seaborn tensorboard matplotlib scikit-learn pandas jupyterlab pip
pip3 install pysindy torch==1.12.0 torchvision
```

## Installation
After installing dependencies, you can install hypersindy:
```
pip3 install hypersindy
```

## Example use case
See example.py as well.
``
python3

from hypersindy.library import Library
from hypersindy.net import Net
from hypersindy.trainer import Trainer
from hypersindy.dataset import SyntheticDataset
from hypersindy.utils import set_random_seed

set_random_seed(0)
device = 2
x_dim = 3
z_dim = 6
data_path = 'x_train.npy'

library = Library(x_dim)
net = Net(library, z_dim).to(device)
dataset = SyntheticDataset(library, fpath=data_path)
trainer = Trainer(net, library, "runs/1", "runs/1.pt", device=device)
trainer.train(dataset)

equations
results
``

Results can be viewed in tensorboard.
``
tensorboard --logdir="runs"
``

## Paper
To reproduce results from the paper, go to the paper folder and view the README there.
``
cd paper
``