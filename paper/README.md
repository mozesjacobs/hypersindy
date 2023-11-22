# HyperSINDy
This repository is the official implementation of [HyperSINDy: Deep Generative Modeling of Nonlinear Stochastic Governing Equations](). 


## Requirements
All the requirements are contained in the environment.yml file
To install the requirements, run:
```
conda env create -f environment.yml
```

Then, activate the conda environment:
```
conda activate hypersindy
```


## Data Generation
To generate the data in the paper, run these commands:
```
cd scripts
python3 generate_lorenz.py
python3 generate_rossler.py
python3 generate_lotkavolterra.py
python3 generate_lorenz96.py
```


## Training
To train the models in the paper, run these commands (assumes current directory is still scripts):
```
python3 train_lorenz.py --s1 1
python3 train_lorenz.py --s2 1
python3 train_lorenz.py --s3 1
python3 train_rossler.py --s1 1
python3 train_rossler.py --s2 1
python3 train_rossler.py --s3 1
python3 train_lotkavolterra.py --s1 1
python3 train_lorenz96.py --s1 1
```
It is recommended that you run each command concurrently (i.e. in separate terminal windows) so training is faster.
Alternatively, you can use the pretrained models described in the Pre-trained Models section below.

In addition, training progress can be viewed using tensorboard.
Open a separate terminal window and make sure the current directory is the base of the repository.
Then, run:
```
tensorboard --logdir="runs/"
```


## Pre-trained Models
Pre-trained models are located in the pretrained_models folder.
To use them, copy everything from pretrained_models into runs:
```
cp -r pretrained_models/* runs/
```
This is the recommended approach if you do not want to retrain all the models.
Alternatively, the scripts can be modified to load from the pretrained_models directory instead of
the runs directory, but this requires more code modifications (changing runs/cp_* to
pretrained_models/cp_* in multiple code files).


## Results
Once the models have been trained (or the the pre-trained models will be used),
the result-generating scripts can be run. To generate the results for the main
text figures, run these commands (assumes current directory is still scripts):
```
python3 fig2a.py
python3 fig2b.py
python3 fig3.py
python3 fig4a.py
python3 fig4b.py
```

To generate the results for the appendix figures, run these commands
(assumes current directory is still scripts):
```
python3 appendix_lorenz.py
python3 appendix_rossler.py
python3 appendix_lorenz_true.py
python3 appendix_rossler.py
python3 appendix_lorenz96.py
```

Alternatively, the notebooks folder contains .ipynb versions of these results-generating scripts.