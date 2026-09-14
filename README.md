# SCINN: Solution Character Informed Neural Network for Scalar Conservation Laws

Weiheng Zeng, Ruoxi Lu, Kun Wang, Tiegang Liu

1. School of Mathematical Sciences, Beihang University
2. International Research Center for Mathematics and Interdisciplinary Sciences, Hangzhou International Innovation Institute of Beihang University

## Introduction

🤔🤔🤔

This project is developed based on the source code architecture of Gradient-annihilated PINNs (https://github.com/Netzuel/GA_PINNs_Repository). The structure is as follows:

-- Cases (the definition of solution cases, including exact solutions and flux functions)

-- Example_Problem (the directory for saved models and training data)

-- Images (the output images generated during training)

-- models.py (the implementation of the SCINN model)

-- utils.py (utility functions: domain generation, initial/boundary conditions, AN discontinuity indicator, plotting, and result saving)

-- training_script.py (the main training script)

-- config.json (the configuration file for problem setup, network architecture, and training parameters)

## Requirements

🥸🥸🥸

This project bases on **Python 3.9**. See `requirements.txt` for the full list of dependencies:

```
tqdm==4.64.0
h5py==3.6.0
scipy==1.7.3
torch==1.12.0
matplotlib==3.6.2
```

## Usage

😇😇😇

Run the training script to start training:

```
python training_script.py
```

The training configuration is controlled by `config.json`:

-- **physical**: problem setup including case number, time steps (`N_t`), space points (`N_x`), and initial points (`N_0`)

-- **neural**: activation functions and training parameters, including:
  - `PRE` — whether to preprocess (normalize) input coordinates
  - `WE` — whether to use weighted equation loss (Lambda weighting)
  - `NUM` — loss weights `[w_E, w_I, w_U, w_S]` for equation, IC/BC, implicit-formula, and shock terms
  - `N_PAR` — epoch interval for saving checkpoints and triggering RAR
  - `N_AN` — epoch interval for re-computing the discontinuity indicator
  - `eta_IF` — multiplier for implicit-formula point reweighting
  - `alpha`, `beta` — WE denominator and exponent parameters
  - `num_hidden`, `num_neurons` — DNN architecture

-- **training_process**: data type, export paths, optimizer, learning rate, epochs, and random seed

To define a new solution case, add a new file under `Cases/` following the structure of `solu_case5.py`, which specifies the temporal/spatial range, flux function `f(u)`, its derivative `df(u)`, and the exact solution.

## Docs

🥳🥳🥳

The paper is published on [Journal of Machine Learning].(https://doi.org/10.4208/jml.251201)
