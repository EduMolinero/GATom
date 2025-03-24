# GATom

**GATom**: a Graph + Gated Attention-based neural neTwork for material property prediction.

This repository host a Graph Neural Network Model to predict the properties of materials for their crystalline structure.
The structures is converted into a graph (which retains all the symmetries) and then it is feeded into the network to infer the desired property, e.g., the value of the gap.

## Preprocessing

This folder contains the `AtomLoader.py`file which generates the atom features vector representations.
It also contains a `main.py` file to generate those representations.

## Crystal Builder

This folder contains the files need to convert the crystal structure into a graph using a variety of algorithms.
It is a fork from the `keras-crystal` (link) library adapted to work with `Pytorch` objects.

## Models 

**Note**: The model is still under development and we are currently writing a paper about the model to explain it with more detail.

## Training & Evaluation

The training plus the evaluation is done through the `Trainer` class inside `training.py`

## Parameters

In this folder, we display a sample `.yml` file for each of the available models.
Furthermore, we include the best set of hyperparameters (found using bayesian optimization) for each dataset.

## Performance

Currently, the best achieved performance for each of the datasets is

| Dataset                     	| _mp_gap_ (MAE eV) 	| _perovskites_ (MAE eV/unitcell) 	| _mp_e_form_ (MAE eV/atom) 	| _mp_is_metal_ (ROCAUC) 	|
|-----------------------------	|-------------------	|------------------------	|---------------------------	|------------------------	|
| Without Residual Connection 	| 0.23              	| 0.03                   	|                           	| 94                     	|
| With Residual Connection    	| 0.21              	| 0.02                   	|                           	| 95                     	|


## How to use

We provide a sample `.sbatch` script on how to run the model. 
It only needs a `.yml` configuration file on the same folder.
Right now, the `main.py` accepts two possible types of calculations which are given as arguments in the parser.

- `--single_calc`: Performs a full training (alongside obtaining evaluation and testing errors) routine for the given dataset.
- `--hyperparam_optim`: Performs a hyperparameter optimization using the Ray Tune library for the given dataset.

## Diagram

We provide a flow diagram of how to proceed for a given dataset:

```mermaid
flowchart LR;
    A[(Dataset)] -->|Atom Features| B(AtomLoader);
    A -->|Graph Representation| G(crystal_builder);
    G --> C(data);
    B --> C;
    C --> D{GATom};
    D --> E(Trainer);
    E --> F[Results];
```