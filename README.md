# GATom

**GATom**: a Graph + Gated Attention-based neural neTwork for materials.

This repository host

### Preprocessing

This folder contains 


### Models 



**Note**: The model is still under development and we are currently writing a paper about the model to explain it with more detail.


### Parameters

In this folder, we display a sample `.yml` file for each of the available models.
Furthermore, we include the best set of hyperparameters (found using bayesian optimization) for each dataset.

### Performance

Currently, the best achieved performance for each of the datasets is

| Dataset                     	| _mp_gap_ (MAE eV) 	| _perovskites_ (MAE eV/unitcell) 	| _mp_e_form_ (MAE eV/atom) 	| _mp_is_metal_ (ROCAUC) 	|
|-----------------------------	|-------------------	|------------------------	|---------------------------	|------------------------	|
| Without Residual Connection 	| 0.23              	| 0.03                   	|                           	| 94                     	|
| With Residual Connection    	| 0.21              	| 0.02                   	|                           	| 95                     	|



### How to use

- train
- hyper
- eval

