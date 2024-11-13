# Estimating Dark Matter Halo Masses with Machine Learning

This repository contains the code and data used in the paper "Estimating Dark Matter Halo Masses in Simulated
Galaxy Clusters with Graph Neural Networks" (Garuda et al. 2024). 

## Data
We use TNG-Cluster snapshot `99` $z=0$ from the TNG Cluster simulations. This data will later be released by the IllustrisTNG team.

## Requirements
This code requires you to download `pytorch` and `torch-geometric`; you can install them by checking the [PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

But, you can also use the `requirements.txt` file to install all the dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage
All the source required to train the model and make your datasets from the raw data is provided in the `src` directory. Best method would be to use a Weights and Biases account to log all the results. You can create a free account [here](https://wandb.ai/). 

Easiest way would be to create the datasets using `data_load.py` and then changing your file paths in `wandb_run_advgnn.py` and running the following command:

```bash
wandb sweep sweep.yaml 
```

This will run the sweep and log all the results to your Weights and Biases account. 

## Citation
TODO