# %%
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from astropy.cosmology import Planck15 as cosmo
from torch_geometric.loader import ClusterData, ClusterLoader
from tqdm import tqdm

from cosmic_graph import create_graph, get_train_valid_indicies_cluster
from metrics import evaluate_model
from model import EdgeInteractionGNN, SAGEGraphConvNet
from training import configure_optimizer, train, validate

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
boxsize = 680 / 0.6774
box_center = boxsize / 2

# %%
subhalos = pd.read_parquet('data/TNGCluster-subhalos_99_cuts_with_features.parquet')
subhalos = subhalos[(subhalos['subhalo_photo_K'] < -22) & (subhalos["subhalo_logsurfacedens"] > 8.5)]

H0 = cosmo.H0.value
dist_f_center_z = subhalos['subhalo_z'] - box_center
hubble_flow_velocity = H0 * dist_f_center_z
subhalos['subhalo_vz_total'] = subhalos['subhalo_vz'] - hubble_flow_velocity

# %%
# Define the objective function for Optuna
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_hidden = trial.suggest_int('n_hidden', 16, 128)
    n_latent = trial.suggest_int('n_latent', 16, 64)
    n_unshared_layers = trial.suggest_int('n_unshared_layers', 2, 10)
    D_link = trial.suggest_int('D_link', 1, 10)
    
    model_params = {
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'n_latent': n_latent,
        'n_unshared_layers': n_unshared_layers,
    }
    
    training_params = {
        'n_epochs': 150,  # Reduced for quicker Optuna trials
        'learning_rate': lr,
        'weight_decay': wd,
        'num_parts': 48,
        'augment': True,
        'early_stopping_patience': 10  # Number of epochs to wait before early stopping
    }

    # Recreate the graph with the new D_link value
    graph = create_graph(subhalos, D_link, boxsize, periodic=False,
                         x_columns=['subhalo_logstellarmass'],
                         y_columns=['subhalo_loghalomass'],
                         pos_columns=['subhalo_x', 'subhalo_y'],
                         vel_columns=['subhalo_vz_total'],
                        )

    train_idx, valid_idx = get_train_valid_indicies_cluster(graph, 0, 4)
    train_loader = ClusterLoader(ClusterData(graph.subgraph(train_idx), num_parts=training_params['num_parts'], recursive=False, log=False), batch_size=1, shuffle=True)
    valid_loader = ClusterLoader(ClusterData(graph.subgraph(valid_idx), num_parts=training_params['num_parts'] // 2, recursive=False, log=False), batch_size=1, shuffle=False)
    
    model = EdgeInteractionGNN(
        n_layers=model_params['n_layers'],
        node_features=graph.x_hydro.shape[1],
        edge_features=graph.edge_attr.shape[1],
        hidden_channels=model_params['n_hidden'],
        latent_channels=model_params['n_latent'],
        n_unshared_layers=model_params['n_unshared_layers'],
        n_out=graph.y.shape[1],
        aggr="max"
    )
    model.to(device)

    optimizer = configure_optimizer(model, lr, wd)
    scheduler = None

    best_val_rmse = float('inf')
    patience_counter = 0

    train_losses = []
    valid_losses = []
    
    for epoch in tqdm(range(training_params['n_epochs']), desc=f'Trial {trial.number}'):
        train_loss = train(train_loader, model, optimizer, scheduler, device, augment=training_params['augment'])
        valid_loss, p, y, subhalo_ids, current_rmse = validate(valid_loader, model, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if current_rmse < best_val_rmse:
            best_val_rmse = current_rmse
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= training_params['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_val_rmse, train_losses, valid_losses


# %%
# Set up Optuna with SQLite storage for parallel execution
study_name = 'gnn_hyperparameter_optimization'
storage_name = f'sqlite:///optuna_{study_name}.db'
n_trials=50

# Create a study and optimize the objective function
study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', load_if_exists=True)

# Manually update the progress bar based on the number of completed trials
completed_trials = len(study.trials)
with tqdm(total=n_trials, initial=completed_trials, desc='Optimization Progress') as pbar:
    def objective_with_progress(trial):
        result, train_losses, valid_losses = objective(trial)
        trial.set_user_attr('train_losses', train_losses)
        trial.set_user_attr('valid_losses', valid_losses)
        pbar.update(1)
        return result

    study.optimize(objective_with_progress, n_trials=n_trials - completed_trials, n_jobs=1)

print(f'Best trial: {study.best_trial.value}')
print(f'Best hyperparameters: {study.best_trial.params}')

# %%
# Plot training and validation loss for the best trial
# best_trial = study.best_trial
# train_losses = best_trial.user_attrs['train_losses']
# valid_losses = best_trial.user_attrs['valid_losses']

# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(valid_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
