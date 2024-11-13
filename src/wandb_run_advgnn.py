import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.cosmology import Planck15 as cosmo
from torch_geometric.loader import ClusterData, ClusterLoader

import wandb
from src.cosmic_graph import (create_graph, get_train_valid_indices_new,
                          get_train_valid_indicies_cluster)
from metrics import evaluate_model
from model import EdgeInteractionGNN
from training import configure_optimizer, train, validate
from plotting import plot_true_vs_pred

# random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(subhalos, mask, D_link, **kwargs):
    if mask is not None:
        subhalos = subhalos[mask]
    
    graph = create_graph(
        subhalos,
        D_link=D_link,
        **kwargs
    )

    return graph


def get_dataloader(graph, training_params, boxsize):
    if training_params['sim'] == 'T300':
        train_idx, valid_idx = get_train_valid_indices_new(graph, k=training_params['k_0'], K=training_params['K'], boxsize=boxsize)
    elif training_params['sim'] == 'TC':
        train_idx, valid_idx = get_train_valid_indicies_cluster(graph, k=training_params['k_0'], K=training_params['K'])

    num_parts = training_params['num_parts']
    
    train_data = ClusterData(graph.subgraph(train_idx), num_parts=num_parts, recursive=False, log=False)
    train_loader = ClusterLoader(train_data, batch_size=training_params['batch_size'], shuffle=True)

    valid_data = ClusterData(graph.subgraph(valid_idx), num_parts=num_parts//2, recursive=False, log=False)
    valid_loader = ClusterLoader(valid_data, batch_size=training_params['batch_size'], shuffle=False)

    return train_loader, valid_loader


def create_model(node_features, edge_features, out_features, model_params):
    model = EdgeInteractionGNN(
        n_layers=model_params["n_layers"],
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=model_params["n_hidden"],
        latent_channels=model_params["n_latent"],
        n_unshared_layers=model_params["n_unshared_layers"],
        n_out=out_features,
        aggr="max"
    )
    model.to(device)
    return model


def save_model(model, epoch, path="model"):
    torch.save(model.state_dict(), f"models/{path}_epoch{epoch}.pth")


def plot_and_save(train_losses, valid_losses, y, p, dists, run=''):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    plot_true_vs_pred(y, p, dists, ax=ax, fig=fig, title=f"True vs Predicted", clabel="Distance from Center [Mpc]", cmap='plasma')
    fig.savefig(f"figures/{run}_true_vs_pred.png")
    
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.ylim(-0.2, 0.1)
    plt.legend()
    plt.savefig(f"figures/{run}_loss_plot.png")


def main():
    pos_columns = ['subhalo_x', 'subhalo_y']
    vel_columns = ['subhalo_vz_total']
    
    subhalos = pd.read_parquet(path)
    
    if config.sim == 'T300':
        subhalos[["subhalo_x", "subhalo_y", "subhalo_z"]] -= subhalos[["subhalo_x", "subhalo_y", "subhalo_z"]].min(0)
        
        subhalos['U-B'] = subhalos['subhalo_photo_U'] - subhalos['subhalo_photo_B']
        subhalos['B-V'] = subhalos['subhalo_photo_B'] - subhalos['subhalo_photo_V']
        subhalos['g-r'] = subhalos['subhalo_photo_g'] - subhalos['subhalo_photo_r']
        subhalos['r-i'] = subhalos['subhalo_photo_r'] - subhalos['subhalo_photo_i']
        subhalos['i-z'] = subhalos['subhalo_photo_i'] - subhalos['subhalo_photo_z']
        subhalos['subhalo_cluster_id'] = subhalos.index.values
        
        mass_surf_dens = np.log10(10**subhalos['subhalo_logstellarmass'] / (np.pi * 10**subhalos['subhalo_logstellarhalfmassradius']**2))
        subhalos["subhalo_logsurfacedens"] = mass_surf_dens
        
        dist_mod = cosmo.distmod(0.3).value
        
        stellar_half_mass_radius_arcseconds = 10**subhalos['subhalo_logstellarhalfmassradius'] * cosmo.arcsec_per_kpc_comoving(0.3)
        area = np.pi * stellar_half_mass_radius_arcseconds**2
        
        apparent_mag = subhalos['subhalo_photo_K'] + dist_mod
        
        surface_brightness = apparent_mag + 2.5 * np.log10(area)
        subhalos['subhalo_surface_brightness'] = surface_brightness
        subhalos['distance_from_center'] = np.ones(len(subhalos))
    
    elif config.sim == 'TC':
        subhalos['subhalo_logstellarhalfmassradius'] = np.log10(subhalos['subhalo_stellarhalfmassradius'])
    
    # mask = None
    mask = (subhalos['subhalo_photo_K'] < -22)
    dist_f_center_z = subhalos['subhalo_z'] - box_center
    hubble_flow_velocity = H0 * dist_f_center_z
    subhalos['subhalo_vz_total'] = (subhalos['subhalo_vz'] - hubble_flow_velocity)

    graph = load_data(subhalos=subhalos, mask=mask, D_link=config.D_link, pos_columns=pos_columns, vel_columns=vel_columns, boxsize=boxsize, periodic=config['periodic'])
    train_loader, valid_loader = get_dataloader(graph, config, boxsize)

    node_features = graph.x_hydro.shape[1]
    edge_features = graph.edge_attr.shape[1]
    out_features = graph.y.shape[1]

    model = create_model(node_features, edge_features, out_features, config)
    optimizer = configure_optimizer(model, config["learning_rate"], config["weight_decay"])
    # scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15, threshold=1e-3)
    
    wandb.watch(model, log='all')
    
    train_losses, valid_losses, valid_rmse = [], [], []
    early_stopping_patience = 35
    min_delta = 0.001
    best_loss = float('inf')
    patience_counter = 0


    for epoch in range(config["epochs"]):
        train_loss = train(train_loader, model, optimizer, None, device, augment=config['augment'], max_grad_norm=None)
        valid_loss, p, y, subhalo_ids, current_rmse, dists  = validate(valid_loader, model, device)
        

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_rmse.append(valid_rmse)

        eval_metrics = evaluate_model(y.ravel(), p.flatten())
        wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, **eval_metrics})


        if epoch == 0 or epoch % 10 == 0 or patience_counter == 0:
            print(f"{epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {current_rmse: >10.6f}")

        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Stopping early due to lack of improvement in validation loss.")
            break
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        if epoch % 50 == 0:
            save_model(model, epoch, wandb.run.name)
    
    save_model(model, epoch, wandb.run.name)
    # save the subhalo ids, true and predicted values
    np.save(f"results/{wandb.run.name}_subhalo_ids.npy", subhalo_ids)
    np.save(f"results/{wandb.run.name}_y.npy", y)
    np.save(f"results/{wandb.run.name}_p.npy", p)
    np.save(f"results/{wandb.run.name}_dists.npy", dists)

    plot_and_save(train_losses, valid_losses, y, p, dists, run=wandb.run.name)


if __name__ == '__main__':
    # Simulation info and Cosmology
    z=0
    H0 = cosmo.H0.value
    h = H0 / 100
    
    # intialize wandb
    wandb.init(
        project='cosmic-graph', 
    )
    config = wandb.config

    if config.sim == 'TC':
        sim_name = 'TNG-Cluster'
        boxsize = 680 / h
        path = 'data/TNGCluster-subhalos_99_new.parquet'
    elif config.sim == 'T300':
        sim_name = 'TNG-300'
        boxsize = 205.0001 / h
        path = 'data/TNG300-1-subhalos_99.parquet'
    else:
        raise Exception("Sim should either be TC or T300")
    
    box_center = boxsize / 2
    
    wandb.run.name = f'{sim_name}_snap{config.snapshot}_{config.run_name}_k0-{config.k_0}_lr-{config.learning_rate}_bs-{config.batch_size}'

    main()