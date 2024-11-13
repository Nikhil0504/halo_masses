import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo
from sklearn.ensemble import RandomForestRegressor

from src.cosmic_graph import create_graph, get_train_valid_indicies_cluster
from plotting import plot_true_vs_pred
from metrics import *

boxsize = 680/0.6774
box_center = boxsize / 2
H0 = cosmo.H0.value
h = H0 / 100

K = 4

# Load your data
df = pd.read_parquet('data/TNGCluster-subhalos_99_new.parquet')

dist_f_center_z = df['subhalo_z'] - box_center
hubble_flow_velocity = H0 * dist_f_center_z
df['subhalo_vz_total'] = (df['subhalo_vz'] - hubble_flow_velocity) / hubble_flow_velocity
df['subhalo_logstellarhalfmassradius'] = np.log10(df['subhalo_stellarhalfmassradius'])


data_graph = create_graph(df, 3, 680/0.6774, periodic=True, pos_columns=["subhalo_x", "subhalo_y"],
                        vel_columns=["subhalo_vz_total"])


train_idx, valid_idx = get_train_valid_indicies_cluster(data_graph, 3, K=4)
train = data_graph.subgraph(train_idx)
valid = data_graph.subgraph(valid_idx)

metrics = {}

with_overdensity = True

for i in range(K):
    print(f"Fold {i}")
    train_idx, valid_idx = get_train_valid_indicies_cluster(data_graph, i, K=K)

    train = data_graph.subgraph(train_idx)
    valid = data_graph.subgraph(valid_idx)

    if with_overdensity:
        # get the indicies of the overdensity that are not nan or inf
        train_overdensity = train.overdensity
        valid_overdensity = valid.overdensity

        train_idx = np.isfinite(train_overdensity)
        valid_idx = np.isfinite(valid_overdensity)

        # X should be the hydro features and the overdensity
        X_train, X_test = np.concatenate([train.x_hydro[train_idx], train_overdensity[train_idx][:, None]], axis=1), np.concatenate([valid.x_hydro[valid_idx], valid_overdensity[valid_idx][:, None]], axis=1)
        y_train, y_test = train.y[train_idx], valid.y[valid_idx]

    else:
        X_train, X_test = train.x_hydro, valid.x_hydro
        y_train, y_test = train.y, valid.y

    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.ravel())
    y_pred_rf = rf.predict(X_test)

    # save the model

    if with_overdensity:
        # plot the y vs p
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        plot_true_vs_pred(y_test.ravel(), y_pred_rf, valid.dist_from_center[valid_idx], ax=ax, fig=fig, title=f"Random Forest Fold {i} Overdensity", clabel="Distance from Center [Mpc/h]", cmap='plasma')

        fig.savefig(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_overdensity.png")

        joblib.dump(rf, f"models/TNG-Cluster_snap99_random_forest_fold_{i}_overdensity.joblib")
        # save y, p, dist_from_center, valid_idx
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_overdensity_y.npy", y_test.ravel())
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_overdensity_p.npy", y_pred_rf)
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_overdensity_dist.npy", valid.dist_from_center[valid_idx])
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_overdensity_idx.npy", valid_idx)
    else:
        # plot the y vs p
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        plot_true_vs_pred(y_test.ravel(), y_pred_rf, valid.dist_from_center, ax=ax, fig=fig, title=f"Random Forest Fold {i}", clabel="Distance from Center [Mpc/h]", cmap='plasma')

        fig.savefig(f"results/TNG-Cluster_snap99_random_forest_fold_{i}.png")

        joblib.dump(rf, f"models/TNG-Cluster_snap99_random_forest_fold_{i}.joblib")
        # save y, p, dist_from_center, valid_idx
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_y.npy", y_test.ravel())
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_p.npy", y_pred_rf)
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_dist.npy", valid.dist_from_center)
        np.save(f"results/TNG-Cluster_snap99_random_forest_fold_{i}_idx.npy", valid_idx)

    # Evaluate models
    metrics_rf = evaluate_model(y_test.ravel().numpy(), y_pred_rf)

    key = f"Random Forest Fold {i}"
    metrics[key] = metrics_rf

results_df = pd.DataFrame(metrics).T

# add mean +/- std
results_df.loc["Random Forest Mean"] = results_df.mean()
results_df.loc["Random Forest Std"] = results_df.std()

results_df.round(3).to_csv("results/TNG-Cluster_snap99_random_forest_metrics.csv")


