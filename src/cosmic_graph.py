import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add

np.random.seed(0)

def create_graph(
        subhalos,
        D_link,
        boxsize=205.0001/0.6774,
        periodic=True,
        x_columns = ["subhalo_logstellarmass"],
        y_columns = ["subhalo_loghalomass"],
        pos_columns = ['subhalo_x', 'subhalo_y', 'subhalo_z'],
        vel_columns = ['subhalo_vx', 'subhalo_vy', 'subhalo_vz'],
        is_central_column = ['is_central'],
        halfmassradius_column = ['subhalo_logstellarhalfmassradius']
    ):
    df = subhalos.copy()

    subhalo_id = torch.tensor(df.index.values, dtype=torch.long)

    df.reset_index(drop=True)

    x_hydro = torch.tensor(df[x_columns].values, dtype=torch.float)
    y = torch.tensor(df[y_columns].values, dtype=torch.float)


    pos_hydro = torch.tensor(df[pos_columns].values, dtype=torch.float)
    vel_hydro = torch.tensor(df[vel_columns].values, dtype=torch.float)

    is_central = torch.tensor(df[is_central_column].values, dtype=torch.int)
    halfmassradius = torch.tensor(df[halfmassradius_column].values, dtype=torch.float)

    cluster_id = torch.tensor(df["subhalo_cluster_id"].values, dtype=torch.long)

    # make the links
    tree = cKDTree(pos_hydro, leafsize=25, boxsize=boxsize)
    edge_index = tree.query_pairs(r=D_link, output_type='ndarray').astype('int')

    # normalize the positions
    df[pos_columns] = df[pos_columns] / (boxsize/2)

    # reverse the pairs so that the first element is always the closest to the second
    edge_index = to_undirected(torch.tensor(edge_index).t().contiguous().type(torch.long))

    # Write in pytorch-geometric format
    edge_index = edge_index.reshape((2,-1))
    num_pairs = edge_index.shape[1]

    # get the edge attributes
    row, col = edge_index
    diff = pos_hydro[row] - pos_hydro[col]
    dist = torch.linalg.norm(diff, axis=1)


    # make cuts but with periodic conditions
    # if periodic:
    #     for i, pos_i in enumerate(diff):
    #         for j, coord in enumerate(pos_i):
    #             if coord > D_link:
    #                 diff[i,j] -= boxsize  # Boxsize normalize to 1
    #             elif -coord > D_link:
    #                 diff[i,j] += boxsize  # Boxsize normalize to 1
    # Apply periodic boundary conditions efficiently
    if periodic:
        mask_pos = diff > D_link
        mask_neg = diff < -D_link
        diff[mask_pos] -= boxsize
        diff[mask_neg] += boxsize

    # define arbitrary coordinate, invarinat to translation/rotation shifts, but not stretches
    centroid = pos_hydro.mean(0)

    # unit vectors
    unit_row = (pos_hydro[row] - centroid) / torch.linalg.norm(pos_hydro[row] - centroid, dim=1, keepdim=True)
    unit_col = (pos_hydro[col] - centroid) / torch.linalg.norm(pos_hydro[col] - centroid, dim=1, keepdim=True)
    unit_diff = diff / dist.reshape(-1, 1)

    # dot products
    cos1 = torch.sum(unit_row * unit_col, dim=1)
    cos2 = torch.sum(unit_row * unit_diff, dim=1)

    # same features for velocity
    vel_diff = vel_hydro[row] - vel_hydro[col]
    vel_norm = torch.linalg.norm(vel_diff, axis=1)
    vel_centroid = vel_hydro.mean(0)

    vel_unit_row = (vel_hydro[row] - vel_centroid) / torch.linalg.norm(vel_hydro[row] - vel_centroid, dim=1, keepdim=True)
    vel_unit_col = (vel_hydro[col] - vel_centroid) / torch.linalg.norm(vel_hydro[col] - vel_centroid, dim=1, keepdim=True)
    vel_unit_diff = vel_diff / vel_norm.reshape(-1, 1)

    vel_cos1 = torch.sum(vel_unit_row * vel_unit_col, dim=1)
    vel_cos2 = torch.sum(vel_unit_row * vel_unit_diff, dim=1)

    # edge attributes
    edge_attr = torch.stack([dist, 
                             # cos1, 
                             # cos2, 
                             vel_norm, 
                             # vel_cos1, 
                             # vel_cos2,
                            ], 
                        dim=1)


    # optimized way to compute overdensity
    overdensity = torch.log10(
        scatter_add(
            10**x_hydro[row, 0], 
            index=col, 
            dim=0, 
            dim_size=len(x_hydro)
        )
    )

    num_nodes = x_hydro.shape[0]

    dist_from_center = torch.tensor(df['distance_from_center'].values, dtype=torch.float)

    data = Data(
        num_nodes=num_nodes,
        x_hydro=x_hydro,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        is_central=is_central,
        pos_hydro=pos_hydro,
        vel_hydro=vel_hydro,
        halfmassradius=halfmassradius,
        subhalo_id=subhalo_id,
        cluster_id=cluster_id,
        overdensity=overdensity,
        dist_from_center=dist_from_center,
    )

    return data


def get_train_valid_indices(data, k, K=3, boxsize=205/0.6774, pad=10, epsilon=1e-10):
    """k must be between `range(0, K)`. 

    `boxsize` and `pad` are both in units of Mpc, and it is assumed that the 
    `data` object has attribute `pos` of shape (N_rows, 3) also in units of Mpc.

    `epsilon` is there so that the modular division doesn't cause the boolean
    logic to wrap around.
    """

    # use x coordinate for train-valid split
    train_1_mask = (
        (data.pos_hydro[:, 0]  > ((k) / K * boxsize + pad) % boxsize) 
        & (data.pos_hydro[:, 0] <= ((k + 1) / K * boxsize - epsilon) % boxsize)
    )

    train_2_mask = (
        (data.pos_hydro[:, 0]  > ((k + 1)/ K * boxsize) % boxsize) 
        & (data.pos_hydro[:, 0] <= ((k + 2) / K * boxsize - pad) % boxsize)
    )

    valid_mask = (
        (data.pos_hydro[:, 0] > ((k + 2) / K * boxsize) % boxsize)
        & (data.pos_hydro[:, 0] <= ((k + 3) / K * boxsize - epsilon) % boxsize)
    )

    # this is the weird pytorch way of doing `np.argwhere`
    train_indices = (train_1_mask  | train_2_mask).nonzero(as_tuple=True)[0] 

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    # ensure zero overlap
    assert (set(train_indices) & set(valid_indices)) == set()

    return train_indices, valid_indices


def get_train_valid_indices_new(data, k, K=3, boxsize=205/0.6774, pad=10, epsilon=1e-10):
    """k must be between `range(0, K)`. 

    `boxsize` and `pad` are both in units of Mpc, and it is assumed that the 
    `data` object has attribute `pos` of shape (N_rows, 3) also in units of Mpc.

    `epsilon` is there so that the modular division doesn't cause the boolean
    logic to wrap around.
    """
    
    # Determine the size of each fold
    fold_size = boxsize / K

    # Initialize masks for training and validation
    train_mask = torch.zeros(data.pos_hydro.shape[0], dtype=torch.bool)
    valid_mask = torch.zeros(data.pos_hydro.shape[0], dtype=torch.bool)

    # Create the validation mask for the k-th fold
    valid_start = (k * fold_size + pad) % boxsize
    valid_end = ((k + 1) * fold_size - pad - epsilon) % boxsize

    if valid_start < valid_end:
        valid_mask |= (data.pos_hydro[:, 0] >= valid_start) & (data.pos_hydro[:, 0] < valid_end)
    else:
        valid_mask |= (data.pos_hydro[:, 0] >= valid_start) | (data.pos_hydro[:, 0] < valid_end)

    
    # Create the training masks for the remaining folds
    for i in range(K):
        if i == k:
            continue
        train_start = (i * fold_size) % boxsize
        train_end = ((i + 1) * fold_size - epsilon) % boxsize

        if train_start < train_end:
            train_mask |= (data.pos_hydro[:, 0] >= train_start) & (data.pos_hydro[:, 0] < train_end)
        else:
            train_mask |= (data.pos_hydro[:, 0] >= train_start) | (data.pos_hydro[:, 0] < train_end)
    

    # Ensure there is no overlap between training and validation sets
    assert (train_mask & valid_mask).sum().item() == 0

    # Get indices
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    return train_indices, valid_indices

# TNG-Cluster
def assign_kfold(cluster_ids, K=4):
    np.random.shuffle(cluster_ids)
    
    # Calculate the size of each fold
    fold_size = len(cluster_ids) // K
    remainder = len(cluster_ids) % K
    
    # Create a dictionary to map cluster IDs to their fold number
    validation_ks_dict = {}
    
    # Assign folds
    current = 0
    for i in range(K):
        if i < remainder:
            fold = cluster_ids[current:current + fold_size + 1]
            current += fold_size + 1
        else:
            fold = cluster_ids[current:current + fold_size]
            current += fold_size
        
        for cluster_id in fold:
            validation_ks_dict[cluster_id] = i
    
    # Convert the dictionary to an array with the same order as cluster_ids
    validation_ks = np.array([validation_ks_dict[cluster_id] for cluster_id in cluster_ids])

    return validation_ks


def get_train_valid_indicies_cluster(subhalos, k, K=4):
    subhalo_cluster_ids = subhalos.cluster_id.unique().numpy()
    
    # Assign K-folds
    validation_ks = assign_kfold(subhalo_cluster_ids, K)

    # Count occurrences of each fold
    unique_folds, counts = np.unique(validation_ks, return_counts=True)
    fold_counts = dict(zip(unique_folds, counts))
    
    print(f'Fold counts: {fold_counts}')
    # Sum up the counts of elements not equal to k and equal to k
    num_not_equal_k = sum(count for fold, count in fold_counts.items() if fold != k)
    num_equal_k = sum(count for fold, count in fold_counts.items() if fold == k)
    print(f'Sum of elements not equal to fold {k}: {num_not_equal_k}')
    print(f'Sum of elements equal to fold {k}: {num_equal_k}')
    
    # Create a dictionary mapping cluster IDs to their assigned fold
    fold_mapping = dict(zip(subhalo_cluster_ids, validation_ks))
    
    # Map fold assignments to the subhalos tensor
    # fold = subhalos.cluster_id.map(fold_mapping)
    fold = np.array([fold_mapping[cid.item()] for cid in subhalos.cluster_id])
    
    # Get train and validation indices
    train_indices = torch.nonzero(torch.tensor(fold != k)).squeeze()
    valid_indices = torch.nonzero(torch.tensor(fold == k)).squeeze()
    
    return train_indices, valid_indices


# simulate fiber collisions like SDSS
def fiber_collisions(x, y, collision_radius):
    from scipy.spatial import cKDTree

    # create a KDTree
    tree = cKDTree(np.array([x, y]).T)

    # find pairs of points that are closer than the collision radius
    pairs = tree.query_pairs(r=collision_radius)

    # remove the second point of each pair
    to_remove = [p[1] for p in pairs]
    x = np.delete(x, to_remove)
    y = np.delete(y, to_remove)

    return x, y