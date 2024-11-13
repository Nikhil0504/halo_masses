import numpy as np
import pandas as pd

try:
    import illustris_python as il
except ImportError:
    raise ImportError("You need to install the illustris_python package. See https://github.com/illustristng/illustris_python.")

h = 0.6774
snapshot = 99

ROOT = '..'
base_path = f"{ROOT}/illustris_data/TNG300-1/output"

cuts = {
    "minimum_log_stellar_mass": 9,
    "minimum_log_halo_mass": 11,
    "minimum_n_star_particles": 50
}

subhalo_fields = ["SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
                  "SubhaloVel", "SubhaloVmax", "SubhaloFlag", "SubhaloGrNr", "SubhaloStellarPhotometrics"]
halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]

subhalos = il.groupcat.loadSubhalos(base_path, snapshot, fields=subhalo_fields)
halos = il.groupcat.loadHalos(base_path, snapshot, fields=halo_fields)

subhalo_pos = subhalos["SubhaloPos"][:] / (h*1e3)
subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
subhalo_halomass = subhalos["SubhaloMassType"][:,1]
subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4] 
subhalo_vel = subhalos["SubhaloVel"][:] 
subhalo_vmax = subhalos["SubhaloVmax"][:]
subhalo_flag = subhalos["SubhaloFlag"][:]
subhalo_photometry = subhalos["SubhaloStellarPhotometrics"][:]
halo_id = subhalos["SubhaloGrNr"][:].astype(int)

halo_mass = halos["Group_M_Crit200"][:]
halo_primarysubhalo = halos["GroupFirstSub"][:].astype(int)
group_pos = halos["GroupPos"][:] / (h*1e3)
group_vel = halos["GroupVel"][:] 

halos = pd.DataFrame(
    np.column_stack(
        (np.arange(len(halo_mass)), group_pos, group_vel, 
         halo_mass, halo_primarysubhalo)
            ),
    columns=['halo_id', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 
             'halo_vz', 'halo_mass', 'halo_primarysubhalo']
)
halos['halo_id'] = halos['halo_id'].astype(int)
halos.set_index("halo_id", inplace=True)

subhalos = pd.DataFrame(
    np.column_stack(
        [halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, 
         subhalo_vel, subhalo_n_stellar_particles, subhalo_stellarmass, subhalo_halomass, 
         subhalo_stellarhalfmassradius, subhalo_vmax, subhalo_photometry]
        ),
        columns=['halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 
                 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 
                 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_stellarmass', 
                 'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax', 
                 'subhalo_photo_U', "subhalo_photo_B", "subhalo_photo_V", 
                 "subhalo_photo_K", "subhalo_photo_g", "subhalo_photo_r", 
                 "subhalo_photo_i", "subhalo_photo_z"],
    )

subhalos["is_central"] = (halos.loc[subhalos.halo_id]["halo_primarysubhalo"].values == subhalos["subhalo_id"].values)


subhalos = subhalos[subhalos["subhalo_flag"] != 0].copy()
subhalos['halo_id'] = subhalos['halo_id'].astype(int)
subhalos['subhalo_id'] = subhalos['subhalo_id'].astype(int)

subhalos["subhalo_logstellarmass"] = np.log10(subhalos["subhalo_stellarmass"] / h)+10
subhalos["subhalo_loghalomass"] = np.log10(subhalos["subhalo_halomass"] / h)+10
subhalos["subhalo_logvmax"] = np.log10(subhalos["subhalo_vmax"])
subhalos["subhalo_logstellarhalfmassradius"] = np.log10(subhalos["subhalo_stellarhalfmassradius"])

subhalos.drop("subhalo_flag", axis=1, inplace=True)
subhalos = subhalos[subhalos["subhalo_loghalomass"] > cuts["minimum_log_halo_mass"]].copy()
# stellar mass and particle cuts
subhalos = subhalos[subhalos["subhalo_n_stellar_particles"] > cuts["minimum_n_star_particles"]].copy()
subhalos = subhalos[subhalos["subhalo_logstellarmass"] > cuts["minimum_log_stellar_mass"]].copy()

subhalos.to_parquet(f'{base_path}/TNG300-1-subhalos_{snapshot}.parquet')
halos.to_parquet(f'{base_path}/TNG300-1-halos_{snapshot}.parquet')

