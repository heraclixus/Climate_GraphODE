"""
Utility functions 
"""
import os, time, yaml, glob
from tqdm import tqdm
import torch
import click
import numpy as np
import random 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import xarray as xr 


"""
https://github.com/microsoft/ClimaX/blob/main/src/climax/utils/data_utils.py
"""


NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]
DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    'NorthAmerica': { # 8x14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': { # 14x10
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe': { # 6x8
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia': { # 10, 14
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia': { # 10, 12
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia': { # 10x14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    },
    'Global': { # 32, 64
        'lat_range': (-90, 90),
        'lon_range': (0, 360)
    }
}

def get_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region]
    lat_range = region['lat_range']
    lon_range = region['lon_range']
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w
    }
HOURS_PER_YEAR = 8760  # 365-day year


"""
obtain training and testing data loader for the spatiotemporal forecast task
"""
def get_data_loader(config_data, args): 
    dataset_path = config_data[args.dataset][args.subdataset]
    forecast_path = config_data[args.dataset]["forecast_path"]
    obs_path =config_data[args.dataset]["obs_path"]
    climatology_path = config_data[args.dataset]["climatology_path"]

    forecast = xr.open_zarr(forecast_path)
    obs = xr.open_zarr(obs_path)
    climatology = xr.open_zarr(climatology_path)



"""
convert the weatherbench2 dataset nc file into numpy arrays
For the dataset of geospatial only: path = data/geopotential_5.625deg/ 
variables = geopotential
"""
def nc2np(path, variables, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}

    constants = xr.open_mfdataset(os.path.join(path, "constants.nc"), combine="by_coords", parallel=True)
    constant_fields = ["land_sea_mask", "orography", "lattitude"]
    constant_values = {}
    for f in constant_fields:
        constant_values[f] = np.expand_dims(constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)).repeat(
            HOURS_PER_YEAR, axis=0
        )
        if partition == "train":
            normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
            normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    for year in tqdm(years):
        np_vars = {}

        # constant variables
        for f in constant_fields:
            np_vars[f] = constant_values[f]

        # non-constant fields
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                np_vars[var] = ds[code].to_numpy()[:HOURS_PER_YEAR]

                if partition == "train":  # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

                clim_yearly = np_vars[var].mean(axis=0)
                if var not in climatology:
                    climatology[var] = [clim_yearly]
                else:
                    climatology[var].append(clim_yearly)

            else:  # multiple-level variables, only use a subset
                assert len(ds[code].shape) == 4
                all_levels = ds["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[:HOURS_PER_YEAR]

                    if partition == "train":  # compute mean and std of each var in each year
                        var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                        if var not in normalize_mean:
                            normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                            normalize_std[f"{var}_{level}"] = [var_std_yearly]
                        else:
                            normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                            normalize_std[f"{var}_{level}"].append(var_std_yearly)

                    clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                    if f"{var}_{level}" not in climatology:
                        climatology[f"{var}_{level}"] = [clim_yearly]
                    else:
                        climatology[f"{var}_{level}"].append(clim_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            if var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            if var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )



"""
convert nc to np only for the geopotential dataset
for this special case, there is no constant.nc 
"""
def nc2np_geopotential(path, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}

    for year in tqdm(years):
        np_vars = {}
        # non-constant fields
        var = "geopotential"
        ps = glob.glob(os.path.join(path, f"*{year}*.nc"))
        ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
        code = NAME_TO_VAR[var]

        if len(ds[code].shape) == 3:  # surface level variables
            ds[code] = ds[code].expand_dims("val", axis=1)
            # remove the last 24 hours if this year has 366 days
            np_vars[var] = ds[code].to_numpy()[:HOURS_PER_YEAR]

            if partition == "train":  # compute mean and std of each var in each year
                var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                if var not in normalize_mean:
                    normalize_mean[var] = [var_mean_yearly]
                    normalize_std[var] = [var_std_yearly]
                else:
                    normalize_mean[var].append(var_mean_yearly)
                    normalize_std[var].append(var_std_yearly)

            clim_yearly = np_vars[var].mean(axis=0)
            if var not in climatology:
                climatology[var] = [clim_yearly]
            else:
                climatology[var].append(clim_yearly)

        else:  # multiple-level variables, only use a subset
            assert len(ds[code].shape) == 4
            all_levels = ds["level"][:].to_numpy()
            all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
            for level in all_levels:
                ds_level = ds.sel(level=[level])
                level = int(level)
                # remove the last 24 hours if this year has 366 days
                np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[:HOURS_PER_YEAR]

                if partition == "train":  # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                        normalize_std[f"{var}_{level}"] = [var_std_yearly]
                    else:
                        normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                        normalize_std[f"{var}_{level}"].append(var_std_yearly)

                clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                if f"{var}_{level}" not in climatology:
                    climatology[f"{var}_{level}"] = [clim_yearly]
                else:
                        climatology[f"{var}_{level}"].append(clim_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
            normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            mean, std = normalize_mean[var], normalize_std[var]
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean.mean(axis=0)
            normalize_mean[var] = mean
            normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )





@click.command()
@click.option("--root_dir", type=str, default="../../data/geopotential_5.625deg")
@click.option("--save_dir", type=str, default="../../data/geopotential_6.526deg_np")
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "toa_incident_solar_radiation",
        "total_precipitation",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2016)
@click.option("--start_test_year", type=int, default=2017)
@click.option("--end_year", type=int, default=2018)
@click.option("--num_shards", type=int, default=8)
def main(
    root_dir,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    print(f"train_years = {train_years}")
    print(f"val_years = {val_years}")
    print(f"test_years = {test_years}")

    os.makedirs(save_dir, exist_ok=True)

    # nc2np(root_dir, variables, train_years, save_dir, "train", num_shards)
    # nc2np(root_dir, variables, val_years, save_dir, "val", num_shards)
    # nc2np(root_dir, variables, test_years, save_dir, "test", num_shards)

    nc2np_geopotential(root_dir, train_years, save_dir, "train", num_shards)
    nc2np_geopotential(root_dir, val_years, save_dir, "val", num_shards)
    nc2np_geopotential(root_dir, test_years, save_dir, "test", num_shards)

    # save lat and lon data
    ps = glob.glob(os.path.join(root_dir, f"*{train_years[0]}*.nc"))
    x = xr.open_mfdataset(ps[0], parallel=True)
    lat = x["lat"].to_numpy()
    lon = x["lon"].to_numpy()
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


if __name__ == "__main__":
    main()