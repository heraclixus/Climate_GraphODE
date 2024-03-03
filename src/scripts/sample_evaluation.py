import xarray as xr 
import matplotlib.pyplot as plt 
import argparse
import yaml
from weatherbench2 import config
from weatherbench2.metrics import MSE, ACC
from weatherbench2.regions import SliceRegion, ExtraTropicalRegion
from weatherbench2.evaluation import evaluate_in_memory



"""
to use weather bench2, have to use a lower version of python 
due to the incompatibility with apache_beam
"""


# Subsample along the time dimension
def subsample(dataset, n_timestep, attribute):
    length = len(dataset['time']) // (n_timestep-1) 
    subsampled_dataset = dataset.isel(time=slice(0, None, length))
    attribute_ts = subsampled_dataset[attribute]
    return attribute_ts.to_numpy()



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="weatherbench2")
    argparser.add_argument("--subdataset", type=str, default="forecast_path")
    args = argparser.parse_args()


    with open("../configs/dataset.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    dataset_path = config_data[args.dataset][args.subdataset]
    # n_timesteps = config_data[args.dataset]["n_timesteps"]
    # attribute = config_data[args.dataset]["attribute"]


    forecast_path = config_data[args.dataset]["forecast_path"]
    obs_path =config_data[args.dataset]["obs_path"]
    climatology_path = config_data[args.dataset]["climatology_path"]

    forecast = xr.open_zarr(forecast_path)
    obs = xr.open_zarr(obs_path)
    climatology = xr.open_zarr(climatology_path)

    paths = config.Paths(
        forecast=forecast_path,
        obs=obs_path,
        output_dir='./',   # Directory to save evaluation results
    )

    selection = config.Selection(
        variables=[
            'geopotential',
            '2m_temperature'
        ],
        levels=[500, 700, 850],
        time_slice=slice('2020-01-01', '2020-12-31'),
    )

    data_config = config.Data(selection=selection, paths=paths)

    eval_configs = {
        'deterministic': config.Eval(
            metrics={
                'mse': MSE(), 
                'acc': ACC(climatology=climatology) 
            },
        )
    }


    regions = {
        'global': SliceRegion(),
        'tropics': SliceRegion(lat_slice=slice(-20, 20)),
        'extra-tropics': ExtraTropicalRegion(),
    }

    eval_configs = {
        'deterministic': config.Eval(
            metrics={
                'mse': MSE(), 
                'acc': ACC(climatology=climatology) 
            },
            regions=regions
        )
    }


    evaluate_in_memory(data_config, eval_configs)   # Takes around 5 minutes
    results = xr.open_dataset('./deterministic.nc')
    print(results)