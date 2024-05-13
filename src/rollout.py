import torch
from torch.utils.data import DataLoader
from module import GlobalForecastModule
from data_module import GlobalForecastDataModule, collate_fn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


"""
load a trained lightning module
"""
def load_model_and_data(checkpoint_path):
    state = torch.load(checkpoint_path)
    # model 
    model_hparams = state["hyper_parameters"]
    new_model = GlobalForecastModule(net_type=model_hparams["net_type"], vars=model_hparams["vars"])
    new_model.load_from_checkpoint(checkpoint_path)
    # data
    config_data = state["datamodule_hyper_parameters"]
    data_module = GlobalForecastDataModule(
        root_dir = config_data["root_dir"],
        variables = config_data["variables"],
        out_variables = config_data["out_variables"],
        buffer_size= config_data["buffer_size"],
        predict_range=config_data["predict_range"]
    )
    data_module.setup()
    return new_model, data_module


"""
for printing and other utilities
"""
def print_transform(x):
    return x.detach().squeeze(0).numpy()



def get_de_normalize_transform(data_module):
    var = data_module.hparams.variables[0]
    normalize_mean = dict(np.load(os.path.join(data_module.hparams.root_dir, "normalize_mean.npz")))[var]
    normalize_std = dict(np.load(os.path.join(data_module.hparams.root_dir, "normalize_std.npz")))[var]
    transform = lambda x: (x * normalize_std) + normalize_mean
    return transform

"""
unroll a trained neural operator to n_steps.
each step is single snapshot unrolling.
a dummy y is introduced for the calling signature of the forward() method
the actual temperature requires denormalization.
"""
def autoregressive_unroll(model:GlobalForecastModule, data_module: GlobalForecastDataModule, n_steps):
    neural_operator = model.net

    # obtain denormalize transform
    transform = get_de_normalize_transform(data_module)

    # setup neural operator    
    neural_operator.eval()
    neural_operator.to("cpu")

    # setup dataloader 
    dataset = data_module.data_test
    lat, lon = data_module.get_lat_lon()
    unroll_dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)
    it = iter(unroll_dataloader)
    step_count = 0
    predict_range = data_module.hparams.predict_range
    print(f"predict_range for current model = {predict_range}")
    pred_snapshot_trajectory = []
    snapshot_trajectory = []
    total_steps = n_steps * predict_range

    while (step_count < total_steps):
        next_x, next_y, _, _, _ = next(it)
        if step_count % predict_range == 0:
            if step_count == 0:
                pred = next_x
                pred_snapshot_trajectory.append(transform(print_transform(pred)))
            _, pred = neural_operator(pred, next_y, lat)    
            next_x, next_y = transform(print_transform(next_x)), transform(print_transform(next_y))
            pred_snapshot_trajectory.append(transform(print_transform(pred)))
            snapshot_trajectory.append(next_x)
        step_count += 1
    
    # remove the last one to match length
    pred_snapshot_trajectory = np.concatenate([pred_snapshot_trajectory[:-1]]).squeeze()
    snapshot_trajectory = np.concatenate([snapshot_trajectory]).squeeze()
    print(pred_snapshot_trajectory.shape)
    print(snapshot_trajectory.shape)
    np.save(f"results/unroll/{model.net_type}/target_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", snapshot_trajectory)
    np.save(f"results/unroll/{model.net_type}/predict_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", pred_snapshot_trajectory)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ckpt", "--checkpoint_path", type=str, required=True)
    parser.add_argument("-n", "--unroll_steps", type=int, default=1)
    args = parser.parse_args()
    neural_operator, data_module = load_model_and_data(args.checkpoint_path)
    autoregressive_unroll(model=neural_operator, data_module=data_module, n_steps=args.unroll_steps)