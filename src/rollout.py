import torch
from torch.utils.data import DataLoader
from module import GlobalForecastModule
from data_module import GlobalForecastDataModule, collate_fn
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os

max_steps_map = {
    2: 8,
    3: 12,
    4: 16,
    5: 20,
    6: 24,
    7: 28,
    8: 32,
    9: 36,
    10: 40
}

"""
load a trained lightning module
"""
def load_model_and_data(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # model 
    model_hparams = state["hyper_parameters"]
    new_model = GlobalForecastModule(net_type=model_hparams["net_type"], vars=model_hparams["vars"])
    new_model.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
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

    visualize_unroll_results(model.net_type, predict_range, data_module.hparams.variables, pred_snapshot_trajectory, snapshot_trajectory)

"""
visualize the result of unrolling
"""
def visualize_unroll_results(model_name, predict_range, variables, pred, target):
    assert len(target) == len(pred)
    n_snapshots = len(target)
    fig, axes = plt.subplots(n_snapshots, 2)
    fig.set_figheight(5 * n_snapshots)
    fig.set_figwidth(10)
    for i in range(n_snapshots):
        scalar_field, scalar_field_pred = target[i], pred[i] 
        # Plot the heatmap
        ax1 = axes[i,0]
        ax2 = axes[i,1]
        im = ax1.imshow(scalar_field, cmap='jet', interpolation='nearest')
        ax1.set_title(f'Temperature Heatmap target, step = {i}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        im2 = ax2.imshow(scalar_field_pred, cmap="jet", interpolation="nearest")
        ax1.set_title(f'Temperature Field Heatmap preidction, step = {i}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(f"figs/unroll/{model_name}_{predict_range}_{variables}", bbox_inches='tight')


def perform_rollout(df, args):
    multiple = args.target_range / args.base_range
    max_steps = int(max_steps_map[multiple])
    target_steps = int(max_steps / multiple)
    checkpoint_path_base = df.loc[df.predict_range == args.base_range].iloc[0]
    checkpoint_path_target = df.loc[df.predict_range == args.target_range].iloc[0]
    model_base, data_module_base = load_model_and_data(checkpoint_path_base.checkpoint)
    model_target, data_module_target = load_model_and_data(checkpoint_path_target.checkpoint)
    autoregressive_unroll(model=model_base, data_module=data_module_base, n_steps=max_steps)
    autoregressive_unroll(model=model_target, data_module=data_module_target, n_steps=target_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_range", "-b", type=int, default=72)
    parser.add_argument("--target_range", "-t", type=int, default=144)
    args = parser.parse_args()
    checkpoint_maps = pd.read_csv("checkpoint_maps.csv")
    fnos = checkpoint_maps.loc[checkpoint_maps.model == "fno"]
    sfnos = checkpoint_maps.loc[checkpoint_maps.model == "sfno"]
    perform_rollout(fnos, args)
    perform_rollout(sfnos, args)