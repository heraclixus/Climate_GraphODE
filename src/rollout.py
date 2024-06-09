import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from module import GlobalForecastModule
from module_refiner import RefinerForecastModule
from data_module import GlobalForecastDataModule, collate_fn
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os


"""
load a trained lightning module
"""
def load_model_and_data(checkpoint_path, refiner=False):
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # model 
    model_hparams = state["hyper_parameters"]
    print(model_hparams["net_type"])

    if not refiner:
        new_model = GlobalForecastModule(net_type=model_hparams["net_type"], vars=model_hparams["vars"], use_geometric_loss=True)
    else:
        new_model = RefinerForecastModule(use_geometric_loss=True, net_type=model_hparams["net_type"], vars=model_hparams["vars"])
    
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
    return x.detach().cpu().squeeze(0).numpy()

def get_de_normalize_transform(data_module):
    normalization = data_module.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    return lambda x: (x-mean_denorm) / std_denorm
    # return transforms.Normalize(mean_denorm, std_denorm)
 
def get_de_normalize_transform_tensor(data_module):
    normalization = data_module.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    return lambda x: (x- torch.tensor(mean_denorm, device=x.device)) / torch.tensor(std_denorm, device=x.device)

"""
unroll a trained neural operator to n_steps.
each step is single snapshot unrolling.
a dummy y is introduced for the calling signature of the forward() method
the actual temperature requires denormalization.
"""
def autoregressive_unroll(model, data_module: GlobalForecastDataModule, n_steps, predict_range, args):
    # neural_operator = model.net
    if args.finetuned:
        model.net.load_state_dict(torch.load(args.checkpoint_no))
    # obtain denormalize transform
    transform = get_de_normalize_transform(data_module)
    # setup neural operator    
    model.net.eval()
    model.net.to("cpu")
    # setup dataloader 
    dataset = data_module.data_test
    lat, lon = data_module.get_lat_lon()
    model.set_lat_lon(lat, lon)
    unroll_dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)
    it = iter(unroll_dataloader)
    step_count = 0
    predict_range = data_module.hparams.predict_range
    print(f"predict_range for current model = {predict_range}")
    pred_snapshot_trajectory = []
    snapshot_trajectory = []
    total_steps = n_steps * predict_range

    while (step_count <= total_steps):
        next_x, next_y, _, _, _ = next(it)
        if step_count % predict_range == 0: # step through predict range since dataset sampled at 1hr window
            if step_count == 0:
                pred = next_x
                pred_snapshot_trajectory.append(transform(print_transform(pred)))
            _, pred = model(pred, next_y)    
            next_x = transform(print_transform(next_x))
            pred_snapshot_trajectory.append(transform(print_transform(pred)))
            snapshot_trajectory.append(next_x)
        step_count += 1
    
    # remove the last one to match length
    pred_snapshot_trajectory = np.concatenate([pred_snapshot_trajectory[:-1]]).squeeze()
    snapshot_trajectory = np.concatenate([snapshot_trajectory]).squeeze()
    print(pred_snapshot_trajectory.shape)
    print(snapshot_trajectory.shape)

    if args.refiner == False:
        np.save(f"results/unroll/{model.net_type}/target_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", snapshot_trajectory)
        np.save(f"results/unroll/{model.net_type}/predict_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", pred_snapshot_trajectory)

    else:
        np.save(f"results/unroll/refiner/{model.net_type}/target_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", snapshot_trajectory)
        np.save(f"results/unroll/refiner/{model.net_type}/predict_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", pred_snapshot_trajectory)


    visualize_unroll_results(model.net_type, predict_range, data_module.hparams.variables, pred_snapshot_trajectory, snapshot_trajectory, n_steps, "test", args.refiner, args.finetuned)

"""
visualize the result of unrolling
"""
def visualize_unroll_results(model_name, predict_range, variables, pred, target, n_steps, eval_type, refiner, finetuned):
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
        ax1.set_title(f'Temperature Heatmap target, step = {i},  {int(i * predict_range / 24)} days', loc="center")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        im2 = ax2.imshow(scalar_field_pred, cmap="jet", interpolation="nearest")
        ax1.set_title(f'Temperature Field Heatmap preidction, step = {i}, {int(i * predict_range / 24)} days', loc="center")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    if not refiner:
        if finetuned:
            print(f"saving figure figs/unroll/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}_finetuned")
            plt.savefig(f"figs/unroll/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}_finetuned", bbox_inches='tight')
        else:
            print(f"saving figure figs/unroll/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}")
            plt.savefig(f"figs/unroll/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}", bbox_inches='tight')
    else:
        if finetuned:
            print(f"saving figure figs/unroll/refiner/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}_finetuned")
            plt.savefig(f"figs/unroll/refiner/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}_finetuned", bbox_inches='tight')
        else:        
            print(f"saving figure figs/unroll/refiner/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}")
            plt.savefig(f"figs/unroll/refiner/{model_name}_{predict_range}_{n_steps}_{variables}_{eval_type}", bbox_inches='tight')


"""
modify so more specific about loading of target and source models.
"""
def perform_rollout(base_model_checkpoint, target_model_checkpoint, args):
    multiple = args.target_range // args.base_range
    target_steps = 2 # target range is the bigger range, only unroll once
    max_steps = int(target_steps * multiple)
    model_base, data_module_base = load_model_and_data(base_model_checkpoint, args.refiner)
    model_target, data_module_target = load_model_and_data(target_model_checkpoint, args.refiner)
    autoregressive_unroll(model=model_base, data_module=data_module_base, n_steps=max_steps, predict_range=args.base_range, args=args)
    autoregressive_unroll(model=model_target, data_module=data_module_target, n_steps=target_steps, predict_range=args.target_range, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_range", "-b", type=int, default=72)
    parser.add_argument("--target_range", "-t", type=int, default=144)
    parser.add_argument("--net_type", "-n", type=str, default="fno")
    parser.add_argument("--checkpoint_no", type=str, default="checkpoints/")
    parser.add_argument("--refiner", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    args = parser.parse_args()
    checkpoint_maps = pd.read_csv("checkpoint_maps.csv")
    net_checkpoints = checkpoint_maps.loc[checkpoint_maps.model == args.net_type]
    base_model_checkpoint = net_checkpoints.loc[(net_checkpoints.predict_range == args.base_range) & (net_checkpoints.refined == args.refiner)].iloc[0]["checkpoint"]
    target_model_checkpoint = net_checkpoints.loc[(net_checkpoints.predict_range == args.target_range) & (net_checkpoints.refined == False)].iloc[0]["checkpoint"]
    print(args)
    print(f"base checkpoint = {base_model_checkpoint}")
    print(f"target = {target_model_checkpoint}")
    perform_rollout(base_model_checkpoint, target_model_checkpoint, args)