import torch
from torch.utils.data import DataLoader
from module import GlobalForecastModule
from data_module import GlobalForecastDataModule
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import argparse
from rollout import get_de_normalize_transform, visualize_unroll_results, collate_fn, print_transform

def load_model_and_data(checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # model 
    model_hparams = state["hyper_parameters"]
    new_model = GlobalForecastModule(net_type=model_hparams["net_type"], vars=model_hparams["vars"], use_geometric_loss=True)
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
    return data_module, new_model


"""
train autoregressive based loss using the ar loss in SFNO Appendix 
"""
def train_autoregressive_steps(neural_operator, lr, data_train, 
                               n_steps_lst: list, lat, n_epochs: int, device: str, retain_graph: bool):
    
    neural_operator.train()
    for n_steps in n_steps_lst:
        optimizer = AdamW(params=neural_operator.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)
        print(f"autoregressive steps = {n_steps}")
        train_dataloader = DataLoader(data_train, batch_size=8)
        train_iterator = iter(train_dataloader)
        iteration_count = 0     
        for epoch in range(n_epochs): 
            ar_losses = []
            geo_losses = []
            for (idx, batch) in enumerate(train_dataloader):
                x, y, _, _, _ = batch
                x = x.to(device)
                y = y.to(device)
                
                if idx == 0: # first iteration
                    pred = x
                    ar_loss = 0
                # when autoregressive steps is reached, reset 
                if idx != 0 and idx % n_steps == 0:
                    
                    optimizer.zero_grad()
                    pred = x
                    ar_loss /= n_steps
                    if retain_graph:               
                        ar_loss.backward(retain_graph=True)
                    else:
                        ar_loss.backward()
                    ar_losses.append(ar_loss.item())
                    ar_loss = 0
                    optimizer.step()
                    
                # ar loss calculation: 
                loss_dict, pred = neural_operator(pred, y, lat)
                loss = loss_dict["loss"]
                geo_losses.append(loss.item())
                if not retain_graph:
                    pred.detach()
                ar_loss += loss
                # print(f'idx = {idx}, loss = {loss}, ar_loss = {ar_loss}')
            scheduler.step()
            print(f"step = {n_steps}, epoch {epoch} finished: average loss = {np.mean(np.array(geo_losses))}, average ar loss = {np.mean(np.array(ar_losses))}")

    print("finetune finished!")
    
    return neural_operator


"""
evaluate the result of autoregressive training. 
"""
def autoregressive_unroll(neural_operator, net_type, data_module: GlobalForecastDataModule, n_steps, eval_type, args):
    # obtain denormalize transform
    transform = get_de_normalize_transform(data_module)

    # setup neural operator    
    neural_operator.eval()

    # setup dataloader
    if eval_type == "test":
        dataset = data_module.data_test
    else:
        dataset = data_module.data_val
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
        next_x, next_y = next_x.to(args.device), next_y.to(args.device)
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
    import os
    savepath = f"results/unroll/{eval_type}/{net_type}"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    np.save(f"{savepath}/target_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", snapshot_trajectory)
    np.save(f"{savepath}/predict_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", pred_snapshot_trajectory)

    visualize_unroll_results(net_type, predict_range, data_module.hparams.variables, pred_snapshot_trajectory, snapshot_trajectory)
    
    print("finished training all steps, saving results...")
    torch.save(neural_operator.state_dict(), f"checkpoints/{net_type}_{args.n_steps_lst}_retain={args.retain_graph}.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", "-ckpt", type=str, default="checkpoints/sfno_72_0523.ckpt")
    parser.add_argument("--n_epochs", "-ne", type=int, default=10)
    parser.add_argument("--n_steps_lst", "-ns", type=list, default=[2,3,4,5,6])
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--retain_graph", action='store_true')
    args = parser.parse_args()
    # checkpoint_maps = pd.read_csv("checkpoint_maps.csv")
    # fnos = checkpoint_maps.loc[checkpoint_maps.model == "fno"]
    # sfnos = checkpoint_maps.loc[checkpoint_maps.model == "sfno"]

    data_module, model_module = load_model_and_data(args.checkpoint)
    neural_operator = model_module.net
    neural_operator.to(args.device)
    # temporary fix for batch size
    data_train, data_val, data_test = data_module.data_train, data_module.data_val, data_module.data_test
    lat, lon = data_module.get_lat_lon()
    
    print(args)

    # autoregressive training
    neural_operator = train_autoregressive_steps(neural_operator, args.learning_rate, data_train, 
                                                args.n_steps_lst, lat, args.n_epochs, args.device, args.retain_graph)

    # evaluation autoregressive
    for ar_step in args.n_steps_lst:
        print(f"unroll evaluation for step size = {ar_step}")
        autoregressive_unroll(neural_operator=neural_operator, net_type=model_module.net_type, data_module=data_module, n_steps=ar_step, eval_type="validation", args=args)
        autoregressive_unroll(neural_operator=neural_operator, net_type=model_module.net_type, data_module=data_module, n_steps=ar_step, eval_type="test", args=args)