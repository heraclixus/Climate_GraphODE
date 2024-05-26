import torch
from torch.utils.data import DataLoader
from module import GlobalForecastModule
from data_module import GlobalForecastDataModule
import numpy as np
from torch.optim import AdamW
import pandas as pd
import argparse

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


def train_autoregressive_steps(neural_operator, optimizer, train_loader: DataLoader, val_loader: DataLoader, 
                               n_steps: int, lat, n_epochs: int, device: str):
    train_iterator = iter(train_loader)
    iteration_count = 0     
    for epoch in range(n_epochs): 
        ar_losses = []
        geo_losses = []
        print(f"epoch = {epoch}")
        for (idx, batch) in enumerate(train_loader):
            x, y, lead_times, _, _ = batch
            x.to(device)
            y.to(device)
            
            if idx == 0: # first iteration
                pred = x
                ar_loss = 0
            # when autoregressive steps is reached, reset 
            if idx != 0 and idx % args.n_steps == 0:
                pred = x
                ar_loss.backward()
                ar_losses.append(ar_loss.item())
                ar_loss = 0
                
            # ar loss calculation: 
            # ar loss seems to be: intermediate results need to call .detach()
            # otherwise, can be backward(retain_graph=True)
            loss_dict, pred = neural_operator(pred, y, lat)
            loss = loss_dict["loss"]
            geo_losses.append(loss.item())
            pred.detach()
            ar_loss += loss
            print(f'idx = {idx}, loss = {loss}, ar_loss = {ar_loss}')
    
        print(f"epoch {epoch} finished: average loss = {np.mean(np.array(geo_losses))}, average ar loss = {np.mean(np.array(ar_losses))}")

    torch.save(neural_operator, "checkpoints/sfno_finetuned1.ckpt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint", "-ckpt", type=str, default="checkpoints/sfno_72_0523.ckpt")
    parser.add_argument("--n_epochs", "-ne", type=int, default=5)
    parser.add_argument("--n_steps", "-ns", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:7")
    args = parser.parse_args()
    checkpoint_maps = pd.read_csv("checkpoint_maps.csv")
    fnos = checkpoint_maps.loc[checkpoint_maps.model == "fno"]
    sfnos = checkpoint_maps.loc[checkpoint_maps.model == "sfno"]

    data_module, model_module = load_model_and_data(args.checkpoint)
    neural_operator = model_module.net
    train_dataloader = data_module.train_dataloader()
    validation_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    lat, lon = data_module.get_lat_lon()
    
    print(f"loaded module, net = {neural_operator}")
    optimizer = AdamW(params=neural_operator.parameters(), lr=1e-5)
    
    # autoregressive training
    neural_operator = train_autoregressive_steps(neural_operator, optimizer, train_dataloader, validation_dataloader, 
                                                 args.n_steps, lat, args.n_epochs, args.device)