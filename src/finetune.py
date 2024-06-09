import torch
from torch.utils.data import DataLoader
from module import GlobalForecastModule
from module_refiner import RefinerForecastModule
from data_module import GlobalForecastDataModule
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from rollout import get_de_normalize_transform, visualize_unroll_results, collate_fn, print_transform, get_de_normalize_transform_tensor
from lib.metrics import lat_weighted_acc, lat_weighted_mse_val, lat_weighted_rmse

def load_model_and_data(checkpoint_path: str, refiner=False):
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # model 
    model_hparams = state["hyper_parameters"]
    if not refiner:
        new_model = GlobalForecastModule(net_type=model_hparams["net_type"], vars=model_hparams["vars"], use_geometric_loss=True)
    else:
        new_model = RefinerForecastModule(use_geometric_loss=True, net_type=model_hparams["net_type"], vars=model_hparams["vars"])
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
    lat, lon = data_module.get_lat_lon()
    climatology = data_module.get_climatology(partition="test")
    new_model.set_lat_lon(lat, lon)
    new_model.set_test_clim(climatology)
    return data_module, new_model


"""
train autoregressive based loss using the ar loss in SFNO Appendix 
"""
def train_autoregressive_steps(neural_operator, lrs, data_train, net_type,
                               n_steps_lst: list, lat, n_epochs_lst: list, device: str, retain_graph: bool):
    
    neural_operator.train()
    for i in range(len(n_steps_lst)):
        n_steps = n_steps_lst[i]
        n_epochs = n_epochs_lst[i]
        lr = lrs[i]
        optimizer = AdamW(params=neural_operator.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)
        print(f"autoregressive steps = {n_steps}, epoch = {n_epochs_lst[i]}")
        train_dataloader = DataLoader(data_train, batch_size=8)
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
            print(f"step = {n_steps}, epoch {epoch+1} finished: average loss = {np.mean(np.array(geo_losses))}, average ar loss = {np.mean(np.array(ar_losses))}")

    print("finetune finished!")
    torch.save(neural_operator.state_dict(), f"checkpoints_finetune/{net_type}_retain={args.retain_graph}.ckpt")    
    return neural_operator



"""
finetune for model trained using refiner
"""
def train_autoregressive_refiner(model_module: RefinerForecastModule, lrs, data_train, 
                                 n_steps_lst: list, n_epochs_lst: list, device: str, retain_graph: bool):
    model_module.train()
    model_module.to(device)
    for i in range(len(n_steps_lst)):
        n_steps = n_steps_lst[i]
        n_epochs = n_epochs_lst[i]
        lr = lrs[i]
        optimizer = AdamW(params=model_module.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)
        print(f"autoregressive steps = {n_steps}, epoch = {n_epochs_lst[i]}")
        train_dataloader = DataLoader(data_train, batch_size=8)
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
                    
                # ar loss calculation
                # modify the train_step from model_module to account for x,y
                # use pred and y 
                k = torch.randint(0, model_module.scheduler.config.num_train_timesteps, (x.shape[0],), device=device)      
                noise_factor = model_module.scheduler.alphas_cumprod.to(device)[k]
                noise_factor = noise_factor.view(-1, *[1 for _ in range(pred.ndim - 1)])
                signal_factor = 1 - noise_factor
                noise = torch.randn_like(y)
                y_noised = model_module.scheduler.add_noise(y, noise, k) 
                target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
                loss_dict, pred = model_module.net(torch.cat((x, y_noised), dim=1), target, lat=model_module.lat)
                loss = loss_dict["loss"]
                geo_losses.append(loss.item())
                if not retain_graph:
                    pred.detach()
                ar_loss += loss
                # print(f'idx = {idx}, loss = {loss}, ar_loss = {ar_loss}')
            scheduler.step()
            print(f"step = {n_steps}, epoch {epoch+1} finished: average loss = {np.mean(np.array(geo_losses))}, average ar loss = {np.mean(np.array(ar_losses))}")
    print(f"finetune finished! saving at checkpoints_finetune/{model_module.net_type}_retain={args.retain_graph}.ckpt")
    torch.save(model_module.net.state_dict(), f"checkpoints_finetune/{model_module.net_type}_retain={args.retain_graph}_net.ckpt") # save the operator 
    torch.save(model_module, f"checkpoints_finetune/{model_module.net_type}_retain={args.retain_graph}.ckpt")  # save the entire lightning module
    return model_module.net # return only the neural operator




"""
evaluate the result of autoregressive training with plots and metrics reportings. 
"""
def autoregressive_unroll(neural_operator, net_type, data_module: GlobalForecastDataModule, n_steps, eval_type, args):
    # obtain denormalize transform
    transform = get_de_normalize_transform(data_module)
    transform_ = get_de_normalize_transform_tensor(data_module)
    # setup neural operator    
    neural_operator.eval()

    # setup dataloader
    if eval_type == "test":
        dataset = data_module.data_test
    else:
        dataset = data_module.data_val
    lat, lon = data_module.get_lat_lon()
    test_clim = data_module.get_climatology(partition="test")
    
    unroll_dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)
    it = iter(unroll_dataloader)
    step_count = 0
    predict_range = data_module.hparams.predict_range
    print(f"predict_range for current model = {predict_range}")
    pred_snapshot_trajectory = []
    snapshot_trajectory = []
    total_steps = n_steps * predict_range

    # metrics
    lat_weighted_mses, lat_weighted_rmses, lat_weighted_accs = [],[],[]
    while (step_count < total_steps):
        next_x, next_y, _, _, out_variables = next(it)
        next_x, next_y = next_x.to(args.device), next_y.to(args.device)
        if step_count % predict_range == 0:
            if step_count == 0:
                pred = next_x
                pred_snapshot_trajectory.append(transform(print_transform(pred)))
            _, pred = neural_operator(pred, next_y, lat)
            
            loss_dicts_all = neural_operator.evaluate(pred, next_y, out_variables, transform_,
                                                      metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
                                                      lat=lat, clim=test_clim,
                                                      log_postfix = f"{int(data_module.hparams.predict_range/ 24)}_days")
            loss_dict1, loss_dict2, loss_dict3 = loss_dicts_all[0], loss_dicts_all[1], loss_dicts_all[2]
            lat_weighted_mses.append(loss_dict1["w_mse"])
            lat_weighted_rmses.append(loss_dict2["w_rmse"])
            lat_weighted_accs.append(loss_dict3["acc"])
            print(f"metrics1 = {lat_weighted_mses}")
            print(f"metrics2 = {lat_weighted_rmses}")
            print(f"metrics3 = {lat_weighted_accs}")            
            next_x, next_y = transform(print_transform(next_x)), transform(print_transform(next_y))
            pred_snapshot_trajectory.append(transform(print_transform(pred)))
            snapshot_trajectory.append(next_x)
        step_count += 1

    mse_avg = np.mean(np.array(lat_weighted_mses))
    rmse_avg = np.mean(np.array(lat_weighted_rmses))
    acc_avg = np.mean(np.array(lat_weighted_accs))
    print(f"unroll average mse = {mse_avg}, rmse = {rmse_avg}, acc = {acc_avg}")
    
    # remove the last one to match length
    pred_snapshot_trajectory = np.concatenate([pred_snapshot_trajectory[:-1]]).squeeze()
    snapshot_trajectory = np.concatenate([snapshot_trajectory]).squeeze()
    print(pred_snapshot_trajectory.shape)
    print(snapshot_trajectory.shape)
    import os
    savepath = f"results/unroll/{eval_type}/{net_type}" if not args.refiner else f"results/unroll/refiner/{eval_type}/{net_type}"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    np.save(f"{savepath}/target_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", snapshot_trajectory)
    np.save(f"{savepath}/predict_{data_module.hparams.variables}_range={predict_range}_steps={n_steps}.npy", pred_snapshot_trajectory)
    visualize_unroll_results(net_type, predict_range, data_module.hparams.variables, pred_snapshot_trajectory, snapshot_trajectory, eval_type=eval_type, n_steps=n_steps, refiner=args.refiner, finetuned=args.finetuned)



def generate_plots_only(n_steps, eval_type, net_type, refiner, finetuned):
    snapshot_trajectory = np.load(f"results/unroll/{eval_type}/{net_type}/target_['2m_temperature']_range=72_steps={n_steps}.npy")
    pred_snapshot_trajectory = np.load(f"results/unroll/{eval_type}/{net_type}/predict_['2m_temperature']_range=72_steps={n_steps}.npy")
    visualize_unroll_results(net_type, 72, "2m_temperature", pred_snapshot_trajectory, snapshot_trajectory, n_steps, eval_type, refiner=refiner, finetuned=finetuned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rates", "-lrs", type=float, default=[1e-4,1e-4,1e-5,1e-5])
    parser.add_argument("--checkpoint", "-ckpt", type=str, default="checkpoints/sfno_72_0523.ckpt")
    parser.add_argument("--n_epochs", "-ne", type=int, default=[30,10,5,5])
    parser.add_argument("--n_steps_lst", "-ns", type=list, default=[2,3,4,5])
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--retain_graph", action='store_true')
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--refiner", action="store_true")
    args = parser.parse_args()
    # checkpoint_maps = pd.read_csv("checkpoint_maps.csv")
    # fnos = checkpoint_maps.loc[checkpoint_maps.model == "fno"]
    # sfnos = checkpoint_maps.loc[checkpoint_maps.model == "sfno"]

    if args.plot_only:
        for n_steps in args.n_steps_lst:
            generate_plots_only(n_steps, "validation", "sfno", args.refiner)
            generate_plots_only(n_steps, "test", "sfno", args.refiner)
        exit(0)

    data_module, model_module = load_model_and_data(args.checkpoint, args.refiner)
    
    # temporary fix for batch size
    data_train, data_val, data_test = data_module.data_train, data_module.data_val, data_module.data_test
    lat, lon = data_module.get_lat_lon()
    
    print(args)
    # autoregressive training
    # change: instead of neural_operator, train the entire lightning module
    #         since refiner has its own forward pass

    if args.refiner:
        neural_operator = train_autoregressive_refiner(model_module, args.learning_rates, data_train, args.n_steps_lst, args.n_epochs, args.device, args.retain_graph, args.refiner)
    else:
        neural_operator = model_module.net
        neural_operator.to(args.device)
        neural_operator = train_autoregressive_steps(neural_operator, args.learning_rates, data_train, model_module.net_type,
                                                     args.n_steps_lst, lat, args.n_epochs, args.device, args.retain_graph, args.refiner)

    # evaluation autoregressive
    for ar_step in args.n_steps_lst:
        print(f"unroll evaluation for step size = {ar_step}")
        autoregressive_unroll(neural_operator=neural_operator, net_type=model_module.net_type, data_module=data_module, n_steps=ar_step, eval_type="validation", args=args)
        autoregressive_unroll(neural_operator=neural_operator, net_type=model_module.net_type, data_module=data_module, n_steps=ar_step, eval_type="test", args=args)