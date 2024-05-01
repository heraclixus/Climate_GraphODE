"""
PyTorch Lightning module for end-to-end training on:
- graphODE
- FNO 
"""
from typing import Any
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
import torch
from lib.metrics import lat_weighted_acc, lat_weighted_mse, lat_weighted_mse_val, lat_weighted_rmse
from lib.utils import LinearWarmupCosineAnnealingLR
from models.SFNO import SFNOWrapper
from models.FNO import FNO2d
from diffusers.schedulers import DDPMScheduler


"""
calling signature for the internal model (net)
- for FNO, no need for lead_times, variables, out_variables (for now)

- forward(x,y, [loss], lat=self.lat)
- evaluate(x, y, metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc], lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )
"""


class GlobalForecastModule(LightningModule):
    """
    LightingForecast Module for Weather Forecast
    """

    def __init__(self, net, 
                 lr, beta_1=0.9, beta_2=0.99, weight_decay=1e-5,
                 warmup_epochs=1000,max_epochs=2000, warmup_start_lr=1e-8, eta_min=1e-8):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"]) # temporary ignore
        self.net = net
        
        ########## Diffusion Model for PDERefiner ##########
        # Default config used in PDERefiner
        num_refinement_steps = 3
        min_noise_std = 4e-7
 
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
         

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        ########## Diffusion Model for PDERefiner ##########
        
    
    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)
    
    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
    
    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    # def training_step(self, batch: Any, batch_idx: int):
    #     x, y, lead_times, variables, out_variables = batch
    #     loss_dict, _ = self.net.forward(x, y, lat=self.lat)
    #     print(loss_dict)
    #     # loss_dict = loss_dict[0]
    #     for var in loss_dict.keys():
    #         self.log(
    #             "train/" + var,
    #             loss_dict[var],
    #             on_step=True,
    #             on_epoch=False,
    #             prog_bar=True,
    #         )
    #     loss = loss_dict["loss"]

    #     return loss

    # def validation_step(self, batch: Any, batch_idx: int):
    #     x, y, lead_times, variables, out_variables = batch

    #     if self.pred_range < 24:
    #         log_postfix = f"{self.pred_range}_hours"
    #     else:
    #         days = int(self.pred_range / 24)
    #         log_postfix = f"{days}_days"

    #     all_loss_dicts = self.net.evaluate(
    #         x,
    #         y,
    #         out_variables=out_variables,
    #         transform=self.denormalization,
    #         metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
    #         lat=self.lat,
    #         clim=self.val_clim,
    #         log_postfix=log_postfix,
    #     )

    #     loss_dict = {}
    #     for d in all_loss_dicts:
    #         for k in d.keys():
    #             loss_dict[k] = d[k]

    #     for var in loss_dict.keys():
    #         self.log(
    #             "val/" + var,
    #             loss_dict[var],
    #             on_step=False,
    #             on_epoch=True,
    #             prog_bar=False,
    #             sync_dist=True,
    #         )
    #     return loss_dict

    # def test_step(self, batch: Any, batch_idx: int):
    #     x, y, lead_times, variables, out_variables = batch

    #     if self.pred_range < 24:
    #         log_postfix = f"{self.pred_range}_hours"
    #     else:
    #         days = int(self.pred_range / 24)
    #         log_postfix = f"{days}_days"

    #     all_loss_dicts = self.net.evaluate(
    #         x,
    #         y,
    #         transform=self.denormalization,
    #         out_variables=out_variables,
    #         metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
    #         lat=self.lat,
    #         clim=self.test_clim,
    #         log_postfix=log_postfix,
    #     )

         
    #     loss_dict = {}
    #     for d in all_loss_dicts:
    #         for k in d.keys():
    #             loss_dict[k] = d[k]

    #     for var in loss_dict.keys():
    #         self.log(
    #             "test/" + var,
    #             loss_dict[var],
    #             on_step=False,
    #             on_epoch=True,
    #             prog_bar=False,
    #             sync_dist=True,
    #         )
    #     # new 4/29: add visualization
    #     #self.net.visualize_spectrum(x, y, self.lat, out_variables, batch_idx)

    #     return loss_dict

    #############################################

    def predict_next_solution(self, x, y):
        # pderefiner: self.hparams.time_future: 1
        # y_noised = torch.randn(
        #     size=(x.shape[0], self.hparams.time_future, *x.shape[2:]), dtype=x.dtype, device=x.device
        # )
 
        y_noised = torch.randn(
            size=(x.shape[0], 1, *x.shape[2:]), dtype=x.dtype, device=x.device
        )

        for k in self.scheduler.timesteps:
            #time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k 
            pred = self.net.forward(torch.cat((x, y_noised), dim=1), y_noised, lat=self.lat)
            #print('\n>>>> Checking pred format -----> ', type(pred), type(pred[0]), type(pred[1]), '\n\n')
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised 
        return y
    
    def train_step(self, batch):
        '''
        PDE-refiner: Add noise to target 
        '''
         
        x, y, lead_times, variables, out_variables = batch  
        # print('\n\nx.shape y.shape', x.shape, y.shape)
        # print(x[:6,0,0,0], y[:6,0,0,0], torch.mean(y))
        # exit()
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)      
        noise_factor = self.scheduler.alphas_cumprod.to(x.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(x.ndim - 1)])
        
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(y)
        y_noised = self.scheduler.add_noise(y, noise, k) 
        #pderefiner version:  pred = self.model(x_in, time=k * self.time_multiplier, z=cond)
        pred = self.net.forward(torch.cat((x, y_noised), dim=1), y, lat=self.lat)
        
        #pderefiner version: loss = self.train_criterion(pred, target)
        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
         
        # SFNO: loss_dict = l2loss_sphere(solver=self.net.sw_solver, prd=pred, tar=target, vars=self.net.vars, lat=self.lat)
        loss_dict = lat_weighted_mse(pred, target, vars=self.net.vars, lat=self.lat) # FNO
        return loss_dict 
       
         
    def training_step(self, batch: Any, batch_idx: int):
        #x, y, lead_times, variables, out_variables = batch
        #loss_dict, _ = self.net.forward(x, y, lat=self.lat)
        loss_dict = self.train_step(batch)
        print(loss_dict)
        # loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]

        return loss
    
    def forward(self, x, y):
        return self.predict_next_solution(x, y)

    def eval_step(self, batch):
        #x, y, cond = batch
        x, y, lead_times, variables, out_variables = batch

        pred = self.predict_next_solution(x, y)
        #loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()} 
        return pred

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        # all_loss_dicts = self.net.evaluate(
        #     x,
        #     y,
        #     out_variables=out_variables,
        #     transform=self.denormalization,
        #     metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
        #     lat=self.lat,
        #     clim=self.val_clim,
        #     log_postfix=log_postfix,
        # )
        ###################### PDE-refiner
        preds = self.eval_step(batch)
        all_loss_dicts = [m(preds, y, self.denormalization, out_variables, self.lat, self.val_clim, log_postfix) for m in [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]]
        ###################### PDE-refiner
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        # all_loss_dicts = self.net.evaluate(
        #     x,
        #     y,
        #     transform=self.denormalization,
        #     out_variables=out_variables,
        #     metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
        #     lat=self.lat,
        #     clim=self.test_clim,
        #     log_postfix=log_postfix,
        # )
        preds = self.eval_step(batch)
        all_loss_dicts = [m(preds, y, self.denormalization, out_variables, self.lat, self.test_clim, log_postfix) for m in [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]]

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}