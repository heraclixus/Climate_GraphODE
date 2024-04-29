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

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        loss_dict, _ = self.net.forward(x, y, lat=self.lat)
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

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            out_variables=out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
        )

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

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            transform=self.denormalization,
            out_variables=out_variables,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )

        # new 4/29: add visualization
        self.net.visualize_spectrum(x, y, self.lat, out_variables, batch_idx)

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