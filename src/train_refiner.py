import os
from data_module import GlobalForecastDataModule
from module_refiner import RefinerForecastModule
from pytorch_lightning.cli import LightningCLI
"""
temporarily, for easier debugging, use two separate classes for the standard module and the refiner based module.
"""

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=RefinerForecastModule,
        datamodule_class=GlobalForecastDataModule,
        seed_everything_default=42,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()