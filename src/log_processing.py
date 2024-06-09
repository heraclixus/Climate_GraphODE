"""
process the checkpoints: obtain, for each checkpoint:
- predict_range
- model_name
- variables 
"""
import os, glob
import torch
from module import GlobalForecastModule
from data_module import GlobalForecastDataModule, collate_fn

"""
load a trained lightning module
"""
def get_metadata(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # model 
    model_hparams = state["hyper_parameters"]
    new_model = GlobalForecastModule(net_type=model_hparams["net_type"], vars=model_hparams["vars"], use_geometric_loss=True)
    # fail gracefully
    try: 
        new_model.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    except:
        print(f"error with {checkpoint_path}, removing...")
        os.remove(checkpoint_path)
        return "sfno", None, None
    config_data = state["datamodule_hyper_parameters"]
    data_module = GlobalForecastDataModule(
        root_dir = config_data["root_dir"],
        variables = config_data["variables"],
        out_variables = config_data["out_variables"],
        buffer_size= config_data["buffer_size"],
        predict_range=config_data["predict_range"]
    )
    net_type = new_model.net_type
    predict_range = config_data["predict_range"]
    vars = config_data["variables"]
    return net_type, predict_range, vars

if __name__ == "__main__":
    fno_checkpoints = [f for f in glob.glob("checkpoints/fno/*.ckpt") if "last-v" in f]
    sfno_checkpoitns = [f for f in glob.glob("checkpoints/sfno/*.ckpt") if "last-v" in f]
    with open("checkpoint_maps.csv", "w") as f:
        f.write("checkpoint,model,predict_range,vars\n")
        for checkpoint in fno_checkpoints:
            net_type, predict_range, vars = get_metadata(checkpoint)
            # assert net_type == "fno"
            print(f"{fno_checkpoints}: net_type = {net_type}, predict_range = {predict_range}, vars = {vars}")
            f.write(f"{checkpoint},{net_type},{predict_range},{vars}\n")
            print(f"finish writing info for {checkpoint}")
        for checkpoint in sfno_checkpoitns:
            net_type, predict_range, vars = get_metadata(checkpoint)
            # assert net_type == "sfno"
            print(f"{fno_checkpoints}: net_type = {net_type}, predict_range = {predict_range}, vars = {vars}")
            f.write(f"{checkpoint},{net_type},{predict_range},{vars}\n")
            print(f"finish writing info for {checkpoint}")