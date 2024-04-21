import torch
import torch.nn as nn
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet
from torch_harmonics import *
from lib.metrics import *
torch.manual_seed(0)
np.random.seed(0)


"""
A wrapper class around the SFNO class to make it agree with the calling convention based on module.py
"""
class SFNOWrapper(nn.Module):
    def __init__(self, 
                 vars,
                 spectral_transform = "sht",
                 operator_type = "driscoll-healy",
                 img_size = [32, 64],
                 grid = "equiangular",
                 scale_factor = 3,
                 in_chans = 3,
                 out_chans = 3,
                 embed_dim = 256,
                 num_layers = 4,
                 activation_function = "relu",
                 encoder_layers = 1,
                 use_mlp = True,
                 mlp_ratio = 2.,
                 drop_rate = 0.,
                 drop_path_rate = 0.,
                 normalization_layer = "none",
                 hard_thresholding_fraction = 1.0,
                 use_complex_kernels = True,
                 big_skip = False,
                 factorization = None,
                 separable = False,
                 rank = 128,
                 pos_embed = False):

        super(SFNOWrapper, self).__init__()
        self.sfno_model = SphericalFourierNeuralOperatorNet(spectral_transform, operator_type, img_size, grid, scale_factor, 
                                                            in_chans, out_chans, embed_dim, num_layers, activation_function, encoder_layers,
                                                            use_mlp, mlp_ratio, drop_rate, drop_path_rate, normalization_layer, 
                                                            hard_thresholding_fraction, use_complex_kernels, big_skip, factorization, 
                                                            separable, rank, pos_embed)
        self.vars = vars
        self.out_chans = len(vars)
        
    # original SFNO only takes in x (b, c, H, W) and output same shape 
    def forward(self, x, y, lat):
        y_pred = self.sfno_model(x)  # (b,c,h,w)
        print(f"y_pred = {y_pred.shape}, y = {y.shape}")
        return lat_weighted_mse(y_pred, y, vars=self.vars, lat=lat), y_pred 


    def evaluate(self, x, y, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]



