import torch
import torch.nn as nn
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver
from torch_harmonics import *
from lib.metrics import *
from scripts.visualize_fourier_modes import *
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
                 in_chans = 1,
                 out_chans = 1,
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
        
        # solver 
        self.nlat, self.nlon = img_size
        self.grid = grid  # equiangular or legendre-gauss
        modes_lat = int(self.sfno_model.h * self.sfno_model.hard_thresholding_fraction)
        modes_lon = int(self.sfno_model.w//2 * self.sfno_model.hard_thresholding_fraction)
        modes_lat = modes_lon = min(modes_lat, modes_lon)

        # TODO: determine dt and grid 
        # sw_solver has issue with unrollling
        # self.sw_solver = ShallowWaterSolver(nlat=self.nlat, nlon=self.nlon, dt=10, lmax=modes_lat, mmax=modes_lon, grid=self.grid)

        self.vars = vars
        self.out_chans = len(vars)
        
    # original SFNO only takes in x (b, c, H, W) and output same shape 
    # SFNO uses its own spherical based loss function for training. 
    def forward(self, x, y, lat):
        y_pred = self.sfno_model(x)  # (b,c,h,w)
        return lat_weighted_mse(y_pred, y, vars=self.vars, lat=lat), y_pred 
        # return l2loss_sphere(solver=self.sw_solver, prd=y_pred, tar=y, vars=self.vars, lat=lat), y_pred
    
    # inference use a different type of metrics 
    def evaluate(self, x, y, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]

    # visualize spectrum after fft 
    def visualize_spectrum(self, x, y,lat, out_variables, batch_id, pred_range, type="fft"):
        _, preds = self.forward(x, y, lat=lat)
        one_step_plot_spectrum(preds, y, vars=out_variables, model_name="SFNO", batch_id=batch_id, predict_range=pred_range, type=type)
