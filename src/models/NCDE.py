"""
Neural CDE with Operator as the drift, other choises also possible: (GNN/CNN)
"""
import torch
import torchcde
from models.SFNO import SFNOWrapper

"""
Utility layers
"""

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        """ TODO: add possibility to have more layers 
        """

        model = [torch.nn.Conv2d(in_size, out_size, 1), torch.nn.BatchNorm2d(out_size), torch.nn.Tanh()] 

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y)"""
        return self._model(x)

# a custom FNO class for this purpose only
class FNO_drift(torch.nn.Module):
    def __init__(self, modes1, modes2, in_channels, hidden_channels, L):
        super(FNO_drift, self).__init__()

        """ an FNO to model F(u(t)) where u(t) is a function of 2 spatial variables """

        self.fc0 = torch.nn.Linear(in_channels, hidden_channels)

        self.net = [ FNO_layer(modes1, modes2, hidden_channels) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, hidden_channels, last=True) ]
        self.net = torch.nn.Sequential(*self.net)

        self.fc1 = torch.nn.Linear(hidden_channels, 128)
        self.fc2 = torch.nn.Linear(128, in_channels)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y)"""
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = self.net(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.tanh(x)
        return x

class FNO_layer(torch.nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last
        self.conv = ConvolutionSpace(width, modes1, modes2)
        self.w = torch.nn.Linear(width, width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y)"""
        x1 = self.conv(x)
        x2 = self.w(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 + x2
        if not self.last:
            x = torch.nn.functional.gelu(x)           
        return x


class ConvolutionSpace(torch.nn.Module):
    def __init__(self, channels, modes1, modes2):
        super(ConvolutionSpace, self).__init__()

        """ ...    
        """
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1. / (channels**2)
        self.weights = torch.nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2,  2))


    def forward(self, x):
        """ x: (batch, channels, dim_x, dim_y)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
        y0, y1 = x.size(3)//2 - self.modes2//2, x.size(3)//2 + self.modes2//2

        # Compute FFT of the input signal to convolve
        x_ft = torch.fft.fftn(x, dim=[2, 3])
        x_ft = torch.fft.fftshift(x_ft, dim=[2, 3])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), device=x.device, dtype=torch.cfloat)
        out_ft[:, :, x0:x1, y0:y1] = compl_mul2d_spatial(x_ft[:, :, x0:x1, y0:y1], torch.view_as_complex(self.weights))

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2, 3])
        x = torch.fft.ifftn(out_ft, dim=[2, 3], s=(x.size(2), x.size(3)))

        return x.real


def compl_mul2d_spatial(a, b):
    """ ...
    """
    return torch.einsum("aibc, ijbc -> ajbc", a, b)



######################
# A CDE model in infinite dimension looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s + \int_0^t g_\theta(z_s)ds
#
# Where z_s is a function of 2 independent space variables
# and where X is your data and f_\theta and g_\theta are neural networks. 
#
# So the first thing we need to do is define such an f_\theta and a g_\theta
# That's what this CDEFunc class does.
######################

class CDEFunc(torch.nn.Module):

    """
    modes1, modes2: number of fourier modes for 2D 
    width: hidden_dim in FNO
    hidden_size: dim of encoded x (z)
    data_size: dim of x
    net_type: fno | sfno | gnn | cnn
    """
    def __init__(self, modes1, modes2, hidden_size, data_size, net_type):
        super().__init__()
        self.data_size = data_size
        # F and G are resolution invariant MLP (acting on the channels).
        if net_type == "FNO":  
            self._F = FNO_drift(modes1=modes1, modes2=modes2, in_channels=hidden_size, hidden_channels=hidden_size, L=1) 
        elif net_type == "SFNO":
            self._F = SFNOWrapper(vars=vars, in_chans=hidden_size, out_chans=hidden_size, num_layers=3, use_geometric_loss=True)
        # self._F = MLP(hidden_size, hidden_size)  
        self._G = MLP(hidden_size, hidden_size * data_size)

    def forward(self, t, z):
        """ z: (batch, hidden_size, dim_x, dim_y)"""
        _, F_out = self._F(z)
        _, G_out = self._G(z)
        G_out = G_out.view(z.size(0), self._hidden_size, self.data_size, z.size(2), z.size(3))
        return F_out, G_out


    def prod(self, t, z, control_gradient):
        # z is of shape (N, dim_x, dim_y, hidden_channels) 
        # control_gradient is of shape (N, dim_x, dim_y, X_size)
        z = z.permute(0, 3, 1, 2)
        control_gradient = control_gradient.permute(0, 3, 1, 2)

        # z is of shape (N, hidden_channels, dim_x, dim_y) 
        # control_gradient is of shape (N, noise_size, dim_x, dim_y)
        Fu, Gu = self.forward(t, z)
        # Gu is of shape (N, hidden_size, noise_size, dim_x, dim_y)
        # Fu is of shape (N, hidden_size, dim_x, dim_y)
        Guxi = torch.einsum('abcde, acde -> abde', Gu, control_gradient)
        sol = Fu + Guxi
        # sol is of shape (N, hidden_size, dim_x, dim_y)       
        return sol.permute(0, 2, 3, 1)


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, data_size, hidden_channels, output_channels, interpolation="linear"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(data_size, hidden_channels)
        self.initial = torch.nn.Linear(data_size, hidden_channels)
        
        # self.readout = torch.nn.Linear(hidden_channels, output_channels)
        readout = [torch.nn.Linear(hidden_channels, 128), torch.nn.ReLU(), torch.nn.Linear(128, output_channels)]
        self._readout = torch.nn.Sequential(*readout)
        self.interpolation = interpolation

    def forward(self, u0, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        
        # u0 is of shape #(N, data_size, dim_x, dim_y)
        z0 = self.initial(u0.permute(0, 2, 3, 1))
        # z0 is of shape #(N, dim_x, dim_y, hidden_size)
        # coeffs is of shape #(N, dim_x, dim_y, dim_t, data_size)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              # t = X.interval,
                              method='euler',
                              t=X._t)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        # z_T = z_T[:, 1]
        
        # z_T is of shape (N, dim_x, dim_y, dim_t, hidden_channels)
        pred_y = self._readout(z_T).permute(0, 4, 3, 1, 2)
        # pred_y is of shape (N, hidden_channels, dim_t, dim_x, dim_y)
        return pred_y


# TODO: test the module against simple x and z values 

if __name__ == "__main__":
    x = torch.randn((32, 3, 64, 32))
    z = torch.randn((32, 12, 64, 32))