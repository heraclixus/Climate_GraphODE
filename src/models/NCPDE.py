"""
Neural PDE Model, involving the ode solver that applies principles of Galerkin projection.
code adapted from https://github.com/crispitagorico/torchspde 
instead of using brownian motion as drive, simply use the observed data (like CDE)
compared to NCDE, the model in this case involve using the Fourier transform.

Version 1: Neural Controlled PDE 
Version 2: Neural PDE

the integration kernel can be either cnn or gnn
can even potentially try fno and sfno
"""

import torch
import torch.nn as nn 
from lib.diffeq_solver_cde import DiffeqSolver, ControlledDiffeqSolver

"""
models a PDE without control, in this case we have the integration equation:
z_t = e^{tL}z_0 + \int_0^t e^{t-s} F(z_s)ds
"""
# TODO: add different kernel functions (cnn | gnn | no, etc.)
class PDEFunction(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.conv_block = [nn.Conv2d(z_dim, z_dim, 1), nn.BatchNorm2d(z_dim), nn.Tanh()]
        self.F = nn.Sequential(*self.conv_block)

    # z: (batch, z_dim, x, y, t)
    def forward(self, z):
        return self.F(z)
    
"""
models a controlled PDE, in this case we have the integration equation:
z_t = e^{tL}z_0 + \int_0^t e^{t-s} [F(z_s) + G(z_s)X_s] ds 
"""
class CPDEFunction(nn.Module):
    """
    z_dim: dim of latent state 
    x_dim: dim of control signal
    """
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        model_F = [nn.Conv2d(z_dim, z_dim, 1), nn.BatchNorm2d(z_dim), nn.Tanh()]
        model_G = [nn.Conv2d(z_dim, z_dim * x_dim, 1), nn.BatchNorm2d(z_dim * x_dim), nn.Tanh()]
        self.F = nn.Sequential(*model_F)
        self.G = nn.Sequential(*model_G)

    # z: (batch, z_dim, x, y, t)
    def forward(self, z):
        return self.F(z), self.G(z).view(z.size(0), self.z_dim, self.x_dim, z.size(2), z.size(3))



"""
a neural pde function that integrates using torchode with PDE function as input
"""
class NeuralPDE(nn.Module):  

    def __init__(self, u_dim, x_dim, z_dim, modes1, modes2, n_iter=4, solver='diffeq', **kwargs):
        super().__init__()
        """
        x_dim: the dimension of the control state space
        z_dim: the dimension of the latent space
        modes1, modes2: Fourier modes
        solver: 'diffeq'
        kwargs: Any additional kwargs to pass to the cdeint solver of torchdiffeq
        """

        # initial lift 
        # can be more expressive as the encoder
        self.lift = nn.Linear(u_dim, z_dim)
        readout = [nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, u_dim)]
        self.readout = nn.Sequential(*readout)
        self.pde_func = PDEFunction(z_dim=z_dim)
        self.solver = DiffeqSolver(z_dim, self.pde_func, modes1, modes2, **kwargs)



    def forward(self, u0, xi, grid=None):
        """ u0: (batch, hidden_size, dim_x, (possibly dim_y))
            xi: (batch, hidden_size, dim_x, (possibly dim_y), dim_t)
            grid: (batch, dim_x, (possibly dim_y), dim_t)
        """
        if grid is not None:
            grid = grid[0]
        z0 = self.lift(u0.permute(0,2,3,1)).permute(0,3,1,2)
        zs = self.solver(z0, xi, grid)
        ys = self.readout(zs.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        return ys



"""
a neural cpde function that integrates using torchode with PDE function as input
"""
class NeuralCPDE(nn.Module):  

    def __init__(self, u_dim, x_dim, z_dim, modes1, modes2, n_iter=4, solver='diffeq', **kwargs):
        super().__init__()
        """
        in_channels: the dimension of the solution state space
        x_dim: the dimension of the control state space
        z_dim: the dimension of the latent space
        modes1, modes2, (possibly modes 3): Fourier modes
        solver: 'fixed_point', 'root_find' or 'diffeq'
        kwargs: Any additional kwargs to pass to the cdeint solver of torchdiffeq
        """
        # initial lift 
        # can be more expressive as the encoder
        self.lift = nn.Linear(u_dim, z_dim)
        readout = [nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, u_dim)]
        self.readout = nn.Sequential(*readout)
        self.pde_func = CPDEFunction(z_dim=z_dim, x_dim=x_dim)
        self.solver = ControlledDiffeqSolver(z_dim=z_dim, cpde_func=self.pde_func, modes1=modes1, modes2=modes2, **kwargs)

    def forward(self, u0, xi, grid=None):
        """ u0: (batch, hidden_size, dim_x, (possibly dim_y))
            xi: (batch, hidden_size, dim_x, (possibly dim_y), dim_t)
            grid: (batch, dim_x, (possibly dim_y), dim_t)
        """
        if grid is not None:
            grid = grid[0]
        z0 = self.lift(u0.permute(0,2,3,1)).permute(0,3,1,2)
        zs = self.solver(z0, xi, grid) # (b, z_dim, x, y, t)
        ys = self.readout(zs.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        return ys



# Tests 
if __name__ == "__main__":
    u0 = torch.randn((32, 3, 64, 32)) # (b,c,x,y)
    xi = torch.randn((32, 2, 64, 32, 10)) # (b,c,x,y,t)

    neural_pde = NeuralPDE(u_dim=3, x_dim=2, z_dim=20, modes1=8, modes2=8)

    ys = neural_pde(u0, xi)
    print(f"result shape = {ys.shape}")