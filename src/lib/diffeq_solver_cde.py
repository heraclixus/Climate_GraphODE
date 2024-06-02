import torch
import torchcde 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from lib.linear_interpolation import LinearInterpolation
from torchdiffeq import odeint


#=============================================================================================
# Convolution in physical space = pointwise mutliplication of complex tensors in Fourier space
#=============================================================================================

def compl_mat_vec_mul_1d(A, z):
    """A: contains complex matrices of coefficients  (2, dim_x, hidden_size, hidden_size)
       z: (batch, 2, dim_x, hidden_size)
       out: (batch, 2, dim_x, hidden_size)
    """
    op = partial(torch.einsum, "xij, bxj-> bxi") 

    return torch.stack([
        op(A[0], z[:, 0]) - op(A[1], z[:, 1]),
        op(A[1], z[:, 0]) + op(A[0], z[:, 1])
    ], dim=1)

def compl_mat_vec_mul_2d(A, z):
    """A: contains complex matrices of coefficients  (2, dim_x, dim_y, hidden_size, hidden_size)
       z: (batch, 2, dim_x, dim_y, hidden_size)
       out: (batch, 2, dim_x, dim_y, hidden_size)
    """
    op = partial(torch.einsum, "xyij, bxyj-> bxyi") 

    return torch.stack([
        op(A[0], z[:, 0]) - op(A[1], z[:, 1]),
        op(A[1], z[:, 0]) + op(A[0], z[:, 1])
    ], dim=1)



#=============================================================================================
# Non-linear ODE
#=============================================================================================

class ODE(torch.nn.Module):
    """Differential Equation solver in Fourier space: R'(t) = A*R(t).
       A is a complex matrix resulting from a prior p;
    """

    def __init__(self, pde_func, z_dim, modes1, modes2):
        super(ODE, self).__init__() 
        
        scale = 1./(z_dim**2)
        self.A = nn.Parameter(scale * torch.rand(2, modes1, modes2, z_dim, z_dim)) 
        self.modes = [modes1, modes2]
        self.pde_func = pde_func

    def forward(self, t, v):
        """ v: (batch, 2, dim_x, (possibly dim y), hidden_size)"""
        dim_x, dim_y = v.shape[2], v.shape[3]
        # v is of shape (batch, 2, modes1, possibly modes2, z_dim) 
        # lower and upper bounds of selected frequencies
        freqs = [ (v.size(2+i)//2 - self.modes[i]//2, v.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]
        out_size = v.size() 
        # compute Av
        freqs = [ (v.size(2+i)//2 - self.modes[i]//2, v.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]
        Av = torch.zeros(v.size(), device=v.device, dtype=v.dtype)
        Av[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] = compl_mat_vec_mul_2d(self.A, v[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] )
       
        v = torch.fft.ifftshift(v, dim=[2, 3]) # centering modes
        v = torch.view_as_complex(v.permute(0,2,3,4,1).contiguous()) # (batch, modes1, modes2, z_dim) -- complex
        z = torch.fft.ifftn(v, dim=[1, 2], s=[dim_x, dim_y]).real.permute(0,3,1,2) # FFT^-1(v) (batch, z_dim, dim_x, dim_y) -- real
        # 2) H o FFT^-1
        # F_z is of shape (batch, z_dim, dim_x, possibly dim_y)        
        F_z = self.pde_func(z) 
        H = F_z
        # 3) FFT o H o FFT^-1
        out_ft = torch.zeros(out_size, device=H.device, dtype=H.dtype)
        v = torch.fft.fftn(H, dim=[2,3]) # FFT(H) (batch, z_dim, dim_x, dim_y) -- complex 
        v = torch.fft.fftshift(v, dim=[2,3]) # centering modes
        v = torch.stack([v.real, v.imag], dim=1) # (batch, 2, z_dim, dim_x, dim_y) 
        v = v.permute(0,1,3,4,2)  # (batch, 2, dim_x, dim_y, z_dim) 
        out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = v[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] 

        # We form the vector field A + FFT o H o FFT^-1
        sol = Av + out_ft
        return sol


#=============================================================================================
# Non-linear controlled ODE
#=============================================================================================

class ControlledODE(torch.nn.Module):
    """Differential Equation solver in Fourier space: R'(t) = A*R(t) + control.
       A is a complex matrix resulting from a prior p;
       control is the space Fourier transform of H (see paper).
    """

    def __init__(self, cpde_func, z_dim, modes1, modes2):
        super(ControlledODE, self).__init__() 
        
        scale = 1./(z_dim**2)
        self.A = nn.Parameter(scale * torch.rand(2, modes1, modes2, z_dim, z_dim)) 
        self.modes = [modes1, modes2]
        self.cpde_func = cpde_func

    def forward(self, t, v):
        """ v: (batch, 2, dim_x, (possibly dim y), hidden_size)"""

        freqs = [ (v.size(2+i)//2 - self.modes[i]//2, v.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]

        Av = torch.zeros(v.size(), device=v.device, dtype=v.dtype)
        Av[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = compl_mat_vec_mul_2d(self.A, v[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] )

        return Av

    def prod(self, t, v, xi):
        # v is of shape (batch, 2, modes1, modes2, z_dim) 
        # xi is of shape (batch, dim_x, dim_y, channel_size)

        # lower and upper bounds of selected frequencies
        freqs = [ (v.size(2+i)//2 - self.modes[i]//2, v.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]
        out_size = v.size() 
        xi = xi[:,0,...] # we had to dupplicate xi so that its shape was compatible with the requirements of cdeint

        # compute Av
        Av = self.forward(t, v)

        # 1) FFT^-1
        dim_x, dim_y = xi.size(1), xi.size(2)
        v = torch.fft.ifftshift(v, dim=[2, 3]) # centering modes
        v = torch.view_as_complex(v.permute(0,2,3,4,1).contiguous()) # (batch, modes1, modes2, z_dim) -- complex
        z = torch.fft.ifftn(v, dim=[1, 2], s=[dim_x, dim_y]).real.permute(0,3,1,2) # FFT^-1(v) (batch, z_dim, dim_x, dim_y) -- real

        # 2) H o FFT^-1
        
        # F_z is of shape (batch, z_dim, dim_x, possibly dim_y)
        # G_z is of shape (batch, z_dim, channel_size, dim_x, possibly dim_y)   
        F_z, G_z = self.cpde_func(z) 
    
        G_z_xi = torch.einsum('bhnxy, bxyn -> bhxy', G_z, xi) # Not sure...
        
        # H is of shape (batch, z_dim, dim_x, possibly dim_y)
        H = F_z + G_z_xi

        # 3) FFT o H o FFT^-1
        out_ft = torch.zeros(out_size, device=H.device, dtype=H.dtype)
      
        v = torch.fft.fftn(H, dim=[2,3]) # FFT(H) (batch, z_dim, dim_x, dim_y) -- complex 
        v = torch.fft.fftshift(v, dim=[2,3]) # centering modes
        v = torch.stack([v.real, v.imag], dim=1) # (batch, 2, z_dim, dim_x, dim_y) 
        v = v.permute(0,1,3,4,2)  # (batch, 2, dim_x, dim_y, z_dim) 
        out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = v[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] 

        # We form the vector field A + FFT o H o FFT^-1
        sol = Av + out_ft
       
        return sol



#=============================================================================================
# PDE solver: linear controlled differential equation solver in Fourier space.
#=============================================================================================

class DiffeqSolver(nn.Module):
    def __init__(self, z_dim, pde_func, modes1, modes2, **kwargs):
        super(DiffeqSolver, self).__init__()

        self.pde_func = pde_func
        
        self.ode = ODE(pde_func, z_dim, modes1, modes2)
  
        self.dims = [2,3] 
        self.modes = [modes1, modes2]

        if 'adjoint' not in kwargs:
            kwargs['adjoint']=False
        self.kwargs = kwargs

    # TODO: check xi and time 
    def forward(self, z0, xi, grid=None):
        """ - z0: (batch, z_dim, dim_x, (possibly dim_y))
            - xi: (batch, x_dim, dim_x, dim_y, dim_t)
            - grid: should be speficied if computing gradients of the solution
        """

        # lower and upper bounds of selected frequencies
        freqs = [ (z0.size(2+i)//2 - self.modes[i]//2, z0.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]

        # compute fourier transform of initial condition  
        z0_ft = torch.fft.fftshift(torch.fft.fftn(z0, dim=self.dims), dim=self.dims) 
        z0_ft = torch.stack([z0_ft.real, z0_ft.imag], dim=1) # (batch, 2, z_dim, dim_x, possibly dim_y)

        # antialiasing (the highest modes are set to zero)  # TODO: would padding be more efficient?
        v0 = torch.zeros(z0_ft.size(), device=z0.device, dtype=z0.dtype)
     
        v0[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = z0_ft[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] 
        
        # reshape for odeint 
        v0 = v0.permute(0,1,3,4,2) # (batch, 2, dim_x, dim_y, z_dim)  

        # maynot be required for ODE for the time steps (could be just the output)
        xi = xi.permute(0,2,3,4,1) # (batch, dim_x, dim_y, dim_t, x_dim)
        xi = torch.stack([xi, torch.zeros_like(xi)], dim=1) # (batch,2, dim_x, possibly dim_y, dim_t, x_dim)
        xi = torchcde.linear_interpolation_coeffs(xi)
        xi = LinearInterpolation(xi) 

        # Solve the ODE,  get v of shape (batch, 2, dim_x, (possibly dim_y), dim_t, z_dim) 
        # odeint gives (dim_t, batch, 2, dim_x, dim_y, z_dim)
        v = odeint(func=self.ode, y0=v0, t=xi._t)
        # Compute z = FFT^-1(v) 
        v = v.permute(1,5,3,4,0,2) # (batch, z_dim, dim_x, dim_y, dim_t, 2) 
        v = torch.view_as_complex(v.contiguous()) # (batch, z_dim, dim_x, dim_y, dim_t) -- complex 
        z = torch.fft.ifftn(torch.fft.ifftshift(v, dim=self.dims), dim=self.dims).real  # (batch, z_dim, dim_x, dim_y, dim_t) -- real 
        return z  # (batch, z_dim, dim_x, dim_t)



#=============================================================================================
# CPDE solver: linear controlled differential equation solver in Fourier space.
#=============================================================================================

class ControlledDiffeqSolver(nn.Module):
    def __init__(self, z_dim, cpde_func, modes1, modes2, **kwargs):
        super(ControlledDiffeqSolver, self).__init__()

        self.cpde_func = cpde_func
        
        self.cde = ControlledODE(cpde_func, z_dim, modes1, modes2)

       
        self.dims = [2,3] 
        self.modes = [modes1, modes2]

        if 'adjoint' not in kwargs:
            kwargs['adjoint']=False
        self.kwargs = kwargs

    def forward(self, z0, xi, grid=None):
        """ - z0: (batch, z_dim, dim_x, (possibly dim_y))
            - xi: (batch, x_dim, dim_x, (possibly dim_y), dim_t)
            - grid: should be speficied if computing gradients of the solution
        """

        # lower and upper bounds of selected frequencies
        freqs = [ (z0.size(2+i)//2 - self.modes[i]//2, z0.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]

        # compute fourier transform of initial condition  
        z0_ft = torch.fft.fftshift(torch.fft.fftn(z0, dim=self.dims), dim=self.dims) 
        z0_ft = torch.stack([z0_ft.real, z0_ft.imag], dim=1) # (batch, 2, z_dim, dim_x, possibly dim_y)

        # antialiasing (the highest modes are set to zero)
        v0 = torch.zeros(z0_ft.size(), device=z0.device, dtype=z0.dtype)
       
        v0[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = z0_ft[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] 
        
        # reshape for cdeint
        v0 = v0.permute(0,1,3,4,2) # (batch, 2, dim_x, dim_y, z_dim)
        xi = xi.permute(0,2,3,4,1) # (batch, dim_x, dim_y, dim_t, x_dim)

        # hack so that xi's shape is compatible with that of v0
        xi = torch.stack([xi, torch.zeros_like(xi)], dim=1) # (batch,2, dim_x, possibly dim_y, dim_t, x_dim)

        # interpolate xi so that it can be queried at any time t 
        xi = torchcde.linear_interpolation_coeffs(xi)
        xi = LinearInterpolation(xi) 

        # Solve the CDE,  get v of shape (batch, 2, dim_x, (possibly dim_y), dim_t, z_dim) 
        v = torchcde.cdeint(X=xi,
                            z0=v0,
                            func=self.cde,
                            t=xi._t,
                            # adjoint = self.kwargs['adjoint'],
                            **self.kwargs) 

        # Compute z = FFT^-1(v)       
        v = v.permute(0,5,2,3,4,1) # (batch, z_dim, dim_x, dim_y, dim_t, 2) 
        v = torch.view_as_complex(v.contiguous()) # (batch, z_dim, dim_x, dim_y, dim_t) -- complex 
        z = torch.fft.ifftn(torch.fft.ifftshift(v, dim=self.dims), dim=self.dims).real  # (batch, z_dim, dim_x, dim_y, dim_t) -- real 
        return z