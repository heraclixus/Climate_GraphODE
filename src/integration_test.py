from models.NCPDE import NeuralCPDE, NeuralPDE
import torch
import torchcde 
from lib.diffeq_solver_cde import DiffeqSolver, ControlledDiffeqSolver
from models.NCPDE import CPDEFunction, PDEFunction

def testNCPDE():
    u0 = torch.randn((32, 3, 64, 32)) # (b,c_u,x,y)
    xi = torch.randn((32, 2, 64, 32, 10)) # (b,c_x, x,y,t)
    neural_pde = NeuralCPDE(u_dim=3, x_dim=2, z_dim=20, modes1=8, modes2=8)
    ys = neural_pde(u0, xi)
    print(f"result shape = {ys.shape}") # (b, c_u, x, y, t) generates an entire series
    print("passed shape test for neural CPDE")

def testNPDE():
    u0 = torch.randn((32, 3, 64, 32)) # (b,c_u,x,y)
    xi = torch.randn((32, 2, 64, 32, 10)) # (b,c_x, x,y,t)
    neural_pde = NeuralPDE(u_dim=3, x_dim=2, z_dim=20, modes1=8, modes2=8)
    ys = neural_pde(u0, xi)
    print(f"result shape = {ys.shape}") # (b, c_u, x, y, t) generates an entire series
    print("passed shape test for neural PDE")


def testPDEFunction():
    z0 = torch.randn((32, 10, 64, 32)) # initial latent state
    xi = torch.randn((32, 3, 64, 32, 5)) # time series size 5
    pde_function = PDEFunction(z_dim=10)
    solver = DiffeqSolver(z_dim=10, pde_func=pde_function, modes1=8, modes2=8)
    z_t = solver(z0, xi)
    print(f"PDE: latent path = {z_t.shape}")
    print("passed shape test for PDE Function")

def testCPDEFunction_wo_forcing():
    z0 = torch.randn((32, 10, 64, 32)) # initial latent state
    xi = torch.zeros((32, 3, 64, 32, 72)) # time series 
    cpde_function = CPDEFunction(z_dim=10, x_dim=3)
    solver_c = ControlledDiffeqSolver(z_dim=10, cpde_func=cpde_function, modes1=8, modes2=8)
    z_t = solver_c(z0, xi)
    print(f"CPDE: latent path = {z_t.shape}")
    print("passed shape test for CPDE Function without input driving")


def testCPDEFunction():
    z0 = torch.randn((32, 10, 64, 32)) # initial latent state
    xi = torch.randn((32, 3, 64, 32, 72)) # time series 
    cpde_function = CPDEFunction(z_dim=10, x_dim=3)
    solver_c = ControlledDiffeqSolver(z_dim=10, cpde_func=cpde_function, modes1=8, modes2=8)
    z_t = solver_c(z0, xi)
    print(f"CPDE: latent path = {z_t.shape}")
    print("passed shape test for CPDE Function")

if __name__ == "__main__":
    # testPDEFunction()
    # testCPDEFunction()
    # testNPDE()
    # testNCPDE()
    testCPDEFunction_wo_forcing()