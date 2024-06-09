## Neural CPDE and Neural PDE

Neural PDE aims to model PDEs of the form: 

$$
du_t = (\mathcal{L}u_t + \mathcal{F}(u_t)) dt 
$$

for linear operator $\mathcal{L}$ and a continuous, non-linear operator $F$. 

Neural Controlled PDE (CPDE) aims to model PDEs of the form:

$$
du_t = (\mathcal{L}u_t + \mathcal{F}(u_t)) dt + G(u_t) dX_t 
$$

where $dX_t$ is a a control signal, function belonging to a Banach space. Here we can say that $u: D \subset \mathbb{R}^2 \rightarrow \mathbb{R}^d$, e.g., $u \in \mathcal{H}_u$, a Hilbert space, then $X_t \in \mathcal{H}_X$, and $G: \mathcal{H}_u \rightarrow L(H_X, H_u)$.

The general CPDE admits an integral form mild solution which can be approximated using an integration (neural) kernel, assuming that $G$ is smooth and $dX_t$ has bounded variation, and the Neural PDE formulation can be done as following: 

$$z_0(x) = L_\theta (u_0(x))$$
$$z_t = \mathcal{F}^{-1}(\text{ODESolve}(\mathcal{F}(z_0), \Psi_{\theta}, [0,t])$$
$$u_t = \Pi_\theta(z_t)$$

where: 

$$\Psi_\theta = A + \mathcal{F} \circ H_\theta \circ \mathcal{F}^{-1}$$

$H_\theta = F_\theta(h(z)) + G_\theta(h(z))X_t$ for NCPDE, $H_\theta = F_\theta(h(z))$ For NPDE, and $X_t$ is the driving function for CPDE and $A, F, G$ are learnable. In the case of climate dataset, we can treat $u_0$ as the input data with all features, while the control path $\{X_t\}$ maybe the time series without the target feature. 

## Implementation

### Neural PDE and Neural CPDE

The implementation for Neural PDE and Neural CPDE are in `models.ncpde.py`. These two classes are used as the "vector field" in the `lib.diffeq_solver_cde` class, a class that calls integration method from `torchdiffeq` and `torchcde`. 

Constructors:

```py
# u_dim is the dimension input data.
# x_dim is the dimension of control signal.
# z_dim is the dimension of latent ODE (after Fourier transform).
# modes are Fourier modes.
neural_pde = NeuralPDE(u_dim, x_dim, z_dim, modes1, modes2)
neural_cpde = NeuralCPDE(u_dim, x_dim, z_dim, modes1, modes2)
```

calling convention:
```py
# u0 is the initial state (b,u_dim, x, y)
# xi is the time series of controlled signal (b, x_dim)
neural_pde(u0, xi)
neural_cde(u0, xi)
```
output should be the target at target time step, of shape $(B, d_u, x, y)$


### DiffeqSolver and ControlledDiffeqSolver

implementation for the solver follows the equation before:
$$z_0(x) = L_\theta (u_0(x))$$
$$z_t = \mathcal{F}^{-1}(\text{ODESolve}(\mathcal{F}(z_0), \Psi_{\theta}, [0,t]))$$
$$u_t = \Pi_\theta(z_t)$$

Here, the ODE and CDE functions are implemented in `lib.diffeq_solver_cde.ODE` and `diffeq_solver_cde.ControlledODE`, respectively. For the first class, it corresponds to a Fourier transform of signal + pde function ($H$)  + a inverse fourier transform. The PDE functions are defined in `models.NCPDE.PDEFunction` and `models.NCPDE.CPDEFunction`. 

```py
cpde_function = CPDEFunction(z_dim, x_dim)
controlledODE = ControlledODE(cpde_func=cpde_function, z_dim, modes1, modes2)
controlled_solver = ControlledDiffeqSolver(z_dim, cpde_func, modes1, modes2)
```


### Integration Tests

The shape test (shapes agree) can be found in `integration_test.py`. 


### Extension

The design of $H$ is key. In the most basic case we have CNN (convolution2D) for the ODEfunction (or CDEFunction) class, but it is possible to use GNN & Neural Operator to do it (Neural Operator is debatable since a Fourier transform is already performed.)


### Open Question: How to choose Control?

For climate forecast, is there a good choice of the control signal? We don't want to leak the data at time $T$.

Also is it possible to combine CDE and ODE? (e.g., support with $\{X_t\}_{t=0}^{T-\Delta t}$ but not for $\{X_{t}\}_{t=T-\Delta t}^{T}$)