# Climate_GraphODE

__Goal: model Climate Dynamics with Graph Neural Network (GNN)-based Neural differential equations,__ with the intermediate goal of weather forecast. 


## Dataset 
- Weatherbench2 
    - ERA5
- ClimaX
- ClimSim 

### ERA 5 

obtain the dataset by wget: 

- For geopotential only: 
```
cd data
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fgeopotential_500&files=geopotential_500_5.625deg.zip" -O geopotential_500_5.625deg.zip
mkdir geopotential_5.625deg
mv geopotential_500_5.625.deg.zip geopotential_5.625deg && cd geopotential_5.625deg
unzip geopotential_500_5.625deg.zip 
cd src/scripts
python data_utils.py 
```
to generate npy datasets in `data/geopotential_500_5.625deg_np/` 


- complete data, all the features:
```
cd data
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=all_5.625deg.zip" -O all_5.625deg.zip
```



## Models 
- LGODE 
- Physics-informed Neural ODE / Graph ODE 
- Graph Neuarl Operators 
- Fourier Neural Operators 
- Spatiotemporal Transformers 
- Stochastic GraphODE 
- Stochastic PDE 

## Project Goals and Milestones 


### Step 1: Download ERA5 Dataset and Examine Data Loading from ClimaX Github Repository
- [x] Download ERA5 
- [x] Examine the data loader from PyTorch Lighthing

This part has the logic in `src/scripts/data_module.py`, where the data loader has a minibatch of objects:
- `x`: input of shape $(B, C_{in}, H, W)$
- `y`: output of shape $(B, C_{out}, H, w)$
- `lead_time`
- input and output variable(s). 

Here $C_{in}$ and $C_{out}$ refers to how many input features to use for training and how many output features for prediction. 


### Step 2a: Incorporate the LGODE model to use the setup 
- current setup is one-step prediction. May need to modify the dataset class to use time series instead of one time dimension. 
- Also, need to construct a graph from the dataset (grids). 

- [x] Modify the dataset class to incorporate time series.  
- [x] Utility function to generate graph for the dataset.

To train LGODE:
```bash
cd src
./train.sh
```

**03/11/23**

- Current train/val/test split:
  - train: 2015 (Jan -> mid-Feb); test: 2017 (Jan -> mid-Feb)
  - Grid size: 10x20 (shrinked) 
  - Batch size: 5
  - Minimal RMSE: 2615.763 

**Issues**

- **Data loading**: The encoder of LGODE creates a spatio-temporal graph of size #timestep * #nodes. Each node represents a grid cell $(i)$ at a unique observed timestamp ($t$); the node attribute is denoted as $x_i^t$. The encoder is able to capture the relationship between $(x_i^t, x_j^t')$, i.e., the observation of grid cell $i$ at time $t$ and grid cell $j$ at time $t'$.  Although this construction captures fine-grained spatio-temporal relationship, it demands high computational resources. In particular, when loading the entire grid containing 2048 nodes with #timestep=33, the process was killed during executing `transfer_one_graph()`. 
  - Sol 1: In the current version, `transfer_one_graph()` creates a $NT\times NT$ matrix and then obtains the index and values of the nonzero entries. We may consider directly providing the nonzero entries and their values without creating the full matrix.
  - Sol 2: Use the original graph topology and assign the observed time series as the node attributes. Suppose we apply a linear transformation to all variables at each observed time step. Each node has a feature matrix of size $d \times T_{\text{obs}}$ where $d$ is the hidden dimension and $T_{\text{obs}}$ is the number of time steps. We then apply a function that aggregates vectors along the temporal dimension to transform the feature matrix to a vector of size $d$. Such a function consists of attention mechanisms that compute the contribution of individual timestamp to the aggregated vector. Encoding spatial interaction can be captured by either GCN or GAT layers after the aggregation over temporal dimension.    
- **Initial Condition Error**: In the prediction, the initial condition's distribution has a much smaller variance than ground truth.
  - Sol 1: Normalize each individual temporal graph sequence to have zero-mean and a fixed standard deviation at the initial time step. Rescale after making prediction. 
  - Sol 2: Modify the prior distribution to incorporate the mean and standard deviation of the initial condition. The ELBO loss  penalizes large KL-divergence between the distribution of hidden initial condition and the prior, which is a normal distribution.  
    


### Step ab: Incorporate the FNO model to use the setup 
The Fourier Neural Operator and neural operator models in general can be used in this case.
- [ ] Use FNO for one-step prediction. 


### Step 4: Visualization 
We need to produce some sensible visualizations for the output feature forecast over the surface of earth. 
- [ ] Finish visualization scripts. 