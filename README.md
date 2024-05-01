# Climate_GraphODE

__Goal: model Climate Dynamics with Graph Neural Network (GNN)-based Neural differential equations,__ with the intermediate goal of weather forecast. 


## Dataset 
- Weatherbench2 
    - ERA5
- ClimaX
- ClimSim 

### ERA 5 

obtain the dataset by wget: 

 
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
    


### Step 2b: Incorporate the FNO model to use the setup 
The Fourier Neural Operator and neural operator models in general can be used in this case.
- [x] Use FNO for one-step prediction. 

Run FNO:

Suggestion on dependency: use the dependency from climaX (as the `torch.fft.rfft` can run into trouble with newer versions of pytorch, same applies to `torch-lightning`). 

```sh
python3 src/train.py --config configs/forecast_fno.yaml --trainer.devices=1 --trainer.max_epochs=500 --data.predict_range=72 --data.out_variables=["2m_temperature"] --data.batch_size=16 --data.variables=["2m_temperature"]
```


### Step 2c: Incorporate the SFNO model to use the setup 

For each new model, to make torch-lightning work, one way is to have one yaml file for each different model, which is the curreny implementation.

Dependency suggestion: build `torch_harmonics` package from source rather than using `pip`, also install `tensorly-torch` and `tensorly` as dependency. 

- [x] check the yaml file for tuples and list for hyperparameters
- [x] use SFNO for one-step prediction.

Run SFNO:

```sh
python3 src/train.py --config configs/forecast_sfno.yaml --trainer.devices=1 --trainer.max_epochs=500 --data.predict_range=72 --data.out_variables=["2m_temperature"] --data.batch_size=128 --data.variables=["2m_temperature"]
```

geopotential
```sh
python3 src/train.py --config configs/forecast_sfno.yaml --trainer.devices=1 --trainer.max_epochs=500 --data.predict_range=72 --data.out_variables=["geopotential"] --data.batch_size=16 --data.variables=["geopotential"]
```

### Preliminary results: 

1. Base setting: 
- Short range (72 hours) weather forecast
- 1 feature (`2m_temperature`)
- 1979-2016: training
- 2017 validation, 2018 test. 
- same MSE based loss for both models. 

SFNO 
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            Test metric            ┃           DataLoader 0            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│             test/acc              │         0.86073899269104          │
│  test/acc_2m_temperature_3_days   │         0.86073899269104          │
│            test/w_mse             │       0.015408281236886978        │
│ test/w_mse_2m_temperature_3_days  │       0.015408281236886978        │
│            test/w_rmse            │         2.624769449234009         │
│ test/w_rmse_2m_temperature_3_days │         2.624769449234009         │
└───────────────────────────────────┴───────────────────────────────────┘
```

FNO

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            Test metric            ┃           DataLoader 0            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│             test/acc              │        0.8777772188186646         │
│  test/acc_2m_temperature_3_days   │        0.8777772188186646         │
│            test/w_mse             │       0.013594500720500946        │
│ test/w_mse_2m_temperature_3_days  │       0.013594500720500946        │
│            test/w_rmse            │        2.4645509719848633         │
│ test/w_rmse_2m_temperature_3_days │        2.4645509719848633         │
└───────────────────────────────────┴───────────────────────────────────┘
```

2. SFNO training with spherical loss 


3. SFNO training with spherical loss and PDE-refiner 
- Short range (72 hours) weather forecast
- 1 feature (`geopotential`)
- 1979-2016: training
- 2017 validation, 2018 test. 
- same MSE based loss for both models. 

SFNO 

Trained after 7 epochs
 
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│            test/acc             │       0.10115499049425125       │
│  test/acc_geopotential_3_days   │       0.10115499049425125       │
│           test/w_mse            │       0.38705867528915405       │
│ test/w_mse_geopotential_3_days  │       0.38705867528915405       │
│           test/w_rmse           │        2084.53271484375         │
│ test/w_rmse_geopotential_3_days │        2084.53271484375         │
└─────────────────────────────────┴─────────────────────────────────┘
```

2m_temperature
Trained after 5 epochs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            Test metric            ┃           DataLoader 0            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│             test/acc              │        0.08266861736774445        │
│  test/acc_2m_temperature_3_days   │        0.08266861736774445        │
│            test/w_mse             │        0.3225284814834595         │
│ test/w_mse_2m_temperature_3_days  │        0.3225284814834595         │
│            test/w_rmse            │         12.00672721862793         │
│ test/w_rmse_2m_temperature_3_days │         12.00672721862793         │
└───────────────────────────────────┴───────────────────────────────────┘

### Step 2d: PDE-Refiner-based Training


### Step 3: Add more featuers ERA5 



### Step 4: Experiment: Longer Time Horizons 

### Step 5: Visualization 
We need to produce some sensible visualizations for the output feature forecast over the surface of earth. 
- [ ] Finish visualization scripts. 