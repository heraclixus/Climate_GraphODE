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

- [ ] Modify the dataset class to incorporate time series.  
- [ ] Utility function to generate graph for the dataset.


### Step ab: Incorporate the FNO model to use the setup 
The Fourier Neural Operator and neural operator models in general can be used in this case.
- [ ] Use FNO for one-step prediction. 


### Step 4: Visualization 
We need to produce some sensible visualizations for the output feature forecast over the surface of earth. 
- [ ] Finish visualization scripts. 