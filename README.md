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
mv geopotential_5.625.deg.zip geopotential_500_5.625deg && cd geopotential_500_5.625deg
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