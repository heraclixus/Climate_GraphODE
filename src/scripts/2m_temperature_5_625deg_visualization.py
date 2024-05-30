'''
Utility Functions
'''

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import ceil, floor

import torch
import torch.nn as nn
import torch.fft as fft


import torch_harmonics as th
from torch_harmonics import *
from torch_harmonics.sht import *
from torch_harmonics.examples import ShallowWaterSolver

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys

# loads the dataset as an xarray
def load_dataset(dataset_path):

  ds = xr.open_dataset(dataset_path)

  temperature = ds['t2m']
  temperature = temperature - 273.15 # Convert Kelvin to Celsius
  lat = ds['lat'].values
  lon = ds['lon'].values

  return temperature, lat, lon

# adapted from torch-harmonics/notebooks/getting_started.ipynb
# visualizes the spherical harmonic transform for a specified hour
def visualize_spherical_harmonic_transform(_data, lon, lat, hour=0):

  sys.path.append("../")
  cmap = 'turbo'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data = nn.functional.interpolate(torch.from_numpy(_data.isel(time=hour).values).unsqueeze(0).unsqueeze(0), size=(512,2*512)).squeeze()
  signal = data.to(device)
  n_theta = data.shape[0]
  n_lambda = data.shape[1]
  sht = RealSHT(n_theta, n_lambda, grid="equiangular").to(device)
  isht = InverseRealSHT(n_theta, n_lambda, grid="equiangular").to(device)
  coeffs = sht(signal)


  lon = np.linspace(-np.pi, np.pi, n_lambda)
  lat = np.linspace(np.pi/2., -np.pi/2., n_theta)
  Lon, Lat = np.meshgrid(lon, lat)
  fig = plt.figure(figsize=(22, 5))
  ax = fig.add_subplot(1, 2, 2, projection='mollweide')
  plt.pcolormesh(Lon, Lat, isht(coeffs).cpu(), cmap=cmap)
  ax.set_title("Temperature map")
  ax.grid(True)
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  plt.colorbar()
  plt.show()

  return coeffs

def animate(_data, lon, lat, time_range=range(10)):

  # animates the time_range specified, in hours
  # for example range(10) animates first 10 hours
  temperature_visual = []
  for i in time_range:
    temp = _data.isel(time=i).values
    data = nn.functional.interpolate(torch.from_numpy(temp).unsqueeze(0).unsqueeze(0), size=(512,2*512)).squeeze()
    temperature_visual.append(data)

  temperature_visual = np.array(temperature_visual)

  if torch.is_tensor(temperature_visual):
      if temperature_visual.is_cuda:
          temperature_visual = temperature_visual.cpu()
  else:
      # Convert NumPy array to tensor and move to CPU
      temperature_visual = torch.from_numpy(temperature_visual).cpu()

  nlat = 512
  nlon = 2*nlat
  lmax = ceil(128)
  mmax = lmax
  dt = 75

  # ShallowWaterSolver has a function plot_griddata that plots 2D data to the
  # surface of a sphere
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  swe_solver = ShallowWaterSolver(nlat, nlon, dt, lmax=lmax, mmax=mmax).to(device)

  # Visualizing the first ten hours of 2018
  fig = plt.figure(figsize=(8, 6), dpi=72)
  moviewriter = animation.writers['pillow'](fps=5)
  moviewriter.setup(fig, './temperature_animation.gif', dpi=72)

  for i in range(temperature_visual.shape[0]):
      plt.clf()
      swe_solver.plot_griddata(temperature_visual[i], fig, cmap="twilight_shifted", antialiased=False)
      plt.draw()
      moviewriter.grab_frame()

  moviewriter.finish()

def main():

  # ignore if you did not upload dataset to Google Drive
  from google.colab import drive
  drive.mount('/content/drive')

  dataset_path = '/content/drive/MyDrive/Climate_GraphODE/data/2m_temperature_5.625deg/2m_temperature_2018_5.625deg.nc'

  temperature, lon, lat = load_dataset(dataset_path)
  coefficients = visualize_spherical_harmonic_transform(temperature, lon, lat, hour=0)
  # print(coefficients)
  animate(temperature, lon, lat, time_range=range(10))

if __name__ == '__main__':
    main()