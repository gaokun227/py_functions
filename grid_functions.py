import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # required python/3.10
import cartopy.feature as cfeature
from netCDF4 import Dataset

# Function to read grid data from a file
def read_grid(file_path):
    with Dataset(file_path, 'r') as dataset:
        lat = dataset.variables['y'][:]
        lon = dataset.variables['x'][:]
    return lat, lon

# Function to plot grid points
def plot_grid(ax, lat, lon, color, marker_size, sel):
    nx, ny = lat.shape
    dx = dy = sel * 2
    idx_list = np.arange(0, nx, dx) #- 1 + dx, dx)
    idy_list = np.arange(0, ny, dy) #- 1 + dy, dy)
    
    for i in idx_list:
        ax.scatter(lon[i, :], lat[i, :], color=color, transform=ccrs.PlateCarree(), s=marker_size)
    for i in idy_list:
        ax.scatter(lon[:, i], lat[:, i], color=color, transform=ccrs.PlateCarree(), s=marker_size)

# Function to plot grid edges
def plot_edges(ax, lat, lon, edge_color, edge_size):
    edges = [(lon[0, :], lat[0, :]), (lon[:, 0], lat[:, 0]), (lon[:, -1], lat[:, -1]), (lon[-1, :], lat[-1, :])]
    for lon_edge, lat_edge in edges:
        ax.scatter(lon_edge, lat_edge, color=edge_color, transform=ccrs.PlateCarree(), s=edge_size)
