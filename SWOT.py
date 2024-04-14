# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:46:01 2024

@author: hp
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np 

my_local_dir = 'C:/Academic La Rochelle/M2/Internship/Data/swot'
dataset= []
for file_path in Path(my_local_dir).rglob('*SWOT*.nc'):
    dataset.append(file_path)    
datasets = []
for fp in dataset:
    ds = xr.open_dataset(fp)
    ds = ds.drop(['i_num_line', 'i_num_pixel'])  # Dropping the variables
    datasets.append(ds)


track536 = xr.concat(datasets, dim='num_lines')
track536= track536.to_dataframe()
track536 = track536.dropna()

#%%
def filter_data_by_coordinates(ds, lat_min, lat_mplt, lon_min, lon_mplt):
    # Use .where to filter the dataset based on condition
    filtered_ds = ds.where(
        (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_mplt) &
        (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_mplt),
      
    )
    return filtered_ds

lat_min, lat_mplt = -19, 25  # 19 S to 25 N
lon_min, lon_mplt = 75, 110  # 75 E to 110 E

# Apply the filter to the dataset
track536= filter_data_by_coordinates(track536, lat_min, lat_mplt, lon_min, lon_mplt)

#%%
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.set_global()
ax.plot(track536['longitude'], track536['latitude'], 'r.', label='Track 536', transform=ccrs.Geodetic())
ax.gridlines(draw_labels=True)
ax.legend()
ax.set_title(" SWOT Satellite Tracks")
plt.show()

#%%
def load_bathymetry(file_path):
   
    bathymetry_df = pd.read_csv(file_path, delimiter=' ', header=None, names=['longitude', 'latitude', 'depth'])
    return bathymetry_df
bathymetry_file = 'C:/Academic La Rochelle/M2/Internship/Data/MergedBathymetryPoldered.xyz'
bathymetry_data = load_bathymetry(bathymetry_file)

track536 = track536.dropna(subset=['longitude', 'latitude'])
def interpolate_bathymetry(bathymetry_df, track_df):
    points = bathymetry_df[['longitude', 'latitude']].values
    values = bathymetry_df['depth'].values
    track_points = track_df[['longitude', 'latitude']].values
    track_df['depth'] = griddata(points, values, track_points, method='nearest')
    track_df = track_df[(track_df['depth'] > 0) & (track_df['depth'] < 100)]
    return track_df

track536 = interpolate_bathymetry(bathymetry_data, track536)

#%%
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in km
    return c * r

# Reference points for each track
ref_points = {
    "536": (22, 91)
}

def calculate_cross_shore_distance(track, track_id):
    ref_point = ref_points[track_id]
    distances = [round(haversine(lon, lat, ref_point[1], ref_point[0])) for lon, lat in zip(track['longitude'], track['latitude'])]
    track['cross_shore_distance'] = distances
    return track

track536 = calculate_cross_shore_distance(track536, "536")


#%%
fig, ax1 = plt.subplots(figsize=(12, 6))
filtered_data = track536[track536['cross_shore_distance'] <= 130]
grouped_data = filtered_data.groupby('cross_shore_distance')['ssha'].mean()
ax1.plot(grouped_data.index, grouped_data.values, 'b', label='ssha')
ax1.set_xlabel('Cross Shore Distance (kms)')
ax1.set_xlim(0, 130)
ax1.set_xticks(np.arange(0, 131, 10))
ax1.set_ylabel('Sea Surface Height anomaly (m)')
ax1.invert_xaxis() 
ax1.legend(loc=2) 
ax1.grid(True)

# Create a second y-axis for the bathymetry data
ax2 = ax1.twinx()
filtered_bathymetry = track536[track536['cross_shore_distance'] <= 130]
grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')
ax2.set_ylabel('Depth (m)')
ax2.invert_yaxis()  # Inverting the y-axis of the secondary y-axis
ax2.legend(loc=3)

# Set the title of the plot
ax1.set_title('Mean Sea Surface Height Anomaly over the period (Track 536)')

#%%
track_filtered = track536.where(track536.cross_shore_distance <= 75).dropna()
mean_sea_level_anomaly = track_filtered.groupby('cross_shore_distance')['ssha'].mean()
distances = mean_sea_level_anomaly.index

filtered_bathymetry = track_filtered[track_filtered['cross_shore_distance'] <= 75]
grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()


fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 12))
sc1 = ax1.scatter(track_filtered['time'], track_filtered['cross_shore_distance'],
                  c=track_filtered['ssha'], cmap='jet', marker='.')

ax1.set_title('Sea Surface Height Anomaly (Track 536) ')
ax1.set_xlabel('time')
ax1.set_ylabel('Distance to coast (km)')
ax1.set_yticks(np.arange(0, 80, 10))
ax1.tick_params(axis='x', rotation=45)
ax1.grid(which='both', linestyle='--', linewidth=0.5)

cbar1 = plt.colorbar(sc1, ax=ax1)
cbar1.set_label('Sea Surface Height Anomaly (m)')


# Plot mean values
ax2.plot(mean_sea_level_anomaly, distances, label='Mean SSHA', marker='.')
ax2.set_title('Mean Sea Level Anomaly')
ax2.set_xlabel('Mean Value')
ax2.set_ylabel('Distance to coast (km)')
ax2.set_yticks(np.arange(0, 80, 10))
ax2.grid(True)
ax2.legend(loc=1)

ax3.plot( grouped_bathymetry.values, grouped_bathymetry.index,'k--', label='Bathymetry') 
ax3.set_xlabel('Bathymetry') 
ax3.set_ylabel('Distance to coast (km)')
ax3.set_xlabel('Depth (meters)')
ax3.set_yticks(np.arange(0, 80, 5))
ax3.grid(True)
ax3.legend()
plt.suptitle(f" SWOT Track 536 Sea Surface Height Anomaly")

plt.tight_layout()
plt.show()

plt.show()

