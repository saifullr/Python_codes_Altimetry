# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:59:29 2024

@author: hp
"""
# lOADING THE DATA 

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np

my_local_dir = 'C:/Academic La Rochelle/M2/Internship/Data/swot'
datasets = []
for file_path in Path(my_local_dir).rglob('*SWOT*.nc'):
    # Extract cycle and pass numbers from the filename
    parts = file_path.stem.split('_')  # Split the filename at underscores
    cycle_number = parts[5]  # The sixth part is the cycle number
    pass_number = parts[6]  # The seventh part is the pass number
    ds = xr.open_dataset(file_path).drop(['i_num_line', 'i_num_pixel'])
    ds = ds.assign_coords({'cycle': int(cycle_number), 'pass': int(pass_number)})
    datasets.append(ds)
    
data= xr.concat(datasets, dim='num_lines')
data = data.to_dataframe()
data = data.dropna()

def track_data(df): 
    tracks_by_pass = {}
    
    for pass_number in df['pass'].unique():
        pass_data = df[df['pass'] == pass_number]
        tracks_by_pass[pass_number] = pass_data
    return tracks_by_pass

tracks_by_pass = track_data(data)
#track189 = tracks_by_pass[189]
track230 = tracks_by_pass[230]
track258 = tracks_by_pass[258]
track467 = tracks_by_pass[467]
track495 = tracks_by_pass[495]
track536 = tracks_by_pass[536]
#%% SHOWING THE TRACK IN A MAP 
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
# ax.set_global()
#ax.stock_img()
#ax.plot(track189['longitude'], track189['latitude'], 'r.', label='Track 189', transform=ccrs.Geodetic())
ax.plot(track230['longitude'], track230['latitude'], 'b.', label='Track 230', transform=ccrs.Geodetic())
ax.plot(track258['longitude'], track258['latitude'], 'g.', label='Track 258', transform=ccrs.Geodetic())
ax.plot(track467['longitude'], track467['latitude'], 'c.', label='Track 467', transform=ccrs.Geodetic())  
ax.plot(track495['longitude'], track495['latitude'], 'y.', label='Track 495', transform=ccrs.Geodetic())
ax.plot(track536['longitude'], track536['latitude'], 'k.', label='Track 536', transform=ccrs.Geodetic())
ax.gridlines(draw_labels=True)
ax.legend()
ax.set_title("SWOT Satellite Tracks")
plt.show()


#%% Interpolate Bathymetry 

#Load Bathymetry 
def load_bathymetry(file_path):
   
    bathymetry_df = pd.read_csv(file_path, delimiter=' ', header=None, names=['longitude', 'latitude', 'depth'])
    return bathymetry_df
bathymetry_file = 'C:/Academic La Rochelle/M2/Internship/Data/MergedBathymetryPoldered.xyz'
bathymetry_data = load_bathymetry(bathymetry_file)

def interpolate_bathymetry(bathymetry_df, track_df):
    points = bathymetry_df[['longitude', 'latitude']].values
    values = bathymetry_df['depth'].values
    track_points = track_df[['longitude', 'latitude']].values
    track_df['depth'] = griddata(points, values, track_points, method='nearest')
    track_df = track_df[(track_df['depth'] > 0) & (track_df['depth'] < 100)]
    return track_df
track189 = interpolate_bathymetry(bathymetry_data, track189)
track230 = interpolate_bathymetry(bathymetry_data, track230)
track258 = interpolate_bathymetry(bathymetry_data, track258)
track467 = interpolate_bathymetry(bathymetry_data, track467)
track495 = interpolate_bathymetry(bathymetry_data, track495)
track536 = interpolate_bathymetry(bathymetry_data, track536)

#%% Calculate cross shore distance 
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
    "189": (21.99, 89.94),
    "258": (21.85, 89.82),
    "230": (20.50, 92.56),
    "467": (22.69, 91.29), 
    "495": (21.95, 88.7),
    "536": (22.09, 91.05)
}

def calculate_cross_shore_distance(track, track_id):
    ref_point = ref_points[track_id]
    distances = [round(haversine(lon, lat, ref_point[1], ref_point[0])) for lon, lat in zip(track['longitude'], track['latitude'])]
    track['cross_shore_distance'] = distances
    return track

track189 = calculate_cross_shore_distance(track189, "189")
track230 = calculate_cross_shore_distance(track230, "230")
track258 = calculate_cross_shore_distance(track258, "258")
track467 = calculate_cross_shore_distance(track467, "467")
track495 = calculate_cross_shore_distance(track495, "495")
track536 = calculate_cross_shore_distance(track536, "536")

#%%
def generate_track_analysis(track_df, track_name):
    # Filter for cross shore distance
    filtered_data = track_df[track_df['cross_shore_distance'] <= 75]

    # Mean ssha
    mean_ssha = filtered_data.groupby('cross_shore_distance')['ssha'].mean()

    # Bathymetry data
    filtered_bathymetry = filtered_data.groupby('cross_shore_distance')['depth'].mean()

    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 16))

    # Scatter plot for ssha
    sc = axs[0].scatter(filtered_data['cycle'], filtered_data['cross_shore_distance'],
                        c=filtered_data['ssha'], cmap='jet', marker='.')
    axs[0].set_title(f'{track_name} - SSHA')
    axs[0].set_xlabel('Cycle')
    axs[0].set_ylabel('Distance to coast (km)')
    axs[0].set_yticks(np.arange(0, 80,10))
    axs[0].grid(True)
    plt.colorbar(sc, ax=axs[0]).set_label('SSHA (m)')

    # Mean SSHA plot for specific cycles
    axs[1].plot(  mean_ssha, mean_ssha.index,'r-', label='Mean SSHA')
    axs[1].set_title(f'{track_name} - Mean SSHA')
    axs[1].set_xlabel('Mean SSHA (m)')
    axs[1].set_ylabel('Distance to coast (km)')
    axs[1].set_yticks(np.arange(0, 80,10))
    axs[1].grid(True)
    axs[1].legend(loc='best')
    # Bathymetry plot
    axs[2].plot(filtered_bathymetry.values, filtered_bathymetry.index, 'k--', label='Bathymetry')
    axs[2].set_title(f'{track_name} - Bathymetry')
    axs[2].set_xlabel('Depth (m)')
    axs[2].set_ylabel('Distance to coast (km)')
    axs[2].set_yticks(np.arange(0, 80,10))
    axs[2].grid(True)
    axs[2].legend(loc='best')

    plt.suptitle(f"{track_name} Analysis")
    plt.tight_layout()
    plt.show()
generate_track_analysis(track189, "Track 189" )
generate_track_analysis(track230, "Track 230")
generate_track_analysis(track258, "Track 258")
generate_track_analysis(track467, "Track 467")
generate_track_analysis(track495, "Track 495")
generate_track_analysis(track536, "Track 536")



#%%

import xarray as xr 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import pandas as pd 
import numpy as np
from scipy.interpolate import griddata

def load_data(file_path):
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe()
    df = df.dropna()

    return df

# Example usage
file_path1 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t231.nc'
file_path2 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t053.nc'
file_path3 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t014.nc'
file_path4 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t090.nc'
track231 = load_data(file_path1)
track053 = load_data(file_path2)
track014 = load_data(file_path3)
track090 = load_data(file_path4)

def load_bathymetry(file_path):
   
    bathymetry_df = pd.read_csv(file_path, delimiter=' ', header=None, names=['longitude', 'latitude', 'depth'])
    return bathymetry_df
bathymetry_file = 'C:/Academic La Rochelle/M2/Internship/Data/MergedBathymetryPoldered.xyz'
bathymetry_data = load_bathymetry(bathymetry_file)

def interpolate_bathymetry(bathymetry_df, track_df):
    points = bathymetry_df[['longitude', 'latitude']].values
    values = bathymetry_df['depth'].values
    track_points = track_df[['longitude_theoretical', 'latitude_theoretical']].values
    track_df['depth'] = griddata(points, values, track_points, method='nearest')
    track_df = track_df[(track_df['depth'] > 0) & (track_df['depth'] < 100)]
    return track_df

track231 = interpolate_bathymetry(bathymetry_data, track231)
track053 = interpolate_bathymetry(bathymetry_data, track053)
track014 = interpolate_bathymetry(bathymetry_data, track014)
track090 = interpolate_bathymetry(bathymetry_data, track090)


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
    "231": (21.616125, 88.308353),
    "053": (22.625739, 91.605314),
    "014": (21.604354, 87.450514),
    "090": (21.803961, 90.196795)
}

def calculate_cross_shore_distance(track, track_id):
    ref_point = ref_points[track_id]
    distances = [round(haversine(lon, lat, ref_point[1], ref_point[0])) for lon, lat in zip(track['longitude_theoretical'], track['latitude_theoretical'])]
    track['cross_shore_distance'] = distances
    return track

track231 = calculate_cross_shore_distance(track231, "231")
track053 = calculate_cross_shore_distance(track053, "053")
track014 = calculate_cross_shore_distance(track014, "014")
track090 = calculate_cross_shore_distance(track090, "090")


def generate_track_analysis(track_df, track_name):
    # Common filtering for all metrics
    track_df = track_df[track_df['validation_flag'] != 1]
    track_filtered = track_df.where((track_df.cycle >= 140) & (track_df.cycle <= 190) & (track_df.cross_shore_distance <= 75))
    track_filtered1 = track_df.where((track_df.cycle >= 40) & (track_df.cycle <= 90) & (track_df.cross_shore_distance <= 75))
    
    filtered_bathymetry = track_df[track_df['cross_shore_distance'] <= 75]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    return track_df 

generate_track_analysis(track053, "Track 053")
generate_track_analysis(track090, "Track 090")
generate_track_analysis(track014, "Track 014")
generate_track_analysis(track231, "Track 231")


#%% 
def calculate_mean_ssha(track_df, ssha_column_name):
    # Filter for cross shore distance <= 75 km
    
    filtered_data = track_df[track_df['cross_shore_distance'] <= 75]
    
    # Group by 'cross_shore_distance' and calculate the mean
    mean_ssha = filtered_data.groupby('cross_shore_distance')[ssha_column_name].mean()
    return mean_ssha

# Calculate mean SSHA for each track
mean_ssha_536 = calculate_mean_ssha(track536, 'ssha')
mean_ssha_495 = calculate_mean_ssha(track495, 'ssha')
mean_ssha_467 = calculate_mean_ssha(track467, 'ssha')
mean_ssha_258 = calculate_mean_ssha(track258, 'ssha')
mean_ssha_230 = calculate_mean_ssha(track230, 'ssha')
mean_ssha_189 = calculate_mean_ssha(track189, 'ssha')
mean_ssha_231 = calculate_mean_ssha(track231, 'sea_level_anomaly')
mean_ssha_053 = calculate_mean_ssha(track053, 'sea_level_anomaly')
mean_ssha_014 = calculate_mean_ssha(track014, 'sea_level_anomaly')
mean_ssha_090 = calculate_mean_ssha(track090, 'sea_level_anomaly')


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_ssha_495.index, mean_ssha_495.values, label='Track 495 (SWOT)', marker='o')
plt.plot(mean_ssha_231.index, mean_ssha_231.values, label='Track 231 (JASON-3)', marker='x')
plt.xlabel('Cross Shore Distance (km)')
plt.ylabel('Mean Sea Level Anomaly (m)')
plt.title('Mean Sea Level Anomaly for Track 495 and Track 231')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Create the subplot grid
fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(30, 20))

# Define the main tracks and the tracks to compare against
main_tracks = [track189, track230, track258, track467, track495, track536]
main_track_labels = ['189', '230', '258', '467', '495', '536']
compare_tracks = [track231, track053, track014, track090]
compare_track_labels = ['231', '053', '014', '090']

# Iterate over the grid
for i, main_track in enumerate(main_tracks):
    for j, compare_track in enumerate(compare_tracks):
        ax = axs[j, i]

        # Determine the correct column name for sea level anomaly
        column_name_main = 'sea_level_anomaly' if main_track_labels[i] in ['231', '053', '014', '090'] else 'ssha'
        column_name_compare = 'sea_level_anomaly' if compare_track_labels[j] in ['231', '053', '014', '090'] else 'ssha'

        # Calculate mean SSHA for each track
        mean_ssha_main = calculate_mean_ssha(main_track, column_name_main)
        mean_ssha_compare = calculate_mean_ssha(compare_track, column_name_compare)

        # Plotting
        ax.plot(mean_ssha_main.index, mean_ssha_main.values, label=f'Track {main_track_labels[i]} (Swot)', marker='o')
        ax.plot(mean_ssha_compare.index, mean_ssha_compare.values, label=f'Track {compare_track_labels[j]} (Jason-3)', marker='x')
        
        # Set x and y labels only for specific subplots
        if j == 3:  # Only the last row
            ax.set_xlabel('Cross Shore Distance (km)')
        if i == 0:  # Only the first column
            ax.set_ylabel('Mean Sea Level Anomaly (m)')

       
       
        ax.legend()
        ax.grid(True)
plt.suptitle("Comparison of Mean Sea Level Anomaly Across Different Tracks")
plt.tight_layout()
plt.show()
