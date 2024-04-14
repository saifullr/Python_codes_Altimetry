# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:56:52 2024

@author: hp
"""
# Loading Data 

import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# Load along Track data 
def load_data(file_path):
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe()
    df = df.dropna()
    df = df[df['validation_flag'] != 1]
    return df

file_path1 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t231.nc'
file_path2 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t053.nc'
file_path3 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t014.nc'
file_path4 = 'C:/Academic La Rochelle/M2/Internship/Data/dt_coastal_j3_phy_20hz_t090.nc'


track231 = load_data(file_path1)
track053 = load_data(file_path2)
track014 = load_data(file_path3)
track090 = load_data(file_path4)


#%%
# Keep only BOB part 
def filter_data_by_coordinates(df, lat_min, lat_max, lon_min, lon_max):
    
    filtered_df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
                     (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]
    return filtered_df

lat_min, lat_max = -19, 25  # 19 S to 25 N
lon_min, lon_max = 75, 110  # 75 E to 110 E

# Apply the filter to each dataset
track231 = filter_data_by_coordinates(track231, lat_min, lat_max, lon_min, lon_max)
track053 = filter_data_by_coordinates(track053, lat_min, lat_max, lon_min, lon_max)
track014 = filter_data_by_coordinates(track014, lat_min, lat_max, lon_min, lon_max)
track090 = filter_data_by_coordinates(track090, lat_min, lat_max, lon_min, lon_max)

#Load Bathymetry 
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
#%% Plot the tracks 
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
# Plot each track
ax.plot(track231['longitude'], track231['latitude'], 'ro', label='Track 231')
ax.plot(track053['longitude'], track053['latitude'], 'bo', label='Track 053')
ax.plot(track014['longitude'], track014['latitude'], 'go', label='Track 014')
ax.plot(track090['longitude'], track090['latitude'], 'yo', label='Track 090')
ax.legend()
ax.gridlines(draw_labels=True)
plt.title("Satellite Tracks")
plt.show()
#%% Calculate the cross-shore distance using Haversine formula 

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

#%% Track 231 monthly mean plotting with bathymetry 

track231['time'] = pd.to_datetime(track231['time'])
track231['ssh'] = track231['sea_level_anomaly'] + track231['mdt']
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']


plt.figure(figsize=(18, 12))

# Loop through each plot type (SWH, SLA, SSH)
for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  
    
    for month in range(1, 13):
        monthly_data = track231[track231['time'].dt.month == month]
        monthly_data = monthly_data[monthly_data['cross_shore_distance'] <= 130]
        grouped_data = monthly_data.groupby('cross_shore_distance')[data_type].mean()
        ax1.plot(grouped_data.index, grouped_data.values, color=colors[month-1], label=month_names[month-1])
    
    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} for Each Month (Track 231)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)
    ax1.legend(ncol=12, loc='upper center', bbox_to_anchor=(0.5, 1))

   
    ax2 = ax1.twinx()
    filtered_bathymetry = track231[track231['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')  
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()

plt.tight_layout()
plt.show()

#%% Mean over the period 
plt.figure(figsize=(18, 12))

for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  

    filtered_data = track231[track231['cross_shore_distance'] <= 130]
    grouped_data = filtered_data.groupby('cross_shore_distance')[data_type].mean()
    ax1.plot(grouped_data.index, grouped_data.values, 'r-')

    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} with filtering (Track 231)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)

    ax2 = ax1.twinx()
    filtered_bathymetry = track231[track231['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')  
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()
    ax2.legend()

plt.tight_layout()
plt.show()

#%% Track053

track053['time'] = pd.to_datetime(track053['time'])
track053['ssh'] = track053['sea_level_anomaly'] + track053['mdt']



plt.figure(figsize=(18, 12))

# Loop through each plot type (SWH, SLA, SSH)
for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  
    
    for month in range(1, 13):
        monthly_data = track053[track053['time'].dt.month == month]
        monthly_data = monthly_data[monthly_data['cross_shore_distance'] <= 130]
        grouped_data = monthly_data.groupby('cross_shore_distance')[data_type].mean()
        ax1.plot(grouped_data.index, grouped_data.values, color=colors[month-1], label=month_names[month-1])
    
    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} for Each Month (Track 053)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)
    ax1.legend(ncol=12, loc='upper center', bbox_to_anchor=(0.5, 1))

   
    ax2 = ax1.twinx()
    filtered_bathymetry = track053[track053['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()

plt.tight_layout()
plt.show()

#%% Mean over the period 
plt.figure(figsize=(18, 12))

for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  

    filtered_data = track053[track053['cross_shore_distance'] <= 130]
    grouped_data = filtered_data.groupby('cross_shore_distance')[data_type].mean()
    ax1.plot(grouped_data.index, grouped_data.values, 'b-', marker='.')  

    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} with filtering (Track 053)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)

    ax2 = ax1.twinx()
    filtered_bathymetry = track053[track053['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry') 
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()
    ax2.legend()

plt.tight_layout()
plt.show()

#%%
def event_track(track, track_name): 
    target_heights = [5, 4, 3, 2, 1] 
    tolerance = 0.5 
    plt.figure(figsize=(18, 12))
    for i, target_height in enumerate(target_heights, start=1):
        plt.subplot(5, 1, i)  
        ax = plt.gca()  
        target_events = track[(track['swh'] >= target_height - tolerance) & 
                                 (track['swh'] <= target_height + tolerance)]
    
        if not target_events.empty:
            first_event_time = target_events['time'].iloc[0]
            event_duration = pd.Timedelta(hours=0.1) 
            event_data = track[(track['time'] >= first_event_time) & 
                                  (track['time'] < first_event_time + event_duration)]
    
            mean_swh = event_data.groupby('cross_shore_distance')['swh'].mean()
    
            ax.plot(mean_swh.index, mean_swh.values, 'b-', marker='.', label=f'{target_height}m wave')
        else:
            ax.plot([], [], 'b-', marker='o', label=f'{target_height}m wave (no data)')
    
        ax.set_xlabel('Cross Shore Distance (kms)')
        ax.set_ylabel('Mean SWH (meters)')
        ax.set_title(f'Mean SWH Over the Period of a Specific {target_height}m Wave Event ({track_name})')
        ax.grid(True)
        ax.invert_xaxis()
        ax.legend()
    
    plt.tight_layout()
    plt.show()

event_track(track053[track053['cross_shore_distance'] <= 250], "Track053")
event_track(track014[track014['cross_shore_distance'] <= 250], "Track014")
event_track(track090[track090['cross_shore_distance'] <= 250], "Track090")
event_track(track231[track231['cross_shore_distance'] <= 250], "Track231")



#%% Track 014

track014['time'] = pd.to_datetime(track014['time'])
track014['ssh'] = track014['sea_level_anomaly'] + track014['mdt']
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']


plt.figure(figsize=(18, 12))

# Loop through each plot type (SWH, SLA, SSH)
for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  
    
    for month in range(1, 13):
        monthly_data = track014[track014['time'].dt.month == month]
        monthly_data = monthly_data[monthly_data['cross_shore_distance'] <= 130]
        grouped_data = monthly_data.groupby('cross_shore_distance')[data_type].mean()
        ax1.plot(grouped_data.index, grouped_data.values, color=colors[month-1], label=month_names[month-1])
    
    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} for Each Month (Track 014)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)
    ax1.legend(ncol=12, loc='upper center', bbox_to_anchor=(0.5, 1))

   
    ax2 = ax1.twinx()
    filtered_bathymetry = track014[track014['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()

plt.tight_layout()
plt.show()

#%% Mean over the period 
plt.figure(figsize=(18, 12))

for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  

    filtered_data = track014[track014['cross_shore_distance'] <= 130]
    grouped_data = filtered_data.groupby('cross_shore_distance')[data_type].mean()
    ax1.plot(grouped_data.index, grouped_data.values, 'g-', marker='.')  

    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} with filtering (Track 014)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)

    ax2 = ax1.twinx()
    filtered_bathymetry = track014[track014['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')  
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()
    ax2.legend()

plt.tight_layout()
plt.show()

#%% Track 090 monthly mean plotting with bathymetry 

track090['time'] = pd.to_datetime(track090['time'])
track090['ssh'] = track090['sea_level_anomaly'] + track090['mdt']
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']


plt.figure(figsize=(18, 12))

# Loop through each plot type (SWH, SLA, SSH)
for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  
    
    for month in range(1, 13):
        monthly_data = track090[track090['time'].dt.month == month]
        monthly_data = monthly_data[monthly_data['cross_shore_distance'] <= 130]
        grouped_data = monthly_data.groupby('cross_shore_distance')[data_type].mean()
        ax1.plot(grouped_data.index, grouped_data.values, color=colors[month-1], label=month_names[month-1])
    
    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} for Each Month (Track 090)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)
    ax1.legend(ncol=12, loc='upper center', bbox_to_anchor=(0.5, 1))

   
    ax2 = ax1.twinx()
    filtered_bathymetry = track090[track090['cross_shore_distance'] <= 130]
    ax2.plot(filtered_bathymetry['cross_shore_distance'], filtered_bathymetry['depth'], 'r--')  
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()

plt.tight_layout()
plt.show()

#%% Mean over the period 
plt.figure(figsize=(18, 12))

for i, data_type in enumerate(['swh', 'sea_level_anomaly', 'ssh'], start=1):
    plt.subplot(3, 1, i)  
    ax1 = plt.gca()  

    filtered_data = track090[track090['cross_shore_distance'] <= 130]
    grouped_data = filtered_data.groupby('cross_shore_distance')[data_type].mean()
    ax1.plot(grouped_data.index, grouped_data.values, 'coral', marker='.') 

    ax1.set_xlabel('Cross Shore Distance (kms)')
    ax1.set_ylabel(f'Mean {data_type} (meters)')
    ax1.set_title(f'Mean {data_type.replace("_", " ").title()} (Track 090)')
    ax1.set_xlim(0, 130)
    ax1.invert_xaxis()
    ax1.grid(True)

    ax2 = ax1.twinx()
    filtered_bathymetry = track090[track090['cross_shore_distance'] <= 130]
    grouped_bathymetry = filtered_bathymetry.groupby('cross_shore_distance')['depth'].mean()
    ax2.plot(grouped_bathymetry.index, grouped_bathymetry.values, 'k--', label='Bathymetry')
    ax2.set_ylabel('Depth (meters)')
    ax2.invert_yaxis()
    ax2.legend()

plt.tight_layout()
plt.show()