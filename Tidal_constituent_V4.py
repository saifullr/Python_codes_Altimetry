# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:54:20 2024

@author: hp
"""

import xarray as xr
import numpy as np
import utide 
import pandas as pd 
from utide import solve, reconstruct 


# Open the dataset
model_ds = xr.open_mfdataset("D:/Bob_SCHISM_ERA5/era_bob_2018_*.nc")


lats = [18.980617, 18.980617, 19.063708, 19.157136]
lons = [87.260604, 88.952498, 91.07928, 92.435164]

def find_nearest_node(lat, lon, lat_array, lon_array):
    dist_sq = (lat_array - lat)**2 + (lon_array - lon)**2
    min_dist_index = dist_sq.argmin()
    return min_dist_index


# Arrays of grid latitudes and longitudes
grid_latitudes = model_ds['SCHISM_hgrid_node_y'].values
grid_longitudes = model_ds['SCHISM_hgrid_node_x'].values


#%%

# Find the nearest node index for the first pair of lat and lon
nearest_node_index = find_nearest_node(lats[0], lons[0], grid_latitudes, grid_longitudes)

# Now select all data for this nearest node index
ctg= model_ds.isel(nSCHISM_hgrid_node=nearest_node_index)
ctg.SCHISM_hgrid_node_x.values, ctg.SCHISM_hgrid_node_y.values

latitude =lats[0]
elevation = ctg['elevation'].values
time =  pd.to_datetime(ctg['time'].values)

coef_ctg= utide.solve(time, elevation, lat=latitude, method='ols', conf_int='linear')

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2019-12-31')

prediction_time = pd.date_range(start=start_date, end=end_date, freq='H')  

prediction_ctg = reconstruct(prediction_time, coef_ctg)



if 'M2' in coef_ctg['name']:
    m2_index = np.where(coef_ctg['name'] == 'M2')[0][0]
    m2_amplitude = coef_ctg['A'][m2_index]
    m2_phase = coef_ctg['g'][m2_index]
    print(f"M2 Amplitude for (lat:{lats[0]}, long:{lons[0]}) : {m2_amplitude}")
    print(f"M2 Phase for (lat:{lats[0]}, long:{lons[0]}) : {m2_phase}")
else:
    print("M2 constituent not found in analysis.")
    
if 'S2' in coef_ctg['name']:
    s2_index = np.where(coef_ctg['name'] == 'S2')[0][0]
    s2_amplitude = coef_ctg['A'][s2_index]
    s2_phase = coef_ctg['g'][s2_index]
    print(f"s2 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {s2_amplitude}")
    print(f"s2 Phase for (lat:{lats[0]}, long:{lons[0]}): {s2_phase}")
else:
    print("s2 constituent not found in analysis.")

if 'K1' in coef_ctg['name']:
    k1_index = np.where(coef_ctg['name'] == 'K1')[0][0]
    k1_amplitude = coef_ctg['A'][k1_index]
    k1_phase = coef_ctg['g'][k1_index]
    print(f"k1 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {k1_amplitude}")
    print(f"k1 Phase for (lat:{lats[0]}, long:{lons[0]}): {k1_phase}")
else:
    print("k1 constituent not found in analysis.")
    
if 'O1' in coef_ctg['name']:
    O1_index = np.where(coef_ctg['name'] == 'O1')[0][0]
    O1_amplitude = coef_ctg['A'][O1_index]
    O1_phase = coef_ctg['g'][O1_index]
    print(f"O1 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {O1_amplitude}")
    print(f"O1 Phase for (lat:{lats[0]}, long:{lons[0]}): {O1_phase}")
else:
    print("O1 constituent not found in analysis.")
    
if 'N2' in coef_ctg['name']:
    N2_index = np.where(coef_ctg['name'] == 'N2')[0][0]
    N2_amplitude = coef_ctg['A'][N2_index]
    N2_phase = coef_ctg['g'][N2_index]
    print(f"N2 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {N2_amplitude}")
    print(f"N2 Phase for (lat:{lats[0]}, long:{lons[0]}): {N2_phase}")
else:
    print("N2 constituent not found in analysis.")    
    
if 'K2' in coef_ctg['name']:
    K2_index = np.where(coef_ctg['name'] == 'K2')[0][0]
    K2_amplitude = coef_ctg['A'][K2_index]
    K2_phase = coef_ctg['g'][K2_index]
    print(f"K2 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {K2_amplitude}")
    print(f"K2 Phase for (lat:{lats[0]}, long:{lons[0]}): {K2_phase}")
else:
    print("K2 constituent not found in analysis.")
if 'P1' in coef_ctg['name']:
    P1_index = np.where(coef_ctg['name'] == 'P1')[0][0]
    P1_amplitude = coef_ctg['A'][P1_index]
    P1_phase = coef_ctg['g'][P1_index]
    print(f"P1 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {P1_amplitude}")
    print(f"P1 Phase for (lat:{lats[0]}, long:{lons[0]}): {P1_phase}")
else:
    print("P1 constituent not found in analysis.")
    
if 'Q1' in coef_ctg['name']:
    Q1_index = np.where(coef_ctg['name'] == 'Q1')[0][0]
    Q1_amplitude = coef_ctg['A'][Q1_index]
    Q1_phase = coef_ctg['g'][Q1_index]
    print(f"Q1 Amplitude for (lat:{lats[0]}, long:{lons[0]}): {Q1_amplitude}")
    print(f"Q1 Phase for (lat:{lats[0]}, long:{lons[0]}): {Q1_phase}")
else:
    print("Q1 constituent not found in analysis.")
    
#%%

# Find the nearest node index for the first pair of lat and lon
nearest_node_index = find_nearest_node(lats[1], lons[1], grid_latitudes, grid_longitudes)

# Now select all data for this nearest node index
ctg1= model_ds.isel(nSCHISM_hgrid_node=nearest_node_index)
ctg1.SCHISM_hgrid_node_x.values, ctg1.SCHISM_hgrid_node_y.values

latitude =lats[1]
elevation = ctg1['elevation'].values
time =  pd.to_datetime(ctg1['time'].values)

coef_ctg1= utide.solve(time, elevation, lat=latitude, method='ols', conf_int='linear')

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2019-12-31')

prediction_time = pd.date_range(start=start_date, end=end_date, freq='H')  

prediction_ctg1 = reconstruct(prediction_time, coef_ctg1)



if 'M2' in coef_ctg1['name']:
    m2_index = np.where(coef_ctg1['name'] == 'M2')[0][0]
    m2_amplitude = coef_ctg1['A'][m2_index]
    m2_phase = coef_ctg1['g'][m2_index]
    print(f"M2 Amplitude for (lat:{lats[1]}, long:{lons[1]}) : {m2_amplitude}")
    print(f"M2 Phase for (lat:{lats[1]}, long:{lons[1]}) : {m2_phase}")
else:
    print("M2 constituent not found in analysis.")
    
if 'S2' in coef_ctg1['name']:
    s2_index = np.where(coef_ctg1['name'] == 'S2')[0][0]
    s2_amplitude = coef_ctg1['A'][s2_index]
    s2_phase = coef_ctg1['g'][s2_index]
    print(f"s2 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {s2_amplitude}")
    print(f"s2 Phase for (lat:{lats[1]}, long:{lons[1]}): {s2_phase}")
else:
    print("s2 constituent not found in analysis.")

if 'K1' in coef_ctg1['name']:
    k1_index = np.where(coef_ctg1['name'] == 'K1')[0][0]
    k1_amplitude = coef_ctg1['A'][k1_index]
    k1_phase = coef_ctg1['g'][k1_index]
    print(f"k1 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {k1_amplitude}")
    print(f"k1 Phase for (lat:{lats[1]}, long:{lons[1]}): {k1_phase}")
else:
    print("k1 constituent not found in analysis.")
    
if 'O1' in coef_ctg1['name']:
    O1_index = np.where(coef_ctg1['name'] == 'O1')[0][0]
    O1_amplitude = coef_ctg1['A'][O1_index]
    O1_phase = coef_ctg1['g'][O1_index]
    print(f"O1 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {O1_amplitude}")
    print(f"O1 Phase for (lat:{lats[1]}, long:{lons[1]}): {O1_phase}")
else:
    print("O1 constituent not found in analysis.")
    
if 'N2' in coef_ctg1['name']:
    N2_index = np.where(coef_ctg1['name'] == 'N2')[0][0]
    N2_amplitude = coef_ctg1['A'][N2_index]
    N2_phase = coef_ctg1['g'][N2_index]
    print(f"N2 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {N2_amplitude}")
    print(f"N2 Phase for (lat:{lats[1]}, long:{lons[1]}): {N2_phase}")
else:
    print("N2 constituent not found in analysis.")    
    
if 'K2' in coef_ctg1['name']:
    K2_index = np.where(coef_ctg1['name'] == 'K2')[0][0]
    K2_amplitude = coef_ctg1['A'][K2_index]
    K2_phase = coef_ctg1['g'][K2_index]
    print(f"K2 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {K2_amplitude}")
    print(f"K2 Phase for (lat:{lats[1]}, long:{lons[1]}): {K2_phase}")
else:
    print("K2 constituent not found in analysis.")
if 'P1' in coef_ctg1['name']:
    P1_index = np.where(coef_ctg1['name'] == 'P1')[0][0]
    P1_amplitude = coef_ctg1['A'][P1_index]
    P1_phase = coef_ctg1['g'][P1_index]
    print(f"P1 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {P1_amplitude}")
    print(f"P1 Phase for (lat:{lats[1]}, long:{lons[1]}): {P1_phase}")
else:
    print("P1 constituent not found in analysis.")
    
if 'Q1' in coef_ctg1['name']:
    Q1_index = np.where(coef_ctg1['name'] == 'Q1')[0][0]
    Q1_amplitude = coef_ctg1['A'][Q1_index]
    Q1_phase = coef_ctg1['g'][Q1_index]
    print(f"Q1 Amplitude for (lat:{lats[1]}, long:{lons[1]}): {Q1_amplitude}")
    print(f"Q1 Phase for (lat:{lats[1]}, long:{lons[1]}): {Q1_phase}")
else:
    print("Q1 constituent not found in analysis.")
    
#%%

# Find the nearest node index for the first pair of lat and lon
nearest_node_index = find_nearest_node(lats[2], lons[2], grid_latitudes, grid_longitudes)

# Now select all data for this nearest node index
ctg2= model_ds.isel(nSCHISM_hgrid_node=nearest_node_index)
ctg2.SCHISM_hgrid_node_x.values, ctg2.SCHISM_hgrid_node_y.values

latitude =lats[2]
elevation = ctg2['elevation'].values
time =  pd.to_datetime(ctg2['time'].values)

coef_ctg2= utide.solve(time, elevation, lat=latitude, method='ols', conf_int='linear')

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2019-12-31')

prediction_time = pd.date_range(start=start_date, end=end_date, freq='H')  

prediction_ctg2 = reconstruct(prediction_time, coef_ctg2)



if 'M2' in coef_ctg2['name']:
    m2_index = np.where(coef_ctg2['name'] == 'M2')[0][0]
    m2_amplitude = coef_ctg2['A'][m2_index]
    m2_phase = coef_ctg2['g'][m2_index]
    print(f"M2 Amplitude for (lat:{lats[2]}, long:{lons[2]}) : {m2_amplitude}")
    print(f"M2 Phase for (lat:{lats[2]}, long:{lons[2]}) : {m2_phase}")
else:
    print("M2 constituent not found in analysis.")
    
if 'S2' in coef_ctg2['name']:
    s2_index = np.where(coef_ctg2['name'] == 'S2')[0][0]
    s2_amplitude = coef_ctg2['A'][s2_index]
    s2_phase = coef_ctg2['g'][s2_index]
    print(f"s2 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {s2_amplitude}")
    print(f"s2 Phase for (lat:{lats[2]}, long:{lons[2]}): {s2_phase}")
else:
    print("s2 constituent not found in analysis.")

if 'K1' in coef_ctg2['name']:
    k1_index = np.where(coef_ctg2['name'] == 'K1')[0][0]
    k1_amplitude = coef_ctg2['A'][k1_index]
    k1_phase = coef_ctg2['g'][k1_index]
    print(f"k1 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {k1_amplitude}")
    print(f"k1 Phase for (lat:{lats[2]}, long:{lons[2]}): {k1_phase}")
else:
    print("k1 constituent not found in analysis.")
    
if 'O1' in coef_ctg2['name']:
    O1_index = np.where(coef_ctg2['name'] == 'O1')[0][0]
    O1_amplitude = coef_ctg2['A'][O1_index]
    O1_phase = coef_ctg2['g'][O1_index]
    print(f"O1 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {O1_amplitude}")
    print(f"O1 Phase for (lat:{lats[2]}, long:{lons[2]}): {O1_phase}")
else:
    print("O1 constituent not found in analysis.")
    
if 'N2' in coef_ctg2['name']:
    N2_index = np.where(coef_ctg2['name'] == 'N2')[0][0]
    N2_amplitude = coef_ctg2['A'][N2_index]
    N2_phase = coef_ctg2['g'][N2_index]
    print(f"N2 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {N2_amplitude}")
    print(f"N2 Phase for (lat:{lats[2]}, long:{lons[2]}): {N2_phase}")
else:
    print("N2 constituent not found in analysis.")    
    
if 'K2' in coef_ctg2['name']:
    K2_index = np.where(coef_ctg2['name'] == 'K2')[0][0]
    K2_amplitude = coef_ctg2['A'][K2_index]
    K2_phase = coef_ctg2['g'][K2_index]
    print(f"K2 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {K2_amplitude}")
    print(f"K2 Phase for (lat:{lats[2]}, long:{lons[2]}): {K2_phase}")
else:
    print("K2 constituent not found in analysis.")
if 'P1' in coef_ctg2['name']:
    P1_index = np.where(coef_ctg2['name'] == 'P1')[0][0]
    P1_amplitude = coef_ctg2['A'][P1_index]
    P1_phase = coef_ctg2['g'][P1_index]
    print(f"P1 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {P1_amplitude}")
    print(f"P1 Phase for (lat:{lats[2]}, long:{lons[2]}): {P1_phase}")
else:
    print("P1 constituent not found in analysis.")
    
if 'Q1' in coef_ctg2['name']:
    Q1_index = np.where(coef_ctg2['name'] == 'Q1')[0][0]
    Q1_amplitude = coef_ctg2['A'][Q1_index]
    Q1_phase = coef_ctg2['g'][Q1_index]
    print(f"Q1 Amplitude for (lat:{lats[2]}, long:{lons[2]}): {Q1_amplitude}")
    print(f"Q1 Phase for (lat:{lats[2]}, long:{lons[2]}): {Q1_phase}")
else:
    print("Q1 constituent not found in analysis.")

#%%

# Find the nearest node index for the first pair of lat and lon
nearest_node_index = find_nearest_node(lats[3], lons[3], grid_latitudes, grid_longitudes)

# Now select all data for this nearest node index
ctg3= model_ds.isel(nSCHISM_hgrid_node=nearest_node_index)
ctg3.SCHISM_hgrid_node_x.values, ctg3.SCHISM_hgrid_node_y.values

latitude =lats[3]
elevation = ctg3['elevation'].values
time =  pd.to_datetime(ctg3['time'].values)

coef_ctg3= utide.solve(time, elevation, lat=latitude, method='ols', conf_int='linear')

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2019-12-31')

prediction_time = pd.date_range(start=start_date, end=end_date, freq='H')  

prediction_ctg3 = reconstruct(prediction_time, coef_ctg3)



if 'M2' in coef_ctg3['name']:
    m2_index = np.where(coef_ctg3['name'] == 'M2')[0][0]
    m2_amplitude = coef_ctg3['A'][m2_index]
    m2_phase = coef_ctg3['g'][m2_index]
    print(f"M2 Amplitude for (lat:{lats[3]}, long:{lons[3]}) : {m2_amplitude}")
    print(f"M2 Phase for (lat:{lats[3]}, long:{lons[3]}) : {m2_phase}")
else:
    print("M2 constituent not found in analysis.")
    
if 'S2' in coef_ctg3['name']:
    s2_index = np.where(coef_ctg3['name'] == 'S2')[0][0]
    s2_amplitude = coef_ctg3['A'][s2_index]
    s2_phase = coef_ctg3['g'][s2_index]
    print(f"s2 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {s2_amplitude}")
    print(f"s2 Phase for (lat:{lats[3]}, long:{lons[3]}): {s2_phase}")
else:
    print("s2 constituent not found in analysis.")

if 'K1' in coef_ctg3['name']:
    k1_index = np.where(coef_ctg3['name'] == 'K1')[0][0]
    k1_amplitude = coef_ctg3['A'][k1_index]
    k1_phase = coef_ctg3['g'][k1_index]
    print(f"k1 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {k1_amplitude}")
    print(f"k1 Phase for (lat:{lats[3]}, long:{lons[3]}): {k1_phase}")
else:
    print("k1 constituent not found in analysis.")
    
if 'O1' in coef_ctg3['name']:
    O1_index = np.where(coef_ctg3['name'] == 'O1')[0][0]
    O1_amplitude = coef_ctg3['A'][O1_index]
    O1_phase = coef_ctg3['g'][O1_index]
    print(f"O1 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {O1_amplitude}")
    print(f"O1 Phase for (lat:{lats[3]}, long:{lons[3]}): {O1_phase}")
else:
    print("O1 constituent not found in analysis.")
    
if 'N2' in coef_ctg3['name']:
    N2_index = np.where(coef_ctg3['name'] == 'N2')[0][0]
    N2_amplitude = coef_ctg3['A'][N2_index]
    N2_phase = coef_ctg3['g'][N2_index]
    print(f"N2 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {N2_amplitude}")
    print(f"N2 Phase for (lat:{lats[3]}, long:{lons[3]}): {N2_phase}")
else:
    print("N2 constituent not found in analysis.")    
    
if 'K2' in coef_ctg3['name']:
    K2_index = np.where(coef_ctg3['name'] == 'K2')[0][0]
    K2_amplitude = coef_ctg3['A'][K2_index]
    K2_phase = coef_ctg3['g'][K2_index]
    print(f"K2 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {K2_amplitude}")
    print(f"K2 Phase for (lat:{lats[3]}, long:{lons[3]}): {K2_phase}")
else:
    print("K2 constituent not found in analysis.")
if 'P1' in coef_ctg3['name']:
    P1_index = np.where(coef_ctg3['name'] == 'P1')[0][0]
    P1_amplitude = coef_ctg3['A'][P1_index]
    P1_phase = coef_ctg3['g'][P1_index]
    print(f"P1 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {P1_amplitude}")
    print(f"P1 Phase for (lat:{lats[3]}, long:{lons[3]}): {P1_phase}")
else:
    print("P1 constituent not found in analysis.")
    
if 'Q1' in coef_ctg3['name']:
    Q1_index = np.where(coef_ctg3['name'] == 'Q1')[0][0]
    Q1_amplitude = coef_ctg3['A'][Q1_index]
    Q1_phase = coef_ctg3['g'][Q1_index]
    print(f"Q1 Amplitude for (lat:{lats[3]}, long:{lons[3]}): {Q1_amplitude}")
    print(f"Q1 Phase for (lat:{lats[3]}, long:{lons[3]}): {Q1_phase}")
else:
    print("Q1 constituent not found in analysis.")
    
#%%
