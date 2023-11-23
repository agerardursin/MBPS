# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Evaluation of the soil water model

NOTE: To change the simulation from 1[d] to 1[hr] time step:
    1) change tsim
        tsim = np.linspace(0, 365, 24*365+1)
    2) change dt
        dt = 1/24
    3) add hour in t_ini and t_end, e.g.:
        t_ini = '20170101 1'
        t_end = '20180101 1'
    4) comment out daily weather data, and uncomment hourly weather data
    5) change temperature string from 'TG' to 'T':
        T = data_weather.loc[t_ini:t_end,'T'].values
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mbps.models.water import Water

plt.style.use('ggplot')

# Simulation time
# TODO: Define the simulation time and integration time step 
tsim = ???
dt = ???

# Initial conditions
# Define the dictionary of initial conditions
x0 = ???

# Castellaro et al. 2009, and assumed values for soil types and layers
# TODO: Define the dictionary of values for model parameters
p = {???}

# Disturbances (assumed constant for test)
# environment temperature [°C], global irradiance [J m-2 d-1], 
# precipitation [mm d-1], leaf area index [-]
t_ini = '20170101'
t_end = '20180101'

# Daily data
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv', # .. to move up one directory from current directory
    skipinitialspace=True, # ignore spaces after comma separator
    header = 47-3, # row with column names, 0-indexed, excluding spaces
    usecols = ['YYYYMMDD', 'TG', 'Q', 'RH'], # columns to use
    index_col = 0, # column with row names from used columns, 0-indexed
    )

# Hourly data
# data_weather = pd.read_csv(
#     '../data/uurgeg_260_2011-2020.csv',
#     skipinitialspace=True, # ignore spaces after comma separator
#     header = 31-3, # row with column names, 0-indexed, excluding spaces
#     usecols = ['YYYYMMDD', 'HH', 'T', 'Q', 'RH'], # columns to use
#     parse_dates = [[0,1]], # Combine first two columns as index
#     index_col = 0, # column with row names, from used & parsed columns, 0-indexed
#     )

data_LAI = pd.read_csv('../data/LAI.csv') # Dummy LAI from grass evaluation

T = data_weather.loc[t_ini:t_end,'TG'].values      # [0.1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end,'Q'].values  # [J cm-2 d-1] Global irr. 
f_prc = data_weather.loc[t_ini:t_end,'RH'].values # [0.1 mm d-1] Precipitation
f_prc[f_prc<0.0] = 0 # correct data that contains -0.1 for very low values

# TODO: Apply the necessary conversions
T = ???
I_glb = ???
f_prc = ???

d = {'I_glb' : np.array([tsim, I_glb]).T, 
    'T' : np.array([tsim, T]).T,
    'f_prc': np.array([tsim, f_prc]).T,
    'LAI' : np.array([data_LAI.iloc[:,0].values, data_LAI.iloc[:,1]]).T
     }

# Controlled inputs
u = {'f_Irg':0}            # [mm d-1]

# Initialize module
water = Water(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0], tsim[-1])
y_water = water.run(tspan, d, u)

# Retrieve simulation results
# TODO: Retrive variables from the dictionary of model outputs

# Plot
# TODO: Make plots for the state variables (as L and theta),
# and the volumetric flows.
# Include lines for the pwp and fc for each layer.

