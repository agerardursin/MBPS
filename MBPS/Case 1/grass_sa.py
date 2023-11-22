# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Sensitivity analysis of the grass growth model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MBPS.mbps.models.grass import Grass

plt.style.use('ggplot')

# TODO: Define the required variables for the grass module

# Simulation time
tsim = np.linspace(0.0, 365.0, 365+1) # [d]
dt = 1 # [d]
# Initial conditions
x0 = {'Ws':0.001,'Wg':0.003}  # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p =  {'a':40.0,          # [m2 kgC-1] structural specific leaf area
     'alpha':2E-9,      # [kgCO2 J-1] leaf photosynthetic efficiency -> was 2e-9 -> 5e-9
     'beta':0.05,      # Lowered Senescence rate since it seems to have been too large
     'k':0.5,           # was 0.5 --> 0.05
     'm':0.1,            # was 0.1 --> 0.9
     'M': 0.02,
     'mu_m':0.5,
     'P0':0.432,
     'phi':0.9,
     'Tmax':42.0,
     'Tmin':0.0,
     'Topt':20.0,
     'Y':0.75,
     'z':1.33
     }
# Model parameters adjusted manually to obtain growth
p['alpha'] = 4.75E-9
p['k'] = 0.18
p['m'] = 0.8

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], and
# water availability index [-]
t_ini = '20150101'
t_end = '20160101'
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv', # .. to move up one directory from current directory
    skipinitialspace=True, # ignore spaces after comma separator
    header = 47-3, # row with column names, 0-indexed, excluding spaces
    usecols = ['YYYYMMDD', 'TG', 'Q', 'RH'], # columns to use
    index_col = 0, # column with row names from used columns, 0-indexed
    )
# Retrieve relevant arrays from pandas dataframe
T = data_weather.loc[t_ini:t_end,'TG'].values    # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end,'Q'].values # [J cm-2 d-1] Global irradiance 
# Aply the necessary conversions of units
T = T * 0.1    # [???] to [???] Env. temperature
I0 = I_gl * 10000   # [???] to [???] Global irradiance to PAR
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
d = {'T':np.array([tsim, T]).T,
     'I0':np.array([tsim, I0]).T,   
     'WAI':np.array([tsim, np.full((tsim.size,),1.0)]).T
     }

# Controlled inputs
u = {'f_Gr':0, 'f_Hr':0}            # [kgDM m-2 d-1]

# Initialize grass module
grass = Grass(tsim, dt, x0, p)

# Normalized sensitivities
ns = grass.ns(x0, p, d=d, u=u, y_keys=('Wg',))

# Calculate mean NS through time
avg = {}
# TODO: use the ns DataFrame to calculate mean NS per parameter
for n_keys in ns.keys():
    avg[n_keys] = np.average(ns[n_keys])

# -- Plots
# TODO: Make the necessary plots (example provided below)
plt.figure(1)
plt.plot(grass.t, ns['Wg','alpha','-'], label='\u03B1 -', linestyle='--')
plt.plot(grass.t, ns['Wg','alpha','+'], label='\u03B1 +')
plt.plot(grass.t, ns['Wg','phi','-'], label='\u03A6 -', linestyle='--')
plt.plot(grass.t, ns['Wg','phi','+'], label='\u03A6 +')
plt.plot(grass.t, ns['Wg','m','-'], label='m -', linestyle='--')
plt.plot(grass.t, ns['Wg','m','+'], label='m +')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel('normalized sensitivity [-]')
plt.title('Normalized Sensitivity vs Time')
plt.show()
