# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Alek Gerard-Ursin & Nynke de Wilde

Evaluation of the grass growth model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MBPS.mbps.models.grass import Grass

plt.style.use('ggplot')

# Grass data
# TODO: define numpy arrays with measured grass data in the Netherlands
data = pd.read_excel("C:/Users/alek-/Downloads/grassGrowthRate2015.xlsx", sheet_name = "Sheet2")
t_grass_data = data['day'].to_numpy()-96
m_grass_data = data['mass'].to_numpy()

# Simulation time
tsim = np.linspace(0.0, 2*365.0, 2*365+1) # [d]
dt = 1 # [d]

# Initial conditions
# TODO: define sensible values for the initial conditions
x0 = {'Ws':0.001,'Wg':0.003}  # [kgC m-2]

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the varameper values in the dictionary p
p = {'a':40.0,          # [m2 kgC-1] structural specific leaf area
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
# TODO: (Further) adjust 2-3 parameter values to match measured growth behaviour
p['alpha'] = 4.75E-9
p['k'] = 0.18
p['m'] = 0.8

# Disturbances
# PAR [J m-2 d-1], env. temperature [°C], and water availability index [-]
# TODO: Specify the corresponding dates to read weather data (see csv file).
t_ini = '20150406'
t_end = '20160405'
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
T = T/10    # [0.1 °C] to [1 °C] Env. temperature
I0 = I_gl * 10000   # [J cm-2 d-1] to [J m-2 d-1] Global irradiance to PAR
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
d = {'T':np.array([tsim, T]).T,
     'I0':np.array([tsim, I0]).T,
     'WAI':np.array([tsim, np.full((tsim.size,),1.0)]).T
     }

# Controlled inputs
u = {'f_Gr':0, 'f_Hr':0}            # [kgDM m-2 d-1]

# Initialize module
# TODO: Call the module Grass to initialize an instance
grass = Grass(tsim,dt,x0,p)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y_grass = grass.run(tspan,d,u)

# Retrieve simulation results
# assuming 0.4 kgC/kgDM (Mohtar et al. 1997, p. 1492)
# TODO: Retrieve the simulation results
t_grass = y_grass['t']
WsDM = y_grass['Ws']
WgDM = y_grass['Wg']


# Plot
# TODO: Make a plot for WsDM, WgDM and grass measurement data.
plt.figure(1)
plt.plot(t_grass,WsDM,label='$W_{s}$')
plt.plot(t_grass,WgDM,label='$W_{g}$')
plt.plot(t_grass_data,m_grass_data,label='Measured data')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$Mass\ Dry\ Matter\ [kg\ C\ m^{-2}]$')
plt.show()

