# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Calibration of the grass model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from MBPS.mbps.models.grass_sol import Grass
from MBPS.mbps.functions.calibration import fcn_residuals, fcn_accuracy

plt.style.use('ggplot')

#### -- Data --
# Simulation time
tsim = np.linspace(0, 365, 365+1) # [d]

# Weather data (disturbances)
t_ini = '19950101'
t_end = '19960101'
t_weather = np.linspace(0, 365, 365+1)
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv', # .. to move up one directory from current directory
    skipinitialspace=True, # ignore spaces after comma separator
    header = 47-3, # row with column names, 0-indexed, excluding spaces
    usecols = ['YYYYMMDD', 'TG', 'Q', 'RH'], # columns to use
    index_col = 0, # column with row names from used columns, 0-indexed
    )

# Grass data. (Organic matter assumed equal to DM) [gDM m-2]
# (Groot and Lantinga, 2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
m_data = m_data/1E3

#### -- Grass model --
# Step size
dt = 1 # [d]

# Initial conditions
# TODO: Define the initial conditions based on the simulation results
# from your model evaluation (e.g., end-of-year values).
x0 = {'Ws':1,'Wg':1.5} # [kgC m-2]

# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p = {'a':40.0,          # [m2 kgC-1] structural specific leaf area
     'alpha':2E-9,      # [kgCO2 J-1] leaf photosynthetic efficiency
     'beta':0.05,       # [d-1] senescence rate 
     'k':0.05,           # [-] extinction coefficient of canopy
     'm':0.9,           # [-] leaf transmission coefficient
     'M':0.02,          # [d-1] maintenance respiration coefficient
     'mu_m':0.5,        # [d-1] max. structural specific growth rate
     'P0':0.432,        # [kgCO2 m-2 d-1] max photosynthesis parameter
     'phi':0.9,         # [-] photoshynth. fraction for growth
     'Tmin':0.0,        # [°C] maximum temperature for growth
     'Topt':20.0,       # [°C] minimum temperature for growth
     'Tmax':42.0,       # [°C] optimum temperature for growth
     'Y':0.75,          # [-] structure fraction from storage
     'z':1.33           # [-] bell function power
     }
# Model parameters adjusted manually to obtain growth
# TODO: Specify the model parameters that you adjusted previously
# (use your own specified values, for as many parameters as you needed)
p['alpha'] = 4.75E-9
p['k'] = 0.18
p['m'] = 0.8

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end,'TG'].values    # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end,'Q'].values  # [J cm-2 d-1] Global irr.

# TODO: Convert T and I_gl to the magnitudes and units required by the model
T = T * 0.1    # [0.1 °C] to [°C] Environment temperature
I0 = I_gl * 10000   # [J cm-2 d-1] to [J m-2 d-1] Global irradiance to PAR

d = {'T':np.array([t_weather, T]).T,
    'I0':np.array([t_weather, I0]).T,
    'WAI':np.array([t_weather, np.full((t_weather.size,),1.0)]).T
    }

# Initialize module
grass = Grass(tsim, dt, x0, p)

#### -- Simulation function --
def fnc_y(p0):
    # Reset initial conditions
    grass.x0 = x0.copy()
    
    # Model parameters
    # TODO: specify 4 relevant parameters to estimate.
    # (these may not necessarily be the ones that you adjusted manually before)
    grass.p['a'] = p0[0]
    grass.p['alpha'] = p0[1]
    grass.p['beta'] = p0[2]
    grass.p['k'] = p0[3]
    grass.p['m'] = p0[4]

    # Controlled inputs
    u = {'f_Gr':0, 'f_Hr':0}            # [kgC m-2 d-1]
    
    # Run simulation
    tspan = (tsim[0],tsim[-1])
    y_grass = grass.run(tspan, d, u)
      
    # Return result of interest (WgDM [kgDM m-2])
    # assuming 0.4 kgC/kgDM (Mohtar et al. 1997, p. 1492)
    return y_grass['Wg']/0.4

#### -- Calibration --

# Run calibration function
# TODO: Specify the initial guess for the parameter values
# These can be the reference values provided by Mohtar et al. (1997),
# You can simply call them from the dictionary p. 
# p0 = np.array([p['alpha'], p['beta'], p['k'], p['m'], p['mu_m'], p['P0'], p['z']]) # Initial guess
p0 = np.array([p['a'],p['alpha'], p['beta'], p['k'], p['m']]) # Initial guess

# Parameter bounds
# TODO: Specify bounds for your parameters, e.g., efficiencies lie between (0,1).
# Use a tuple of tuples for min and max values:
# ((p1_min, p2_min, p3_min), (p1_max, p2_max, p3_max))
bnds = ((0.5, 1E-10, 1E-4, 1E-10, 0.01), (50.0, 1E-6, 0.2, 0.2, 0.8))
# Call the lest_squares function.
# Our own residuals function takes the necessary positional argument p0,
# and the additional arguments fcn_y, t ,tdata, ydata.
y_ls = least_squares(fcn_residuals, p0, bounds=bnds,
                     args=(fnc_y, grass.t, t_data, m_data),
                     kwargs={'plot_progress':True})

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Run calibrated simulation
# TODO: Retrieve the parameter estimates from
# the output of the least_squares function
p_hat = y_ls['x']
# TODO: Run the model output simulation function with the estimated parameters
# (this is the calibrated model output)
WgDM_hat = fnc_y(p_hat)

#### -- Plot results --
# TODO: Make one figure comparing the calibrated model against
# the measured data
plt.figure(1)
plt.plot(t_weather,WgDM_hat)
plt.plot(t_data,m_data,'o')
plt.show()