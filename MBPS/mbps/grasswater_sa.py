# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Evaluation of the grass & water model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import sys
sys.path.append('../mbps/models/')
from models.grass_sol import Grass
from models.water_sol import Water

plt.style.use('ggplot')

# Simulation time
tsim = np.linspace(0, 365, int(365/5)+1) # [d]

# Weather data (disturbances shared across models)
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

# -- Grass sub-model --
# Step size
dt_grs = 1 # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws':???,'Wg':???} # [kgC m-2]

# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p_grs = {'a':40.0,          # [m2 kgC-1] structural specific leaf area
     'alpha':2E-9,      # [kgCO2 J-1] leaf photosynthetic efficiency
     'beta':0.05,       # [d-1] senescence rate 
     'k':0.5,           # [-] extinction coefficient of canopy
     'm':0.1,           # [-] leaf transmission coefficient
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
# TODO: Adjust a few parameters to obtain growth.
# Satrt by using the modifications from Case 1.
# If needed, adjust further those or additional parameters
p_grs[???] = ???
p_grs[???] = ???

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end,'TG'].values    # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end,'Q'].values  # [J cm-2 d-1] Global irr.

T = T/10                   # [0.1 °C] to [°C] Environment temperature
I0 = 0.45*I_gl*1E4/dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR

d_grs = {'T':np.array([t_weather, T]).T,
    'I0':np.array([t_weather, I0]).T,   
    }

# Initialize module
grass = Grass(tsim, dt_grs, x0_grs, p_grs)

# -- Water sub-model --
dt_wtr = 1 # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the soil water sub-model
x0_wtr = {'L1':???, 'L2':???, 'L3':???, 'DSD':???} # 3*[mm], [d]

# Castellaro et al. 2009, and assumed values for soil types and layers
p_wtr = {'alpha':1.29E-6,   # [mm J-1] Priestley-Taylor parameter
     'gamma':0.68,      # [mbar °C-1] Psychrometric constant
     'alb':0.23,        # [-] Albedo (assumed constant crop & soil)
     'kcrop':0.90,      # [mm d-1] Evapotransp coefficient, range (0.85-1.0)
     'WAIc':0.75,       # [-] WDI critical, range (0.5-0.8)
     'theta_fc1':0.36,      # [-] Field capacity of soil layer 1
     'theta_fc2':0.32,      # [-] Field capacity of soil layer 2
     'theta_fc3':0.24,      # [-] Field capacity of soil layer 3
     'theta_pwp1':0.21,     # [-] Permanent wilting point of soil layer 1 
     'theta_pwp2':0.17,     # [-] Permanent wilting point of soil layer 2
     'theta_pwp3':0.10,     # [-] Permanent wilting point of soil layer 3
     'D1':150,        # [mm] Depth of Soil layer 1
     'D2':250,        # [mm] Depth of soil layer 2
     'D3':600,        # [mm] Depth of soil layer 3
     'krf1':0.25,     # [-] Rootfraction layer 1 (guess)
     'krf2':0.50,     # [-] Rootfraction layer 2 (guess)
     'krf3':0.25,     # [-] Rootfraction layer 2 (guess)
     'mlc':0.2,       # [-] Fraction of soil covered by mulching
     'S':10,          # [mm d-1] parameter of precipitation retention
     }

# Disturbances
# global irradiance [J m-2 d-1], environment temperature [°C], 
# precipitation [mm d-1], leaf area index [-].
T = data_weather.loc[t_ini:t_end,'TG'].values      # [0.1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end,'Q'].values  # [J cm-2 d-1] Global irr. 
f_prc = data_weather.loc[t_ini:t_end,'RH'].values # [0.1 mm d-1] Precipitation
f_prc[f_prc<0.0] = 0 # correct data that contains -0.1 for very low values

T = T/10                # [0.1 °C] to [°C] Environment temperature
I_glb = I_glb*1E4/dt_wtr    # [J cm-2 d-1] to [J m-2 d-1] Global irradiance
f_prc = f_prc/10/dt_wtr     # [0.1 mm d-1] to [mm d-1] Precipitation

d_wtr = {'I_glb' : np.array([t_weather, I_glb]).T, 
    'T' : np.array([t_weather, T]).T,
    'f_prc': np.array([t_weather, f_prc]).T,
     }

# Initialize module
water = Water(tsim, dt_wtr, x0_wtr, p_wtr)

#%% Simulation function
def fnc_y():
    # Reset initial conditions
    grass.x0 = x0_grs.copy()
    water.x0 = x0_wtr.copy()
    
    # Initial disturbance
    d_grs['WAI'] = np.array([[0,1,2,3,4], [1.,]*5]).T
    
    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tsim[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Integration span
        tspan = (tsim[idx], tsim[idx+1])
        # Controlled inputs
        u_grs = {'f_Gr':0, 'f_Hr':0}   # [kgDM m-2 d-1]
        u_wtr = {'f_Irg':0}            # [mm d-1]
        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model    
        y_wtr = water.run(tspan, d_wtr, u_wtr)
        # Retrieve water model outputs for grass model
        d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])
   
    # Return result of interest (WgDM [kgDM m-2])
    return grass.y['Wg']/0.4

#%% Sensitivity analysis

# Run reference simulation
WgDM = fnc_y()

# Initialize NS arrays for grass model
n_grs_rows, n_grs_cols = WgDM.size, len(p_grs.keys())
NS_grs_min = np.full((n_grs_rows, n_grs_cols), np.nan)
NS_grs_pls = np.full((n_grs_rows, n_grs_cols), np.nan)

# Iterate over grass model parameters
j = 0
for k in p_grs.keys():
    # Reset object parameters to reference values
    p_grs_ref = p_grs.copy()
    grass.p = p_grs_ref
    # Run model with p-5%
    grass.p[k] = 0.95*p_grs_ref[k]
    if k[0]=='T':
        grass.p[k] = p_grs_ref[k] - 5.0 
    WgDM_min = fnc_y()
    # Run model with p+5%
    grass.p[k] = 1.05*p_grs_ref[k]
    if k[0]=='T':
        grass.p[k] = p_grs_ref[k] + 5.0 
    WgDM_pls = fnc_y()
    # Sensitivities and Normalized sensitivities
    S_min = (WgDM_min - WgDM)/(0.95*p_grs_ref[k] - p_grs_ref[k])
    S_pls = (WgDM_pls - WgDM)/(1.05*p_grs_ref[k] - p_grs_ref[k])
    NS_grs_min[:,j] = S_min*p_grs_ref[k]/np.mean(WgDM)
    NS_grs_pls[:,j] = S_pls*p_grs_ref[k]/np.mean(WgDM)
    j += 1

# Initialize NS arrays for water model
n_wtr_rows, n_wtr_cols = WgDM.size, len(p_wtr.keys())
NS_wtr_min = np.full((n_wtr_rows, n_wtr_cols), np.nan)
NS_wtr_pls = np.full((n_wtr_rows, n_wtr_cols), np.nan)
j = 0

# Iterate over water model parameters
for k in p_wtr.keys():
    # Reset object parameters to reference values
    p_wtr_ref = p_wtr.copy()
    water.p = p_wtr_ref
    # Run model with p-5%
    water.p[k] = 0.95*p_wtr_ref[k]
    if k[0]=='T':
        water.p[k] = p_wtr_ref[k] - 5.0 
    WgDM_min = fnc_y()
    # Run model with p+5%
    water.p[k] = 1.05*p_wtr_ref[k]
    if k[0]=='T':
        water.p[k] = p_wtr_ref[k] + 5.0 
    WgDM_pls = fnc_y()
    # Sensitivities and Normalized sensitivities
    S_min = (WgDM_min - WgDM)/(0.95*p_wtr_ref[k] - p_wtr_ref[k])
    S_pls = (WgDM_pls - WgDM)/(1.05*p_wtr_ref[k] - p_wtr_ref[k])
    NS_wtr_min[:,j] = S_min*p_wtr_ref[k]/np.mean(WgDM)
    NS_wtr_pls[:,j] = S_pls*p_wtr_ref[k]/np.mean(WgDM)
    j += 1

# Sum-normalized sensitivities
NS_grs_min_sum = np.sum(np.abs(NS_grs_min),axis=1).reshape((grass.t.size,1))
SNS_grs_min = np.abs(NS_grs_min)/NS_grs_min_sum
NS_grs_pls_sum = np.sum(np.abs(NS_grs_pls),axis=1).reshape((grass.t.size,1))
SNS_grs_pls = np.abs(NS_grs_pls)/NS_grs_pls_sum

NS_wtr_min_sum = np.sum(np.abs(NS_wtr_min),axis=1).reshape((grass.t.size,1))
SNS_wtr_min = np.abs(NS_wtr_min)/NS_wtr_min_sum
NS_wtr_pls_sum = np.sum(np.abs(NS_wtr_pls),axis=1).reshape((grass.t.size,1))
SNS_wtr_pls = np.abs(NS_wtr_pls)/NS_wtr_pls_sum

#%% Figures SA grass
# Retrieve simulation results
t_grs = grass.t
t_wtr = water.t

# Plots
cmap = colormaps.get_cmap('tab10')

fig1, ((ax1a,ax1b),(ax1c,ax1d)) = plt.subplots(2,2, sharex=True, sharey=True)
ax1a.plot(t_grs, SNS_grs_pls[:,0], label=r'$a+$', color=cmap(0))
ax1a.plot(t_grs, SNS_grs_min[:,0], label=r'$a-$', linestyle='--', color=cmap(0))
ax1a.plot(t_grs, SNS_grs_pls[:,1], label=r'$\alpha+$', color=cmap(0.25))
ax1a.plot(t_grs, SNS_grs_min[:,1], label=r'$\alpha-$', linestyle='--', color=cmap(0.25))
ax1a.plot(t_grs, SNS_grs_pls[:,2], label=r'$\beta+$', color=cmap(0.50))
ax1a.plot(t_grs, SNS_grs_min[:,2], label=r'$\beta-$', linestyle='--', color=cmap(0.50))
ax1a.plot(t_grs, SNS_grs_pls[:,3], label=r'$k+$', color=cmap(0.75))
ax1a.plot(t_grs, SNS_grs_min[:,3], label=r'$k-$', linestyle='--', color=cmap(0.75))
ax1a.legend()
ax1a.set_ylim(-0.2, 0.8)
ax1a.set_ylabel(r'$SNS\ [-]$')

ax1b.plot(t_grs, SNS_grs_pls[:,4], label=r'$m+$',color=cmap(0))
ax1b.plot(t_grs, SNS_grs_min[:,4], label=r'$m-$', linestyle='--', color=cmap(0))
ax1b.plot(t_grs, SNS_grs_pls[:,5], label=r'$M+$',color=cmap(0.25))
ax1b.plot(t_grs, SNS_grs_min[:,5], label=r'$M-$', linestyle='--', color=cmap(0.25))
ax1b.plot(t_grs, SNS_grs_pls[:,6], label=r'$\mu_{m}+$', color=cmap(0.5))
ax1b.plot(t_grs, SNS_grs_min[:,6], label=r'$\mu_{m}-$', linestyle='--', color=cmap(0.5))
ax1b.plot(t_grs, SNS_grs_pls[:,7], label=r'$P0+$', color=cmap(0.75))
ax1b.plot(t_grs, SNS_grs_min[:,7], label=r'$P0-$', linestyle='--', color=cmap(0.75))
ax1b.legend()

ax1c.plot(t_grs, SNS_grs_pls[:,8], label=r'$\phi+$', color=cmap(0))
ax1c.plot(t_grs, SNS_grs_min[:,8], label=r'$\phi-$', linestyle='--', color=cmap(0))
ax1c.plot(t_grs, SNS_grs_pls[:,9], label=r'$Tmin+$',color=cmap(0.25))
ax1c.plot(t_grs, SNS_grs_min[:,9], label=r'$Tmin-$', linestyle='--', color=cmap(0.25))
ax1c.plot(t_grs, SNS_grs_pls[:,10], label=r'$Topt+$', color=cmap(0.5))
ax1c.plot(t_grs, SNS_grs_min[:,10], label=r'$Topt-$', linestyle='--', color=cmap(0.5))
ax1c.plot(t_grs, SNS_grs_pls[:,11], label=r'$Tmax+$', color=cmap(0.75))
ax1c.plot(t_grs, SNS_grs_min[:,11], label=r'$Tmax-$', linestyle='--', color=cmap(0.75))
ax1c.legend()
ax1c.set_ylim(-0.2, 0.8)
ax1c.set_ylabel(r'$SNS\ [-]$')
ax1c.set_xlabel('time'+r'$[d]$')

ax1d.plot(t_grs, SNS_grs_pls[:,12], label=r'$Y+$', color=cmap(0))
ax1d.plot(t_grs, SNS_grs_min[:,12], label=r'$Y-$', linestyle='--', color=cmap(0))
ax1d.plot(t_grs, SNS_grs_pls[:,13], label=r'$z+$',color=cmap(0.25))
ax1d.plot(t_grs, SNS_grs_min[:,13], label=r'$z-$', linestyle='--', color=cmap(0.25))
ax1d.legend()
ax1d.set_xlabel('time'+r'$[d]$')

#%% Figures SA water
fig2, ((ax2a,ax2b),(ax2c,ax2d),(ax2e,ax2f)) = plt.subplots(3,2, 
                                                           sharex=True, sharey=True)
ax2a.plot(t_wtr, SNS_wtr_pls[:,0], label=r'$\alpha+$', color=cmap(0))
ax2a.plot(t_wtr, SNS_wtr_min[:,0], label=r'$\alpha-$', linestyle='--', color=cmap(0))
ax2a.plot(t_wtr, SNS_wtr_pls[:,1], label=r'$\gamma+$', color=cmap(0.25))
ax2a.plot(t_wtr, SNS_wtr_min[:,1], label=r'$\gamma-$', linestyle='--', color=cmap(0.25))
ax2a.plot(t_wtr, SNS_wtr_pls[:,2], label=r'$alb+$', color=cmap(0.50))
ax2a.plot(t_wtr, SNS_wtr_min[:,2], label=r'$alb-$', linestyle='--', color=cmap(0.50))
ax2a.plot(t_wtr, SNS_wtr_pls[:,3], label=r'$k_{crop}+$', color=cmap(0.75))
ax2a.plot(t_wtr, SNS_wtr_min[:,3], label=r'$k_{crop}-$', linestyle='--', color=cmap(0.75))
ax2a.legend()
ax2a.set_ylim(-0.2, 0.8)
ax2a.set_ylabel(r'$SNS\ [-]$')

ax2b.plot(t_wtr, SNS_wtr_pls[:,4], label=r'$WAIc+$',color=cmap(0))
ax2b.plot(t_wtr, SNS_wtr_min[:,4], label=r'$WAIc-$', linestyle='--', color=cmap(0))
ax2b.plot(t_wtr, SNS_wtr_pls[:,5], label=r'$\theta_{fc1}+$',color=cmap(0.25))
ax2b.plot(t_wtr, SNS_wtr_min[:,5], label=r'$\theta_{fc1}-$', linestyle='--', color=cmap(0.25))
ax2b.plot(t_wtr, SNS_wtr_pls[:,6], label=r'$\theta_{fc2}+$', color=cmap(0.5))
ax2b.plot(t_wtr, SNS_wtr_min[:,6], label=r'$\theta_{fc2}-$', linestyle='--', color=cmap(0.5))
ax2b.plot(t_wtr, SNS_wtr_pls[:,7], label=r'$\theta_{fc3}+$', color=cmap(0.75))
ax2b.plot(t_wtr, SNS_wtr_min[:,7], label=r'$\theta_{fc3}-$', linestyle='--', color=cmap(0.75))
ax2b.legend()

ax2c.plot(t_wtr, SNS_wtr_pls[:,8], label=r'$\theta_{pwp1}+$', color=cmap(0))
ax2c.plot(t_wtr, SNS_wtr_min[:,8], label=r'$\theta_{pwp1}-$', linestyle='--', color=cmap(0))
ax2c.plot(t_wtr, SNS_wtr_pls[:,9], label=r'$\theta_{pwp2}+$',color=cmap(0.25))
ax2c.plot(t_wtr, SNS_wtr_min[:,9], label=r'$\theta_{pwp2}-$', linestyle='--', color=cmap(0.25))
ax2c.plot(t_wtr, SNS_wtr_pls[:,10], label=r'$\theta_{pwp3}+$', color=cmap(0.5))
ax2c.plot(t_wtr, SNS_wtr_min[:,10], label=r'$\theta_{pwp3}-$', linestyle='--', color=cmap(0.5))
ax2c.legend()
ax2c.set_ylim(-0.2, 0.8)
ax2c.set_ylabel(r'$SNS\ [-]$')

ax2d.plot(t_wtr, SNS_wtr_pls[:,11], label=r'$D1+$', color=cmap(0))
ax2d.plot(t_wtr, SNS_wtr_min[:,11], label=r'$D1-$', linestyle='--', color=cmap(0))
ax2d.plot(t_wtr, SNS_wtr_pls[:,12], label=r'$D2+$',color=cmap(0.25))
ax2d.plot(t_wtr, SNS_wtr_min[:,12], label=r'$D2-$', linestyle='--', color=cmap(0.25))
ax2d.plot(t_wtr, SNS_wtr_pls[:,13], label=r'$D3+$',color=cmap(0.5))
ax2d.plot(t_wtr, SNS_wtr_min[:,13], label=r'$D3-$', linestyle='--', color=cmap(0.5))
ax2d.legend()

ax2e.plot(t_wtr, SNS_wtr_pls[:,14], label=r'$k_{rf1}+$', color=cmap(0))
ax2e.plot(t_wtr, SNS_wtr_min[:,14], label=r'$k_{rf1}-$', linestyle='--', color=cmap(0))
ax2e.plot(t_wtr, SNS_wtr_pls[:,15], label=r'$k_{rf2}+$',color=cmap(0.25))
ax2e.plot(t_wtr, SNS_wtr_min[:,15], label=r'$k_{rf2}-$', linestyle='--', color=cmap(0.25))
ax2e.plot(t_wtr, SNS_wtr_pls[:,16], label=r'$k_{rf3}+$',color=cmap(0.5))
ax2e.plot(t_wtr, SNS_wtr_min[:,16], label=r'$k_{rf3}-$', linestyle='--', color=cmap(0.5))
ax2e.legend()
ax2e.set_ylim(-0.2, 0.8)
ax2e.set_ylabel(r'$SNS\ [-]$')
ax2e.set_xlabel('time'+r'$[d]$')

ax2f.plot(t_wtr, SNS_wtr_pls[:,17], label=r'$mlc+$', color=cmap(0))
ax2f.plot(t_wtr, SNS_wtr_min[:,17], label=r'$mlc-$', linestyle='--', color=cmap(0))
ax2f.plot(t_wtr, SNS_wtr_pls[:,18], label=r'$S+$',color=cmap(0.25))
ax2f.plot(t_wtr, SNS_wtr_min[:,18], label=r'$S-$', linestyle='--', color=cmap(0.25))
ax2f.legend()
ax2f.set_xlabel('time'+r'$[d]$')

# References
# Groot, J.C.J., and Lantinga, E.A., (2004). An object oriented model
#   of the morphological development and digestability of perennial
#   ryegrass. Ecological Modelling 177(3-4), 297-312.
