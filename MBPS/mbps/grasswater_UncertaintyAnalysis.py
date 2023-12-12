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
from matplotlib import colormaps
import sys

sys.path.append('../mbps/models/')
sys.path.append('../mbps/functions/')

from models.grass_sol import Grass
from models.water_sol import Water
from scipy.optimize import least_squares

from models.grass_sol import Grass
from functions.calibration import fcn_residuals, fcn_accuracy
from functions.uncertainty import fcn_plot_uncertainty

plt.style.use('ggplot')

# Simulation time
tsim = np.linspace(0, 365, int(365 / 5) + 1)  # [d]
# tsim = np.linspace(0, 365*2, int(365/5)+1) # [d]

# Weather data (disturbances shared across models)
t_ini = '19950101'
# t_ini = '19940101'
t_end = '19960101'
# t_weather = np.linspace(0, 365*2, 365*2+1)
t_weather = np.linspace(0, 365, 365 + 1)
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv',  # .. to move up one directory from current directory
    skipinitialspace=True,  # ignore spaces after comma separator
    header=47 - 3,  # row with column names, 0-indexed, excluding spaces
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    # usecols = ['YEAR','MO', 'DY', 'TG', 'Q', 'RH'],
    index_col=0,  # column with row names from used columns, 0-indexed
)
# t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])+365
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
m_data = m_data / 1E3

# -- Grass sub-model --
# Step size
dt_grs = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws': 1.0 / 50, 'Wg': 1.5 / 50}  # [kgC m-2]

# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p_grs = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
         'alpha': 2E-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
         'beta': 0.05,  # [d-1] senescence rate
         'k': 0.5,  # [-] extinction coefficient of canopy
         'm': 0.1,  # [-] leaf transmission coefficient
         'M': 0.02,  # [d-1] maintenance respiration coefficient
         'mu_m': 0.5,  # [d-1] max. structural specific growth rate
         'P0': 0.432,  # [kgCO2 m-2 d-1] max photosynthesis parameter
         'phi': 0.9,  # [-] photoshynth. fraction for growth
         'Tmin': 0.0,  # [°C] maximum temperature for growth
         'Topt': 20.0,  # [°C] minimum temperature for growth
         'Tmax': 42.0,  # [°C] optimum temperature for growth
         'Y': 0.75,  # [-] structure fraction from storage
         'z': 1.33  # [-] bell function power
         }

# Model parameters adjusted manually to obtain growth
# TODO: Adjust a few parameters to obtain growth.
# Satrt by using the modifications from Case 1.
# If needed, adjust further those or additional parameters
p_grs['alpha'] = 8E-9  # 8.346E-09#4.478E-9#4.75E-9#4.009E-09
p_grs['beta'] = 0.04145  # #directly from grass_cal output
p_grs['k'] = 0.18  # 0.18#0.1757
p_grs['m'] = 0.8  # 0.8#0.6749
p_grs['phi'] = 0.9  # 8.591E-01+0.1

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I0 = 0.45 * I_gl * 1E4 / dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR

sd_temp = np.std(T)
sd_I0 = np.std(I0)

noise_temp = np.random.normal(0,sd_temp,len(T))
noise_I0 = np.random.normal(0,sd_I0,len(I0))

d_grs = {'T': np.array([t_weather, T]).T,
         'I0': np.array([t_weather, I0]).T,
         }

# Initialize module
grass = Grass(tsim, dt_grs, x0_grs, p_grs)

# -- Water sub-model --
dt_wtr = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the soil water sub-model
x0_wtr = {'L1': 0.36 * 150, 'L2': 0.32 * 250, 'L3': 0.24 * 600, 'DSD': 1}  # 3*[mm], [d]

# Castellaro et al. 2009, and assumed values for soil types and layers
p_wtr = {'alpha': 1.29E-6,  # [mm J-1] Priestley-Taylor parameter
         'gamma': 0.68,  # [mbar °C-1] Psychrometric constant
         'alb': 0.23,  # [-] Albedo (assumed constant crop & soil)
         'kcrop': 0.85,  # [mm d-1] Evapotransp coefficient, range (0.85-1.0)
         'WAIc': 0.75,  # [-] WDI critical, range (0.5-0.8)
         'theta_fc1': 0.36,  # [-] Field capacity of soil layer 1
         'theta_fc2': 0.32,  # [-] Field capacity of soil layer 2
         'theta_fc3': 0.24,  # [-] Field capacity of soil layer 3
         'theta_pwp1': 0.21,  # [-] Permanent wilting point of soil layer 1
         'theta_pwp2': 0.17,  # [-] Permanent wilting point of soil layer 2
         'theta_pwp3': 0.10,  # [-] Permanent wilting point of soil layer 3
         'D1': 150,  # [mm] Depth of Soil layer 1
         'D2': 250,  # [mm] Depth of soil layer 2
         'D3': 600,  # [mm] Depth of soil layer 3
         'krf1': 0.25,  # [-] Rootfraction layer 1 (guess)
         'krf2': 0.50,  # [-] Rootfraction layer 2 (guess)
         'krf3': 0.25,  # [-] Rootfraction layer 2 (guess)
         'mlc': 0.2,  # [-] Fraction of soil covered by mulching
         'S': 10,  # [mm d-1] parameter of precipitation retention
         }
p_wtr['kcrop'] = 0.9
# Disturbances
# global irradiance [J m-2 d-1], environment temperature [°C],
# precipitation [mm d-1], leaf area index [-].
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.
f_prc = data_weather.loc[t_ini:t_end, 'RH'].values  # [0.1 mm d-1] Precipitation
f_prc[f_prc < 0.0] = 0  # correct data that contains -0.1 for very low values

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I_glb = I_glb * 1E4 / dt_wtr  # [J cm-2 d-1] to [J m-2 d-1] Global irradiance
f_prc = f_prc / 10 / dt_wtr  # [0.1 mm d-1] to [mm d-1] Precipitation

sd_Iglb = np.std(I_glb)
sd_prc = np.std(f_prc)

noise_Iglb = np.random.normal(0,sd_Iglb,len(I_glb))
noise_prc = np.random.normal(0,sd_prc,len(f_prc))

d_wtr = {'I_glb': np.array([t_weather, I_glb]).T,
         'T': np.array([t_weather, T]).T,
         'f_prc': np.array([t_weather, f_prc]).T,
         }

# Initialize module
water = Water(tsim, dt_wtr, x0_wtr, p_wtr)


# %% Simulation function
def fnc_y(p0):
    # Reset initial conditions
    grass.x0 = x0_grs.copy()
    water.x0 = x0_wtr.copy()

    # grass.p['alpha'] = p0[0]
    # grass.p['phi'] = p0[0]#p0[1]
    grass.p['alpha'] = p0[0]
    grass.p['phi'] = p0[1]
    water.p['kcrop'] = p0[2]  # p0[2]
    # water.p['krf3'] = p0[2]

    # Initial disturbance
    d_grs['WAI'] = np.array([[0, 1, 2, 3, 4], [1., ] * 5]).T

    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tsim[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Integration span
        tspan = (tsim[idx], tsim[idx + 1])
        # Controlled inputs
        u_grs = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]
        u_wtr = {'f_Irg': 0}  # [mm d-1]
        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model
        y_wtr = water.run(tspan, d_wtr, u_wtr)
        # Retrieve water model outputs for grass model
        d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

    # Return result of interest (WgDM [kgDM m-2])
    return grass.y['Wg'] / 0.4


#### -- Calibration --

# Run calibration function
# p0 = np.array([p_grs['alpha'], p_grs['beta']]) # Initial guess
# bnds = ((1E-12, 1E-4), (1E-3, 0.1))
# p0 = np.array([p_grs['alpha'], p_grs['phi'], p_grs['beta']]) # Initial guess
# bnds = ((1E-12, 0.1, 1E-4), (1E-3, 0.99, 0.1))

# Final
# p0 = np.array([p_grs['alpha'], p_grs['phi']]) # Initial guess
# bnds = ((1E-12, 1E-4), (1E-3, 0.99))

# %% Challenge
# p0 = np.array([p_grs['alpha'], p_grs['m'], p_grs['phi'], p_wtr['kcrop']]) # Initial guess
# bnds = ((1E-12, 1E-4, 0.1, 0.85), (1E-3, 0.8, 0.99, 1))
# p0 = np.array([p_grs['m'], p_grs['phi'], p_wtr['kcrop']]) # Initial guess
# bnds = ((1E-4, 0.1, 0.85), (0.8, 0.99, 1))
# p0 = np.array([p_grs['alpha'], p_wtr['kcrop']]) # Initial guess
# bnds = ((1E-12, 0.85), (1E-3, 1))
# p0 = np.array([p_grs['alpha'], p_grs['phi'], p_wtr['kcrop'], p_wtr['krf3']]) # Initial guess
# bnds = ((1E-12, 1E-3, 0.85, 0.1), (1E-3, 0.99, 1, 0.75))
# p0 = np.array([p_grs['alpha'], p_grs['phi'], p_wtr['kcrop'], p_wtr['krf3']]) # Initial guess
# bnds = ((1E-12, 1E-3, 0.85, 0.1), (1E-3, 0.99, 1, 0.75))
# p0 = np.array([p_grs['alpha'], p_wtr['kcrop'], p_wtr['alpha']]) # Initial guess
# bnds = ((1E-12, 0.85, 1E-12), (1E-3, 1, 1E-3))
p0 = np.array([p_grs['alpha'], p_grs['phi'], p_wtr['kcrop']])  # Initial guess
bnds = ((1E-12, 1E-3, 0.85), (1E-3, 0.99, 1))
# bnds = ((1E-12, 1E-4, 1), (1E-3, 0.99,100))
y_ls = least_squares(fcn_residuals, p0, bounds=bnds,
                     args=(fnc_y, grass.t, t_data, m_data),
                     kwargs={'plot_progress': True})

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Run calibrated simulation
p_hat = y_ls['x']
WgDM_hat = fnc_y(p_hat)

#### -- Plot results --
plt.figure(1)
plt.plot(grass.t, WgDM_hat, label='WgDM')
plt.plot(t_data, m_data,
         linestyle='None', marker='o', label='WgDM data')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$grass\ biomass\ [kgDM\ m^{-2}]$')
plt.show()

#%% Uncertainty Analysis
# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# Simulation time array
tsim = np.linspace(0, 365, 365+1) # [d] 

# Monte Carlo simulations
n_sim = 1000 # number of simulations

# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size,n_sim), np.nan)
m_arr1 = np.full((tsim.size,n_sim), np.nan)
m_arr2 = np.full((tsim.size,n_sim), np.nan)
m_arr3 = np.full((tsim.size,n_sim), np.nan)

# Run simulations
for j in range(n_sim):
    print('sim: ', j)
    alpha = rng.normal(p_hat[0], y_calib_acc['sd'][0])
    phi = rng.normal(p_hat[1], y_calib_acc['sd'][1])
    kcrop = rng.normal(p_hat[2], y_calib_acc['sd'][2])
    p0 = (alpha, phi, kcrop)
    p01 = (p_hat[0], phi, kcrop)
    p02 = (alpha, p_hat[1], kcrop)
    p03 = (alpha, phi, p_hat[2])
    # pool()
    m = fnc_y(p0)
    m1 = fnc_y(p01)
    m2 = fnc_y(p02)
    m3 = fnc_y(p03)
    m_arr[:,j] = m
    m_arr1[:, j] = m1
    m_arr2[:, j] = m2
    m_arr3[:, j] = m3
    
# Plot results
plt.figure(1)
plt.plot(tsim, m_arr[:,0:12])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(2)
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50,0.68,0.95])
# ax2.plot(tsim, np.full(tsim.shape,1.110), color='k')
# ax2.plot(tsim, np.full(tsim.shape,1.420), color='k')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# alpha unchanged
plt.figure(3)
plt.plot(tsim, m_arr1[:,0:12])
plt.title(r'$\alpha\ unchanged$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(4)
ax3 = plt.gca()
ax3 = fcn_plot_uncertainty(ax3, tsim, m_arr1, ci=[0.50,0.68,0.95])
# ax2.plot(tsim, np.full(tsim.shape,1.110), color='k')
# ax2.plot(tsim, np.full(tsim.shape,1.420), color='k')
plt.title(r'$\alpha\ unchanged$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# phi unchanged
plt.figure(5)
plt.plot(tsim, m_arr2[:,0:12])
plt.title(r'$\phi\ unchanged$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(6)
ax4 = plt.gca()
ax4 = fcn_plot_uncertainty(ax4, tsim, m_arr2, ci=[0.50,0.68,0.95])
# ax2.plot(tsim, np.full(tsim.shape,1.110), color='k')
# ax2.plot(tsim, np.full(tsim.shape,1.420), color='k')
plt.title(r'$\phi\ unchanged$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# kcrop unchanged
plt.figure(7)
plt.plot(tsim, m_arr3[:,0:12])
plt.title(r'$k_{crop}\ unchanged$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(8)
ax5 = plt.gca()
ax5 = fcn_plot_uncertainty(ax5, tsim, m_arr3, ci=[0.50,0.68,0.95])
# ax2.plot(tsim, np.full(tsim.shape,1.110), color='k')
# ax2.plot(tsim, np.full(tsim.shape,1.420), color='k')
plt.title(r'$k_{crop}\ unchanged$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
plt.show()