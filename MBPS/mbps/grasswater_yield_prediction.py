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


d_wtr = {'I_glb': np.array([t_weather, I_glb]).T,
         'T': np.array([t_weather, T]).T,
         'f_prc': np.array([t_weather, f_prc]).T,
         }

# Initialize module
water = Water(tsim, dt_wtr, x0_wtr, p_wtr)

# Controlled inputs
u_grs = {'f_Gr': 0, 'f_Hr': 0, # [kgDM m-2 d-1]
         'f_Hr1':0.5, 'f_Hr2':0.2*0.4, # [kgDM m-2 d-1]
         'Hr_1':100, 'Hr_2': 195, # [d]
         'W_hr_1': 0.5*0.4, 'W_hr_2': 0.1*0.4}  # [kgDm]
u_wtr = {'f_Irg': 0, 'WAI_scale':1,'DSD_lim':1,'fr_S':0.2}  # [mm d-1]


# %% Simulation function
def fnc_y(p0, u_grass, u_water, u_in_grs=False, u_in_wtr=False):
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

        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grass, u_in_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model
        y_wtr = water.run(tspan, d_wtr, u_water, u_in_wtr)
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
                     args=(fnc_y, grass.t, t_data, m_data, u_grs, u_wtr),
                     kwargs={'plot_progress': True})

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Run calibrated simulation
p_hat = y_ls['x']
WgDM_hat = fnc_y(p_hat,u_grs, u_wtr, u_in_grs=True, u_in_wtr=False)
hrvst = grass.f['f_Hr']
total_harvest = np.nancumsum(hrvst)/0.4
ttl_hrvst = np.nansum(hrvst)/0.4
print('Total harvest: ', ttl_hrvst)
#### -- Plot results --
plt.figure(1)
plt.plot(grass.t, WgDM_hat, label='WgDM')
# plt.plot(grass.t,total_harvest, label='Harvest')
# plt.plot(t_data, m_data,
#          linestyle='None', marker='o', label='WgDM data')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$grass\ biomass\ [kgDM\ m^{-2}]$')
plt.show()

'''
# %% Uncertainty Analysis
# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# Simulation time array
tsim = np.linspace(0, 365, 365 + 1)  # [d] 

# Monte Carlo simulations
n_sim = 1000  # number of simulations

# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size, n_sim), np.nan)
m_arr1 = np.full((tsim.size, n_sim), np.nan)
m_arr2 = np.full((tsim.size, n_sim), np.nan)
m_arr3 = np.full((tsim.size, n_sim), np.nan)
m_arr4 = np.full((tsim.size, n_sim), np.nan)
m_arr5 = np.full((tsim.size, n_sim), np.nan)
m_arr6 = np.full((tsim.size, n_sim), np.nan)
m_arr7 = np.full((tsim.size, n_sim), np.nan)
m_arr8 = np.full((tsim.size, n_sim), np.nan)
m_arr9 = np.full((tsim.size, n_sim), np.nan)

# Init Harvest Arrays
Hr_arr = np.full((tsim.size, n_sim), np.nan)
Hr_arr1 = np.full((tsim.size, n_sim), np.nan)
Hr_arr2 = np.full((tsim.size, n_sim), np.nan)
Hr_arr3 = np.full((tsim.size, n_sim), np.nan)
Hr_arr4 = np.full((tsim.size, n_sim), np.nan)
Hr_arr5 = np.full((tsim.size, n_sim), np.nan)
Hr_arr6 = np.full((tsim.size, n_sim), np.nan)
Hr_arr7 = np.full((tsim.size, n_sim), np.nan)
Hr_arr8 = np.full((tsim.size, n_sim), np.nan)
Hr_arr9 = np.full((tsim.size, n_sim), np.nan)

# Controlled input
u_grsNoise = dict(u_grs)
u_wtrNoise = dict(u_wtr)

# Run simulations
for j in range(n_sim):
    print('sim: ', j)
    f_Hr1 = rng.normal(u_grs['f_Hr1'], u_grs['f_Hr1']*0.1)
    f_Hr2 = rng.normal(u_grs['f_Hr2'], u_grs['f_Hr2']*0.1)
    Hr_1 = rng.normal(u_grs['Hr_1'], u_grs['Hr_1']*0.1)
    Hr_2 = max(rng.normal(u_grs['Hr_2'], u_grs['Hr_2']*0.1),Hr_1+5)
    W_hr_1 = rng.normal(u_grs['W_hr_1'], u_grs['W_hr_1']*0.1)
    W_hr_2 = rng.normal(u_grs['W_hr_2'], u_grs['W_hr_2']*0.1)
    WAI_s = rng.normal(u_wtr['WAI_scale'], u_wtr['WAI_scale']*0.1)
    DSD_lim = max(rng.normal(u_wtr['DSD_lim'], u_wtr['DSD_lim']), 0)
    fr_S = max(rng.normal(u_wtr['fr_S'], u_wtr['fr_S']*0.3),0.01)

    u_grsNoise['f_Hr1'] = f_Hr1
    u_grsNoise['f_Hr2'] = f_Hr2
    u_grsNoise['Hr_1'] = Hr_1
    u_grsNoise['Hr_2'] = Hr_2
    u_grsNoise['W_hr_1'] = W_hr_1
    u_grsNoise['W_hr_2'] = W_hr_2

    u_grs1 = dict(u_grsNoise)
    u_grs1['f_Hr1'] = u_grs['f_Hr1']

    u_grs2 = dict(u_grsNoise)
    u_grs2['f_Hr2'] = u_grs['f_Hr2']

    u_grs3 = dict(u_grsNoise)
    u_grs3['Hr_1'] = u_grs['Hr_1']

    u_grs4 = dict(u_grsNoise)
    u_grs4['Hr_2'] = u_grs['Hr_2']

    u_grs5 = dict(u_grsNoise)
    u_grs5['W_hr_1'] = u_grs['W_hr_1']

    u_grs6 = dict(u_grsNoise)
    u_grs6['W_hr_2'] = u_grs['W_hr_2']

    u_wtrNoise['WAI_scale'] = WAI_s
    u_wtrNoise['DSD_lim'] = DSD_lim
    u_wtrNoise['fr_S'] = fr_S

    u_wtr1 = dict(u_wtrNoise)
    u_wtr1['WAI_scale'] = u_wtr['WAI_scale']
    u_wtr2 = dict(u_wtrNoise)
    u_wtr2['DSD_lim'] = u_wtr['DSD_lim']
    u_wtr3 = dict(u_wtrNoise)
    u_wtr3['fr_S'] = u_wtr['fr_S']


    m = fnc_y(p_hat, u_grsNoise, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    # grass
    m1 = fnc_y(p_hat, u_grs1, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr1[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m2 = fnc_y(p_hat, u_grs2, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr2[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m3 = fnc_y(p_hat, u_grs3, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr3[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m4 = fnc_y(p_hat, u_grs4, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr4[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m5 = fnc_y(p_hat, u_grs5, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr5[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m6 = fnc_y(p_hat, u_grs6, u_wtrNoise, u_in_grs=True, u_in_wtr=True)
    Hr_arr6[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    # water
    m7 = fnc_y(p_hat, u_grsNoise, u_wtr1, u_in_grs=True, u_in_wtr=True)
    Hr_arr7[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m8 = fnc_y(p_hat, u_grsNoise, u_wtr2, u_in_grs=True, u_in_wtr=True)
    Hr_arr8[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m9 = fnc_y(p_hat, u_grsNoise, u_wtr3, u_in_grs=True, u_in_wtr=True)
    Hr_arr9[:, j] = np.nancumsum(grass.f['f_Hr'])/0.4

    m_arr[:, j] = m
    m_arr1[:, j] = m1
    m_arr2[:, j] = m2
    m_arr3[:, j] = m3
    m_arr4[:, j] = m4
    m_arr5[:, j] = m5
    m_arr6[:, j] = m6
    m_arr7[:, j] = m7
    m_arr8[:, j] = m8
    m_arr9[:, j] = m9

# Plot results
# plt.figure(1)
# plt.plot(tsim, m_arr[:, 0:12])
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(1)
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50, 0.68, 0.95])
# ax2.plot(tsim, Hr_arr, color='k')
plt.title('All input noise')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# alpha unchanged
# plt.figure(3)
# plt.plot(tsim, m_arr1[:, 0:12])
# plt.title(r'$\alpha\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(2)
ax3 = plt.gca()
ax3 = fcn_plot_uncertainty(ax3, tsim, m_arr1, ci=[0.50, 0.68, 0.95])
# ax3.plot(tsim, Hr_arr1, color='k')
plt.title('$All\ input\ noise\ except\ f_{Hr1}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# phi unchanged
# plt.figure(5)
# plt.plot(tsim, m_arr2[:, 0:12])
# plt.title(r'$\phi\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(3)
ax4 = plt.gca()
ax4 = fcn_plot_uncertainty(ax4, tsim, m_arr2, ci=[0.50, 0.68, 0.95])
# ax4.plot(tsim, Hr_arr2, color='k')
plt.title('$All\ input\ noise\ except\ f_{Hr2}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# # kcrop unchanged
# plt.figure(7)
# plt.plot(tsim, m_arr3[:, 0:12])
# plt.title(r'$k_{crop}\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(4)
ax5 = plt.gca()
ax5 = fcn_plot_uncertainty(ax5, tsim, m_arr3, ci=[0.50, 0.68, 0.95])
# ax5.plot(tsim, Hr_arr3, color='k')
plt.title('$All\ input\ noise\ except\ Hr_{1}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# alpha unchanged
# plt.figure(9)
# plt.plot(tsim, m_arr1[:, 0:12])
# plt.title(r'$\alpha\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(5)
ax6 = plt.gca()
ax6 = fcn_plot_uncertainty(ax6, tsim, m_arr4, ci=[0.50, 0.68, 0.95])
# ax6.plot(tsim, Hr_arr4, color='k')
plt.title('$All\ input\ noise\ except\ Hr_{2}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# phi unchanged
# plt.figure(11)
# plt.plot(tsim, m_arr2[:, 0:12])
# plt.title(r'$\phi\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(6)
ax7 = plt.gca()
ax7 = fcn_plot_uncertainty(ax7, tsim, m_arr5, ci=[0.50, 0.68, 0.95])
# ax7.plot(tsim, Hr_arr5, color='k')
plt.title('$All\ input\ noise\ except\ W_{Hr1}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# kcrop unchanged
# plt.figure(13)
# plt.plot(tsim, m_arr3[:, 0:12])
# plt.title(r'$k_{crop}\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(7)
ax8 = plt.gca()
ax8 = fcn_plot_uncertainty(ax8, tsim, m_arr6, ci=[0.50, 0.68, 0.95])
# ax8.plot(tsim, Hr_arr6, color='k')
plt.title('$All\ input\ noise\ except\ W_{Hr2}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# alpha unchanged
# plt.figure(15)
# plt.plot(tsim, m_arr1[:, 0:12])
# plt.title(r'$\alpha\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(8)
ax9 = plt.gca()
ax9 = fcn_plot_uncertainty(ax9, tsim, m_arr7, ci=[0.50, 0.68, 0.95])
# ax9.plot(tsim, Hr_arr7, color='k')
plt.title('$All\ input\ noise\ except\ WAI_{scale}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# # phi unchanged
# plt.figure(17)
# plt.plot(tsim, m_arr2[:, 0:12])
# plt.title(r'$\phi\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(9)
ax10 = plt.gca()
ax10 = fcn_plot_uncertainty(ax10, tsim, m_arr8, ci=[0.50, 0.68, 0.95])
# ax10.plot(tsim, Hr_arr8, color='k')
plt.title('$All\ input\ noise\ except\ DSD_{lim}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

# kcrop unchanged
# plt.figure(19)
# plt.plot(tsim, m_arr3[:, 0:12])
# plt.title(r'$k_{crop}\ unchanged$')
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(10)
ax11 = plt.gca()
ax11 = fcn_plot_uncertainty(ax11, tsim, m_arr9, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ f_{S}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
# plt.show()

# Harvest Plots
plt.figure(11)
ax12 = plt.gca()
ax12 = fcn_plot_uncertainty(ax12, tsim, Hr_arr, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('All input noise')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(12)
ax13 = plt.gca()
ax13 = fcn_plot_uncertainty(ax13, tsim, Hr_arr1, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ f_{Hr1}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(13)
ax13 = plt.gca()
ax13 = fcn_plot_uncertainty(ax13, tsim, Hr_arr2, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ f_{Hr2}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(14)
ax14 = plt.gca()
ax14 = fcn_plot_uncertainty(ax14, tsim, Hr_arr3, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ Hr_{1}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(15)
ax15 = plt.gca()
ax15 = fcn_plot_uncertainty(ax15, tsim, Hr_arr4, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ Hr_{2}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(16)
ax16 = plt.gca()
ax16 = fcn_plot_uncertainty(ax16, tsim, Hr_arr5, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ W_{Hr1}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(17)
ax17 = plt.gca()
ax17 = fcn_plot_uncertainty(ax17, tsim, Hr_arr6, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ W_{Hr2}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(18)
ax18 = plt.gca()
ax18 = fcn_plot_uncertainty(ax18, tsim, Hr_arr7, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ WAI_{scale}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(19)
ax19 = plt.gca()
ax19 = fcn_plot_uncertainty(ax19, tsim, Hr_arr8, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ DSD_{lim}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(20)
ax20 = plt.gca()
ax20 = fcn_plot_uncertainty(ax20, tsim, Hr_arr9, ci=[0.50, 0.68, 0.95])
# ax11.plot(tsim, Hr_arr9, color='k')
plt.title('$All\ input\ noise\ except\ f_{S}$')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
plt.show()
'''