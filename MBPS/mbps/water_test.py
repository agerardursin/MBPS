# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names --

Initial test of the soil water model
"""

import numpy as np
import matplotlib.pyplot as plt

from mbps.models.water import Water

plt.style.use('ggplot')

# Simulation time
# TODO: Define the simulation time array and integration time step
tsim = np.linspace(0, 365, 365+1)
dt = 1

# Initial conditions
# TODO: define the dictionary of initial conditions
x0 = {'L1': 25, 'L2':40, 'L3': 60, 'DSD':15}

# Castellaro et al. 2009, and assumed values for soil types and layers
# Define the dictonary of parameter values
p = {'S':10,                   # [mm d-1] parameter of precipitation retention
     'alpha':1.29E-6,          # [mm J-1] Priestley-Taylor parameter
     'gamma': 0.68,            # [mbar °C-1] Psychrometric constant
     'alb': 0.23,              # [-] Albedo of soil
     'kcrop': 0.90,            # [-] Evapotranspiration coefficient
     'WAIc': 0.75,             # [-] Critical water value for water availability index
     'theta_fc1': 0.36,        # [-] Field capacity of soil layer 1
     'theta_fc2': 0.32,        # [-] Field capacity of soil layer 2
     'theta_fc3': 0.24,        # [-] Field capacity of soil layer 3
     'theta_pwp1': 0.21,       # [-] Permanent wilting point of soil layer 1
     'theta_pwp2': 0.17,       # [-] Permanent wilting point of soil layer 2
     'theta_pwp3': 0.10,       # [-] Permanent wilting point of soil layer 3
     'D1': 150,                # [mm] Depth of Soil layer 1
     'D2': 250,                # [mm] Depth of soil layer 2
     'D3': 600,                # [mm] Depth of soil layer 3
     'krf1': 0.25   ,          # [-] Rootfraction layer 1
     'krf2': 0.50,             #[-] Rootfraction layer 2
     'krf3':0.25,              # [-] Rootfraction layer 3
     'mlc': 0.2                # [-] Fraction of soil covered by mulch
         
     }

# Disturbances (assumed constant for test)
# global irradiance [J m-2 d-1], environment temperature [°C], 
# precipitation [mm d-1], leaf area index [-]
# TODO: Define sensible constant values for the disturbances
d = {'I_glb':np.array([tsim, np.full((tsim.size,), 10000000)]).T,
     'T':np.array([tsim, np.full((tsim.size,), 20)]).T,   
     'f_prc':np.array([tsim, np.full((tsim.size,), 2)]).T,
     'LAI':np.array([tsim, np.full((tsim.size,), 0.65)]).T
     }

# Controlled inputs
u = {'f_Irg':0}            # [mm d-1]

# Initialize module
water = Water(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0],tsim[-1])
y_water = water.run(tspan, d, u)

# Retrieve simulation results
# TODO: Retrieve the arrays from the dictionary of model outputs.
t_water = y_water['t']
L1 = y_water['L1'] 
L2 = y_water['L2']
L3 = y_water['L3']

fc1 = np.full(len(tsim), (p['theta_fc1'] * p['D1']))
fc2 = np.full(len(tsim), (p['theta_fc2'] * p['D2']))
fc3 = np.full(len(tsim), (p['theta_fc3'] * p['D3']))

pwp1 = np.full(len(tsim), (p['theta_pwp1'] * p['D1']))
pwp2 = np.full(len(tsim), (p['theta_pwp2'] * p['D2']))
pwp3 = np.full(len(tsim), (p['theta_pwp3'] * p['D3']))

# Plots
# TODO: Plot the state variables, (as L and theta) and flows.
# Include lines for the fc and pwp for each layer.
plt.figure(1)
plt.plot(t_water, L1, label = 'L1', color = 'g')
plt.plot(t_water, L2, label = 'L2', color = 'b')
plt.plot(t_water, L3, label = 'L3', color = 'r')
plt.plot(t_water, fc1, label = 'fc1', linestyle = 'dashed', color = 'g')
plt.plot(t_water, fc2, label = 'fc2', linestyle = 'dashed', color = 'b')
plt.plot(t_water, fc3, label = 'fc3', linestyle = 'dashed', color = 'r')
plt.plot(t_water, pwp1, label = 'pwp1', linestyle = 'dashdot', color = 'g')
plt.plot(t_water, pwp2, label = 'pwp2', linestyle = 'dashdot', color = 'b')
plt.plot(t_water, pwp3, label = 'pwp3', linestyle = 'dashdot', color = 'r')
plt.xlabel("t_water")
plt.ylabel('soil water [mm]')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(t_water, L1/p['D1'], label = 'theta1', color ='g')
plt.plot(t_water, L2/p['D2'], label = 'theta2', color ='b')
plt.plot(t_water, L3/p['D3'], label = 'theta3', color ='r')
plt.plot(t_water, fc1/p['D1'], label = 'fc1', linestyle = 'dashed', color = 'g')
plt.plot(t_water, fc2/p['D2'], label = 'fc2', linestyle = 'dashed', color = 'b')
plt.plot(t_water, fc3/p['D3'], label = 'fc3', linestyle = 'dashed', color = 'r')
plt.plot(t_water, pwp1/p['D1'], label = 'pwp1', linestyle = 'dashdot', color = 'g')
plt.plot(t_water, pwp2/p['D2'], label = 'pwp2', linestyle = 'dashdot', color = 'b')
plt.plot(t_water, pwp3/p['D3'], label = 'pwp3', linestyle = 'dashdot', color = 'r')
plt.xlabel("t_water")
plt.ylabel('volume fraction [-]')
plt.legend()

plt.figure(3)
plt.plot(t_water, water.f['f_Pe'], label = 'f_Pe')
plt.plot(t_water, water.f['f_Ev']*-1, label = 'f_Ev')
plt.plot(t_water, water.f['f_Irg'], label = 'f_Irg')
plt.plot(t_water, water.f['f_Tr1']*-1, label = 'f_Tr1')
plt.plot(t_water, water.f['f_Dr1']*-1, label = 'f_Dr1')
plt.legend(loc='upper right')
plt.xlabel('time [d]')
plt.ylabel('flow rate [mm d-1]')

plt.figure(4)
plt.plot(t_water, water.f['f_Tr2']*-1, label = 'f_Tr2')
plt.plot(t_water, water.f['f_Dr2']*-1, label = 'f_Dr2')
plt.legend()
plt.xlabel('time [d]')
plt.ylabel('flow rate [mm d-1]')

plt.figure(5)
plt.plot(t_water, water.f['f_Tr3']*-1, label = 'f_Tr3')
plt.plot(t_water, water.f['f_Dr3']*-1, label = 'f_Dr3')
plt.legend()
plt.xlabel('time [d]')
plt.ylabel('flow rate [mm d-1]')
