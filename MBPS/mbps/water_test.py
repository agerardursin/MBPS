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
tsim = ???
dt = ???

# Initial conditions
# TODO: define the dictionary of initial conditions
x0 = ???

# Castellaro et al. 2009, and assumed values for soil types and layers
# Define the dictonary of parameter values
p = {???
     }

# Disturbances (assumed constant for test)
# global irradiance [J m-2 d-1], environment temperature [°C], 
# precipitation [mm d-1], leaf area index [-]
# TODO: Define sensible constant values for the disturbances
d = {'I_glb':np.array([tsim, np.full((tsim.size,), ???)]).T,
     'T':np.array([tsim, np.full((tsim.size,), ???)]).T,   
     'f_prc':np.array([tsim, np.full((tsim.size,), ???)]).T,
     'LAI':np.array([tsim, np.full((tsim.size,), ???)]).T
     }

# Controlled inputs
u = {'f_irg':0}            # [mm d-1]

# Initialize module
water = Water(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0],tsim[-1])
y_water = water.run(tspan, d, u)

# Retrieve simulation results
# TODO: Retrieve the arrays from the dictionary of model outputs.
t_water = y_water['t']
L1 = y_water['L1'] 
L2 = ???
L3 = ???

# Plots
# TODO: Plot the state variables, (as L and theta) and flows.
# Include lines for the fc and pwp for each layer.
???

