# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the logistic growth model

This comment section is required for the beginning of any Python file.
It provides a general description of the file's contents.
"""

''' Explanation
Comment sections are created with triple quotation marks.
In the course, we use these sections to explain details of the code.
The script file starts by importing the functions used in the file.
We suggest to import first Python functions, then import our own functions.
'''

# Import Python functions
import numpy as np
import matplotlib.pyplot as plt
# Import our own functions
from models.log_growth import LogisticGrowth

# Simulation time array

# TODO: Define the array for simulation time, using the function 'linspace'.
# The array ranges from 0 to 10, with a step size of 1.0 [d].
# How many elements should this array have?
tsim = np.linspace(0, 10, 11)  # [d]
tsim2 = np.linspace(0, 10, 101)  # [d]

# Initialize model object

# TODO: Define the following variables:
# model integration time step size of 1.0 [d]
# dictionary of initial conditions for m = 1.0 [gDM m-2]
# dictionary of parameters for r = 1.2 [d-1] and K = 100 [gDM m-2]
dt = 1.0  # [d] time-step size
dt2 = 0.1  # [d] time-step size
x0 = {'m': 1.0}  # [gDM m-2] initial conditions
p = {'r': 1.2, 'K': 100}  # [d-1], [gDM m-2] model parameters
lg = LogisticGrowth(tsim, dt, x0, p)
lg2 = LogisticGrowth(tsim, dt2, x0, p)

# Run model
tspan = (tsim[0], tsim[-1])
y = lg.run(tspan)
y2 = lg2.run(tspan)


# Analytical Solution
m0, r, K = x0['m'], p['r'], p['K']
m_anl = K/(1 + (K-m0)/m0*np.exp(-r*tsim))
m2_anl = K/(1 + (K-m0)/m0*np.exp(-r*tsim2))

# Difference Calculations
# dt = 1
diff_ef = y['m'] - m_anl
diff_rk = y['m_rk'] - m_anl
#dt = 0.1
diff2_ef = y2['m'] - m2_anl
diff2_rk = y2['m_rk'] - m2_anl

it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    idx=it.index
    rel_error_ef = abs(diff_ef[idx]/m_anl[idx])
    rel_error_rk = abs(diff_rk[idx]/m_anl[idx])

    if rel_error_ef >= 0.1:
        print("The relative error of the Euler-Forward method (dt = 1) at time = " + repr(tsim[idx]) + " days is greater than 1%")
    if rel_error_rk >= 0.1:
        print("The relative error of the Runga-Kutta method (dt = 1) at time = " + repr(tsim[idx]) + " days is greater than 1%")

it = np.nditer(tsim2[:-1], flags=['f_index'])
for ti in it:
    idx=it.index
    rel_error_ef = abs(diff2_ef[idx]/m2_anl[idx])
    rel_error_rk = abs(diff2_rk[idx]/m2_anl[idx])

    if rel_error_ef >= 0.1:
        print("The relative error of the Euler-Forward method (dt = 0.1) at time = " + repr(tsim2[idx]) + " days is greater than 1%")
    if rel_error_rk >= 0.1:
        print("The relative error of the Runga-Kutta method (dt = 0.1) at time = " + repr(tsim2[idx]) + " days is greater than 1%")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Logistic Growth tutorial')
#plt.figure(1)
ax1.plot(tsim, m_anl, "C4", label="Analytical")
ax1.plot(y['t'], y['m'], "g", label="Euler-forward")
ax1.plot(y['t_rk'], y['m_rk'], "C1--", label="Runge-Kutta 4th Order")
ax1.set_xlabel(r'$time\ [d]$')
ax1.set_ylabel(r'$mass\ [gDM\ m^{-2}]$')
ax1.set_title('dt = 1')
ax1.legend()
ax2.plot(tsim, m_anl, "C4", label="Analytical")
ax2.plot(y2['t'], y2['m'], "g", label="Euler-forward")
ax2.plot(y2['t_rk'], y2['m_rk'], "C1--", label="Runge-Kutta 4th Order")
ax2.set_xlabel(r'$time\ [d]$')
ax2.set_ylabel(r'$mass\ [gDM\ m^{-2}]$')
ax2.set_title('dt = 0.1')
ax2.legend()

fig2, (ax3,ax4) = plt.subplots(1,2)
fig.suptitle('Logistic Growth Difference')
#plt.figure(1)
ax3.plot(tsim, diff_ef, "g", label="error EF")
ax3.plot(tsim, diff_rk, "C1--", label="error RK4")
ax3.set_xlabel(r'$time\ [d]$')
ax3.set_ylabel(r'$error\ [gDM\ m^{-2}]$')
ax3.set_title('dt = 1')
ax3.legend()
ax4.plot(tsim2, diff2_ef, "g", label="error EF")
ax4.plot(tsim2, diff2_rk, "C1--", label="error RK4")
ax4.set_xlabel(r'$time\ [d]$')
ax4.set_ylabel(r'$error\ [gDM\ m^{-2}]$')
ax4.set_title('dt = 0.1')
ax4.legend()
plt.show()

