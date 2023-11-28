# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Tutorial for the uncertainty analysis of the logistic growth model.
This tutorial covers first the calibration of the logistic growth model,
then the identification of the uncertainty in the parameter estimates,
and their propagation through the model.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from MBPS.mbps.models.log_growth import LogisticGrowth
from MBPS.mbps.functions.calibration import fcn_residuals, fcn_accuracy
from MBPS.mbps.functions.uncertainty import fcn_plot_uncertainty

plt.style.use('ggplot')

# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# -- Calibration --
# Grass data, Wageningen 1984 (Bouman et al. 1996)
# Cummulative yield [kgDM m-2]
tdata = np.array([87, 121, 155, 189, 217, 246, 273, 304])
mdata = np.array([0.05, 0.21, 0.54, 0.88, 0.99, 1.02, 1.04, 1.13])

# Simulation time array
tsim = np.linspace(0, 365, 365+1) # [d]
tspan = (tsim[0], tsim[-1])

# Initialize and run reference object
dt = 1.0                   # [d] time step size
x0 = {'m':0.01}            # [gDM m-2] initial conditions
p = {'r':0.01,'K':1.0}     # [d-1], [kgDM m-2] model parameters (ref. values)
lg = LogisticGrowth(tsim,dt,x0,p)
y = lg.run(tspan)

# Define function to simulate model as a function of estimated array 'p0'.
def fcn_y(p0):
    # # Reset initial conditions
    # lg.x0 = x0.copy()
    # Reassign parameters from array to object
    lg.p['r'] = p0[0]
    lg.p['K'] = p0[1]
    lg.x0['m'] = p0[2]
    # Simulate the model
    y = lg.run(tspan)
    # Retrieve result (model output of interest)
    m = y['m']
    return m

# Run calibration function
p0 = np.array([p['r'], p['K'], x0['m']]) # Initial guess
y_ls = least_squares(fcn_residuals, p0,
                     args=(fcn_y, lg.t, tdata, mdata),
                     kwargs={'plot_progress':True})

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Simulate model with initial and estimated parameters
p_hat_arr = y_ls['x']
m_hat = fcn_y(p_hat_arr)
K_hat = p_hat_arr[1]
r_hat = p_hat_arr[0]
m0_hat = p_hat_arr[2]
# Plot calibration results
plt.figure(1)
plt.plot(lg.t, m_hat, label=r'$\hat{m}$')
plt.plot(tdata, mdata, label=r'$m_{data}$', linestyle='None', marker='o')
plt.xlabel('time ' + r'$[d]$')
plt.ylabel('cummulative mass '+ r'$[kgDM\ m^{-2}]$')
plt.legend()
plt.show()
# -- Uncertainty Analysis --
# Monte Carlo simulations
n_sim = 1000 # number of simulations
# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size,n_sim), np.nan)
# m = np.full((tsim.size, np.nan)
out = np.zeros(tsim.size)
sd_err = y_calib_acc['sd']
k_err = sd_err[1]
m0_err = sd_err[2]
# Run simulations
# np.set_printoptions(formatter={'float_kind':'{:.3E}'.format})
# print('m0 \n {} \n'.format(x0['m']))
# print('k_err \n {} \n'.format(k_err))

for j in range(n_sim):
    # TODO: Fill in the Monte Carlo simulations
    # m[0] = x0['m']
    K = rng.normal(K_hat, k_err)
    m0 = rng.normal(m0_hat*4,m0_err)
    # m0 = np.maximum(m0, 0.000001)
    p_in = np.array([r_hat, K, m0])
    # m = x0['m']*K / (x0['m'] + (K - x0['m'])*np.exp(-r_hat*tsim))
    # dm_dt = r_hat*m*(1-(m/K))
    dm_dt = fcn_y(p_in)
    m_arr[:, j] = dm_dt
# Plot results
# TODO: Plot the confidence intervals using 'fcn_plot_uncertainty'
# Plot results
plt.figure(2)
# TODO: Make a plot for the first 12 simulations (the first 12 columns in y_arr)
plt.plot(tsim, m_arr[:, 0:11]),
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(3)
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50,0.68,0.95])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')
plt.show()

# References

# Bouman, B.A.M., Schapendonk, A.H.C.M., Stol, W., van Kralingen, D.W.G.
#  (1996) Description of the growth model LINGRA as implemented in CGMS.
#  Quantitative Approaches in Systems Analysis no. 7
#  Fig. 3.4, p. 35