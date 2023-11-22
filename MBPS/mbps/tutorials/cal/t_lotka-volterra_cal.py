# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Tutorial for the calibration of the Lotka-Volterra model
Exercise 3
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from MBPS_startcourse.mbps.models.lotka_volterra import LotkaVolterra
from MBPS_startcourse.mbps.functions.calibration import fcn_residuals, fcn_accuracy

plt.style.use('ggplot')

# Simulation time array
tsim = np.arange(0, 365, 1)
tspan = (tsim[0], tsim[-1])

# Initialize reference object
dt = 1.0                    # [d] time-step size
x0 = {'prey':50, 'pred':50} # populations [preys, preds]
# Model parameters
# p1 [d-1], p2 [pred-1 d-1], p3 [prey-1 d-1], p4 [d-1]
# p = {'p3':0.01/30, 'p4':1.0/30}
p = {'p2':0.02/30, 'p4':1.0/30}
# Initialize object
lv = LotkaVolterra(tsim, dt, x0, p)

# Data
t_data = np.array([60, 120, 180, 240, 300, 360])
y_data = np.array([[96, 191, 61, 83, 212, 41],  # [preys]
                   [18, 50, 64, 35, 40, 91]]).T # [preds]

# Define function to simulate model based on estimated array 'p0'.
# -- Exercise 3.1. Estimate p1 and p2
# -- Exercise 3.2. Estimate p1 and p3
def fcn_y(p0):
    # Reset initial conditions
    lv.x0 = x0.copy()
    # Reassign parameters from array p0 to object
    lv.p['p1'] = p0[0]
    lv.p['p3'] = p0[1]
    # Simulate the model
    y = lv.run(tspan)
    # Retrieve result (model output of interest)
    # Note: For computational speed in the least squares routine,
    # it is best to compute the residuals based on numpy arrays.
    # We use rows for time, and columns for model outputs.
    # TODO: retrieve the model outputs into a numpy array for populations 'pop'
    pop = np.array([y['prey'],y['pred']]).T
    return pop

# Run calibration function
# -- Exercise 3.1. Estimate p1 and p2
# -- Exercise 3.2. estimate p1 and p3
p0 = np.array([1.0/30, 0.01/30]) # Initial guess
# p0_2 = np.array([0.1/30,0.1/30])
# p0_3 = np.array([1.0/30,1.0/30])
# p0_4 = np.array([1.0/30,0.02/30])
# plt.figure('Calibrated model and data')

# p0_arr = np.array([p0, p0_2, p0_3, p0_4])
# for p_0 in p0_arr:

y_ls = least_squares(fcn_residuals, p0,
                     bounds = ([1E-6, 1E-6], [np.inf, np.inf]),
                     args=(fcn_y, lv.t, t_data, y_data),
                     )
# y_ls2 = least_squares(fcn_residuals, p0_2,
#                      bounds = ([1E-6, 1E-6], [np.inf, np.inf]),
#                      args=(fcn_y, lv.t, t_data, y_data),
#                      )
# y_ls3 = least_squares(fcn_residuals, p0_3,
#                      bounds = ([1E-6, 1E-6], [np.inf, np.inf]),
#                      args=(fcn_y, lv.t, t_data, y_data),
#                      )
# y_ls4 = least_squares(fcn_residuals, p0_4,
#                      bounds = ([1E-6, 1E-6], [np.inf, np.inf]),
#                      args=(fcn_y, lv.t, t_data, y_data),
#                      )
# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)
# y_calib_acc2 = fcn_accuracy(y_ls2)
# y_calib_acc3 = fcn_accuracy(y_ls3)
# y_calib_acc4 = fcn_accuracy(y_ls4)

# Run model output function with the estimated parameters
p_hat = y_ls['x']
# p_hat2 = y_ls2['x']
# p_hat3 = y_ls3['x']
# p_hat4 = y_ls4['x']
y_hat = fcn_y(p_hat)
# y_hat2 = fcn_y(p_hat2)
# y_hat3 = fcn_y(p_hat3)
# y_hat4 = fcn_y(p_hat4)
# p_out = np.r[p_out,[p]]

# Plot calibrated model
# -- Exercise 3.1 and 3.2
# TODO: plot the model output based on the estimated parameters,
# together with the data.
# plt.plot()
fig, axs = plt.subplots(1,1)#(2, 2)
#plt.figure(1)
axs.plot(tsim,y_hat[:,0],label='Prey interp')
axs.plot(t_data,y_data[:,0],'o',label='Prey data')
axs.plot(tsim,y_hat[:,1],label='Predators')
axs.plot(t_data,y_data[:,1],'o',label='Predators data')
axs.set_xlabel(r'$time\ [d]$')
axs.set_ylabel(r'$population\ [-]$')
# axs.set_title(r'$p_{1}=\ 0.01/30, p_{2}=\ 0.01/30,$')
fig.suptitle('Calibrated model and data')
axs.legend()
# axs[0,1].plot(tsim,y_hat2[:,0],label='Prey interp')
# axs[0,1].plot(t_data,y_data[:,0],'o',label='Prey data')
# axs[0,1].plot(tsim,y_hat2[:,1],label='Predators')
# axs[0,1].plot(t_data,y_data[:,1],'o',label='Predators data')
# axs[0,1].set_xlabel(r'$time\ [d]$')
# axs[0,1].set_ylabel(r'$population\ [#]$')
# axs[0,1].set_title(r'$p_{1}=\ 0.1/30, p_{2}=\ 0.1/30,$')
# axs[0,1].legend()
# axs[1,0].plot(tsim, y_hat3[:,0], label='Prey interp')
# axs[1,0].plot(t_data, y_data[:,0],'o', label='Prey data')
# axs[1,0].plot(tsim, y_hat3[:,1], label='Predators')
# axs[1,0].plot(t_data, y_data[:,1],'o', label='Predators data')
# axs[1,0].set_xlabel(r'$time\ [d]$')
# axs[1,0].set_ylabel(r'$population\ [#]$')
# axs[1,0].set_title(r'$p_{1}=\ 1.0/30, p_{2}=\ 1.0/30,$')
# axs[1,0].legend()
# axs[1,1].plot(tsim, y_hat4[:,0], label='Prey interp')
# axs[1,1].plot(t_data, y_data[:,0],'o', label='Prey data')
# axs[1,1].plot(tsim, y_hat4[:,1], label='Predators')
# axs[1,1].plot(t_data, y_data[:,1],'o', label='Predators data')
# axs[1,1].set_xlabel(r'$time\ [d]$')
# axs[1,1].set_ylabel(r'$population\ [#]$')
# axs[1,1].set_title(r'$p_{1}=\ 1.0/30, p_{2}=\ 0.02/30,$')
# axs[1,1].legend()
plt.show()