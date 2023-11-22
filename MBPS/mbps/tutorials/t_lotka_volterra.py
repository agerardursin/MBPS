#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the Lotka-Volterra model

    dx1/dt = p1*x1 - p2*x1*x2
    dx2/dt = p3*x1*x2 - p4*x2
"""
# Import Python packages
import numpy as np
import matplotlib.pyplot as plt
# IMport our own functions
from models.lotka_volterra import LotkaVolterra

# Simulation time array
# FIXME: Define an array ranging from 0 to 365 [d],
# with step size 1.0
start=0.0
stop=365.0
step=1.0
tsim = np.linspace(start,stop,366)

# Initialize object
# FIXME: Define the following variables
    # - model integration step size dt = 7[d],
    # - dictionary of initial conditions for prey = 50 and pred = 50 [#],
    # - dictionary of model parameters (assume 1 [month] = 30 [d])
    #   p1=1 [month-1], p2=0.02 [pred-1 month-1],
    #   p3=0.01 [prey-1 month-1], p4=1 [month-1]
dt = 7.0
dt2 = 1.0
x0 = {'prey':50.0, 'pred':50.0}
p = {'p1':1.0/30, 'p2':0.02/30, 'p3':0.01/30, 'p4':1.0/30}
# FIXME: assign arguments to initialize the object.
# See the documentation of LotkaVolterra for help.
population = LotkaVolterra(tsim,dt,x0,p)
population2 = LotkaVolterra(tsim,dt2,x0,p)
# Run model
tspan = (tsim[0],tsim[-1])
y = population.run(tspan)
y2 = population2.run(tspan)

# Retrieve results
# FIXME: Retrieve the variables from the dictionary of results 'y'.
# See the file 'lotka_volterra.py'
# to find the names returned by the function 'output'
t = y['t']
t_sp = y['t_sp']
t2 = y2['t']
t2_sp = y2['t_sp']
prey = y['prey']
prey_sp = y['prey_sp']
prey2 = y2['prey']
prey2_sp = y2['prey_sp']
pred = y['pred']
pred_sp = y['pred_sp']
pred2 = y2['pred']
pred2_sp = y2['pred_sp']

# Plot results
# FIXME: Add labels to the axes.
# For math format, use: r'$ text goes here $'
# To show # in the text, use \#
plt.style.use('ggplot')
'''
plt.figure(1)
plt.plot(t,prey,label='Preys')
plt.plot(t,pred,label='Predators')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$population\ [#]$')
'''

fig, (ax1, ax2) = plt.subplots(1, 2)
#plt.figure(1)
ax1.plot(t,prey,label='Preys')
ax1.plot(t_sp,prey_sp,label='Preys Sci-Py')
ax1.plot(t,pred,label='Predators')
ax1.plot(t_sp,pred_sp,label='Predators Sci-Py')
ax1.set_xlabel(r'$time\ [d]$')
ax1.set_ylabel(r'$population\ [#]$')
ax1.set_title('dt = 7 days')
ax1.legend()
ax2.plot(t2,prey2,label='Preys')
ax2.plot(t2_sp,prey2_sp,label='Preys Sci-Py')
ax2.plot(t2,pred2,label='Predators')
ax2.plot(t2_sp,pred2_sp,label='Predators Sci-Py')
ax2.set_xlabel(r'$time\ [d]$')
ax2.set_ylabel(r'$population\ [#]$')
ax2.set_title('dt = 1 day')
ax2.legend()
plt.show()