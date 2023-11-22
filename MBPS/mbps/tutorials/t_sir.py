# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Tutorial for the disease model (SIR)
    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I
"""
# TODO: Create the script to simulate the SIR model,
# and analyse parameter beta.
# Import Python packages
import numpy as np
import matplotlib.pyplot as plt
# Import our own functions
from models.sir import SIR

# Simulation Time array
# dt = 1 day
start=0.0
stop=365.0
step=1.0
num_points = int(stop/step) + 1
tsim = np.linspace(start,stop,num_points)

# Initialize Object
dt = step
x0 = {'S':0.99, 'I':0.01, 'R':0.0}
p = {'beta':0.50, 'gamma':0.02}
p2 = {'beta':0.10, 'gamma':0.02}

# Initialize SIR object
population = SIR(tsim,dt,x0,p)
population2 = SIR(tsim,dt,x0,p2)

# Run model
tspan = (tsim[0], tsim[-1])
y = population.run(tspan)
y2 = population2.run(tspan)

# Retrieve results
t = y['t']
S = y['S']
I = y['I']
R = y['R']
t2 = y2['t']
S2 = y2['S']
I2 = y2['I']
R2 = y2['R']
# Plot results
plt.style.use('ggplot')

#plt.figure(1)
#plt.plot(t,S,label='S')
#plt.plot(t,I,label='I')
#plt.plot(t,R,label='R')
#plt.legend()
#plt.xlabel(r'$time\ [d]$')
#plt.ylabel(r'$population\ fraction\ [-]$')
#plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2)
#plt.figure(1)
ax1.plot(t,S,label='S')
ax1.plot(t,I,label='I')
ax1.plot(t,R,label='R')
ax1.set_xlabel(r'$time\ [d]$')
ax1.set_ylabel(r'$population\ fraction\ [-]$')
ax1.set_title(r'$beta\ =\ 0.5\ [d^{-1}]$')
ax1.legend()
ax2.plot(t2,S2,label='S')
ax2.plot(t2,I2,label='I')
ax2.plot(t2,R2,label='R')
ax2.set_xlabel(r'$time\ [d]$')
ax2.set_ylabel(r'$population\ fraction\ [-]$')
ax2.set_title(r'$beta\ =\ 0.1\ [d^{-1}]$')
ax2.legend()
plt.show()