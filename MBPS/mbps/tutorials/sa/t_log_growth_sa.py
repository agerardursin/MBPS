# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Tutorial for the sensitivity analysis of the logistic growth model
"""
import numpy as np
import matplotlib.pyplot as plt

from MBPS_startcourse.mbps.models.log_growth import LogisticGrowth

plt.style.use('ggplot')

# Simulation time array
tsim = np.linspace(0, 10, 100+1)    # [d]
tspan = (tsim[0], tsim[-1])         # [d]

# Initialize and run reference object
dt = 0.1                        # [d] time-step size
x0 = {'m':1.0}                  # [gDM m-2] initial conditions
p_ref = {'r':1.2,'K':100.0}     # [d-1], [gDM m-2] model parameters (ref. values)
lg = LogisticGrowth(tsim,dt,x0,p_ref)   # Initialize obgect
y = lg.run(tspan)                       # Run object
y_mean = np.mean(y['m'])

# One-at-a-time parameter changes, simulation, S, and NS
# dm/dr-
p_rmin = p_ref.copy()                           # define p with r-
p_rmin['r'] = 0.95*p_ref['r']                   # update value of r-
lg_rmin = LogisticGrowth(tsim, dt, x0, p_rmin)  # initialize object
y_rmin = lg_rmin.run(tspan)                     # run object
S_rmin = (y_rmin['m']-y['m'])/(p_rmin['r']-p_ref['r'])  # sensitivity
NS_rmin = S_rmin *(p_ref['r']/y['m'])
NS_p_rmin = S_rmin *(p_ref['r']/y_mean)

# TODO: Code the sensitivity S_rpls
# dm/dr+
p_rmax = p_ref.copy()                           # define p with r-
p_rmax['r'] = 1.05*p_ref['r']                   # update value of r-
lg_rmax = LogisticGrowth(tsim, dt, x0, p_rmax)  # initialize object
y_rmax = lg_rmax.run(tspan)                     # run object
S_rmax = (y_rmax['m']-y['m'])/(p_rmax['r']-p_ref['r'])  # sensitivity
NS_rmax = S_rmax *(p_ref['r']/y['m'])
NS_p_rmax = S_rmax *(p_ref['r']/y_mean)

# TODO: Code sensitivity S_Kmin
# dm/dK-
p_kmin = p_ref.copy()                           # define p with r-
p_kmin['K'] = 0.95*p_ref['K']                   # update value of r-
lg_kmin = LogisticGrowth(tsim, dt, x0, p_kmin)  # initialize object
y_kmin = lg_kmin.run(tspan)                     # run object
S_kmin = (y_kmin['m']-y['m'])/(p_kmin['K']-p_ref['K'])  # sensitivity
NS_kmin = S_kmin *(p_ref['K']/y['m'])
NS_p_kmin = S_kmin *(p_ref['K']/y_mean)

# TODO: Code sensitivity S_Kpls
# dm/dK+
p_kmax = p_ref.copy()                           # define p with r-
p_kmax['K'] = 1.05*p_ref['K']                   # update value of r-
lg_kmax = LogisticGrowth(tsim, dt, x0, p_kmax)  # initialize object
y_kmax = lg_kmax.run(tspan)                     # run object
S_kmax = (y_kmax['m']-y['m'])/(p_kmax['K']-p_ref['K'])  # sensitivity
NS_kmax = S_kmax *(p_ref['K']/y['m'])
NS_p_kmax = S_kmax *(p_ref['K']/y_mean)

NS_p_r = (NS_p_rmin + NS_p_rmax) * 0.5
NS_p_k = (NS_p_kmin + NS_p_kmax) * 0.5

SNS_sum = NS_p_k + NS_p_r
SNS_r = NS_p_r / SNS_sum
SNS_k = NS_p_k / SNS_sum

# Plot results
# m with changes in r
# TODO: Make a plot m vs t changing r+/-5%
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Logistic Growth')
#plt.figure(1)
ax1.plot(y['t'], y['m'], "g", label="$r_{ref}$")
ax1.plot(y_rmin['t'], y_rmin['m'],"--", label="$r\ =\ $95%$\ r_{ref}$")
ax1.plot(y_rmax['t'], y_rmax['m'],"--", label="$r\ =\ $105%$\ r_{ref}$")
ax1.set_xlabel(r'$time\ [d]$')
ax1.set_ylabel(r'$mass\ [gDM\ m^{-2}]$')
ax1.set_title('r')
ax1.legend()
ax2.plot(y['t'], y['m'], "g", label="$K_{ref}$")
ax2.plot(y_kmin['t'], y_kmin['m'],"--", label="$K\ =\ $95%$\ K_{ref}$")
ax2.plot(y_kmax['t'], y_kmax['m'],"--", label="$K\ =\ $105%$\ K_{ref}$")
ax2.set_xlabel(r'$time\ [d]$')
ax2.set_ylabel(r'$mass\ [gDM\ m^{-2}]$')
ax2.set_title('K')
ax2.legend()
# plt.show()
# m with changes in K
# TODO: Make a plot m vs t changing K+/-5%
# plt.figure(2)


# S on r
# TODO: Make a plot S vs t changing r+/-5%
# plt.figure(3)
fig2, (ax3, ax4) = plt.subplots(1, 2)
fig2.suptitle('Sensitivity Analysis')
#plt.figure(1)
ax3.plot(y_rmin['t'], S_rmin,"--", label="$r\ =\ $95%$\ r_{ref}$")
ax3.plot(y_rmax['t'], S_rmax, label="$r\ =\ $105%$\ r_{ref}$")
ax3.set_xlabel(r'$time\ [d]$')
ax3.set_ylabel(r'$Sensitivity\ [gDM\ m^{-2}\ d^{-1}]$')
ax3.set_title('r')
ax3.legend()
ax4.plot(y_kmin['t'], S_kmin,"--", label="$K\ =\ $95%$\ K_{ref}$")
ax4.plot(y_kmax['t'], S_kmax, label="$K\ =\ $105%$\ K_{ref}$")
ax4.set_xlabel(r'$time\ [d]$')
ax4.set_ylabel(r'$Sensitivity\ [-]$')
ax4.set_title('K')
ax4.legend()
# plt.show()

# S on K
# TODO: Make a plot S vs t changing K+/-5%
# plt.figure(4)


# NS
# TODO: Make a plot NS vs. t, changing r & K +/- 5%
fig3, (ax5, ax6,ax7) = plt.subplots(1, 3)
fig3.suptitle('Normalized Sensitivity Analysis')
#plt.figure(1)
ax5.plot(y_rmin['t'], NS_rmin,"--", label="$r\ =\ $95%$\ r_{ref}$")
ax5.plot(y_rmax['t'], NS_rmax, label="$r\ =\ $105%$\ r_{ref}$")
# ax5.set_xlabel(r'$time\ [d]$')
# ax5.set_ylabel(r'$Sensitivity\ [-]$')
# ax5.set_title('r')
# ax5.legend()
ax5.plot(y_kmin['t'], NS_kmin,"--", label="$K\ =\ $95%$\ K_{ref}$")
ax5.plot(y_kmax['t'], NS_kmax, label="$K\ =\ $105%$\ K_{ref}$")
ax5.set_xlabel(r'$time\ [d]$')
ax5.set_ylabel(r'$Sensitivity\ [-]$')
ax5.set_title('NS')
ax5.legend()
ax6.plot(y_rmin['t'], NS_p_rmin,"--", label="$r\ =\ $95%$\ r_{ref}$")
ax6.plot(y_rmax['t'], NS_p_rmax, label="$r\ =\ $105%$\ r_{ref}$")
ax6.plot(y_kmin['t'], NS_p_kmin,"--", label="$K\ =\ $95%$\ K_{ref}$")
ax6.plot(y_kmax['t'], NS_p_kmax, label="$K\ =\ $105%$\ K_{ref}$")
ax6.set_xlabel(r'$time\ [d]$')
ax6.set_ylabel(r'$Sensitivity\ [-]$')
ax6.set_title('$NS_{p}$')
ax6.legend()
ax7.plot(y_rmin['t'], SNS_r,"--", label="r")
ax7.plot(y_kmin['t'], SNS_k,"--", label="K")
ax7.set_xlabel(r'$time\ [d]$')
ax7.set_ylabel(r'$Sensitivity\ [-]$')
ax7.set_title('$SNS_{p}$')
ax7.legend()
plt.show()

