# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- Fill in your team names --

Tutorial: Soil water model analysis.
1. Slope of saturation vapour pressure.
2. Reference evapotranspiration.
"""
# TODO: import the required packages
import numpy as np
import matplotlib.pyplot as plt

# TODO: specify the matplotlib style
plt.style.use('ggplot')

# Measured saturation vapour pressure
# TODO: define arrays for the measured data T and Pvs.
# Specify the units in comments
T_data = [10.0, 20.0, 30.0, 40.0]
Pvs_data = [12.28, 23.39, 42.46, 73.84]
cnst = [-0.629332895268684, -1.083990013098273, -2.020438352923272, -3.973356905732089]

# Air temperature [K]
# TODO: define an array for sesnible values of T
T = np.linspace(0, 42, 421)

# Model parameteres and variables
alpha = 1.291   # [mm MJ-1 (m-2)] Priestley-Taylor parameter
gamma = 0.68    # [mbar °C-1] Psychrometric constant
Irr_gl = 18.0   # [MJ m-2 d-2] Global irradiance
alb = 0.23      # [-] albedo (crop)
Rn = 0.408*Irr_gl*1-(alb)  # [MJ m-2 d-1] Net radiation

# Model equations
# TODO: Define variables for the model
# Exercise 1. Pvs, Delta
Pvs = np.exp(21.3 - 5304/(T+273.0))
delta = (5304/(T+273)**2) * np.exp(21.3 - 5304/(T+273))
# it = 0
rel_error = 0
# rel_error = np.empty([1, 4])
for i in range(0, T.size):
    if T[i] < T_data[0]:
        Pvs[i] += cnst[0] * (T[i]/T_data[0])
    elif T_data[0] <= T[i] < T_data[1]:
        Pvs[i] += (cnst[0] * (T_data[0]/T[i]) + cnst[1] * (T[i]/T_data[1]))*0.5
    elif T_data[1] <= T[i] < T_data[2]:
        Pvs[i] += (cnst[1] * (T_data[1]/T[i]) + cnst[2] * (T[i]/T_data[2]))*0.5
    elif T_data[2] <= T[i] < T_data[3]:
        Pvs[i] += (cnst[2] * (T_data[2]/T[i]) + cnst[3] * (T[i]/T_data[3]))*0.5
    else:
        Pvs[i] += cnst[3] * (T[i]/T_data[3])
    if np.isin(T[i],T_data):
        idx = np.asarray(np.where(T_data == T[i]))
        ind = idx[0][0]
        rel_error += (Pvs_data[ind] - Pvs[i])/Pvs_data[ind]
T += 273
rel_error = rel_error/4
# Exercise 2. ET0


# Relative error
# TODO: Calculate the average relative error
# Tip: The numpy functions np.isin() or np.where() can help you retrieve the
# modelled values for Pvs at the corresponding value for T_data.


# Figures
# TODO: Make the plots
# Exercise 1. Pvs vs. T and Delta vs. T,
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(T,Pvs,label='$P^{sat}_{vap}$')
ax1.set_xlabel(r'$Temperature\ [K]$')
ax1.set_ylabel(r'$Saturation Vapor Pressure\ [mbar]$')
ax1.set_title(r'$P^{sat}_{vap}\ vs\ Temperature$')
ax1.legend()
ax2.plot(T,delta,label='$\delta$')
ax2.set_xlabel(r'$Temperature\ [K]$')
ax2.set_ylabel(r'$\delta\ [mbar\ K^{-1}]$')
ax2.set_title(r'$\delta\ vs\ Temperature$')
ax2.legend()
plt.show()

# Exercise 2. ET0 vs. T
