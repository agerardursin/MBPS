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
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

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
def Pvs(p0):
    cnst = p0
    Pvs_out = np.exp(21.3 - 5304/(T+273.0))
    Pvs_out += cnst
    return Pvs_out

def fcn_residuals(p0):
    pvs = Pvs(p0)
    f_interp = interp1d(T,pvs)
    pvs_k = f_interp(T_data)
    err = Pvs_data - pvs_k
    return err

delta = (5304/(T+273)**2) * np.exp(21.3 - 5304/(T+273))
# it = 0
rel_error = 0
# rel_error = np.empty([1, 4])
# for i in range(0, T.size):
#     if T[i] < T_data[0]:
#         Pvs[i] += cnst[0] * (T[i]/T_data[0])
#     elif T_data[0] <= T[i] < T_data[1]:
#         Pvs[i] += (cnst[0] * (T_data[0]/T[i]) + cnst[1] * (T[i]/T_data[1]))*0.5
#     elif T_data[1] <= T[i] < T_data[2]:
#         Pvs[i] += (cnst[1] * (T_data[1]/T[i]) + cnst[2] * (T[i]/T_data[2]))*0.5
#     elif T_data[2] <= T[i] < T_data[3]:
#         Pvs[i] += (cnst[2] * (T_data[2]/T[i]) + cnst[3] * (T[i]/T_data[3]))*0.5
#     else:
#         Pvs[i] += cnst[3] * (T[i]/T_data[3])
#     if np.isin(T[i],T_data):
#         idx = np.asarray(np.where(T_data == T[i]))
#         ind = idx[0][0]
#         rel_error += (Pvs_data[ind] - Pvs[i])/Pvs_data[ind]
p0 = cnst[0]

y_lsq = least_squares(fcn_residuals,p0)
cnst_hat = y_lsq['x']
pvs_hat = Pvs(cnst_hat)
res = y_lsq['fun']
cost = y_lsq['cost']
# rel_error = rel_error/4
# Exercise 2. ET0
ET0 = alpha*Rn*delta/(delta+gamma)

# Relative error
# TODO: Calculate the average relative error
# Tip: The numpy functions np.isin() or np.where() can help you retrieve the
# modelled values for Pvs at the corresponding value for T_data.
T_data_plt = [(x + 273.0) for x in T_data]
T += 273

np.set_printoptions(formatter={'float_kind':'{:.3E}'.format})
print('Residuals \n {} \n'.format(res))
print('Cost \n {} \n'.format(cost))

# Figures
# TODO: Make the plots
# Exercise 1. Pvs vs. T and Delta vs. T,
fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
ax1.plot(T,pvs_hat,label='$P^{sat}_{vap}$')
ax1.plot(T_data_plt,Pvs_data,'o',label='$P^{sat}_{data}$')
ax1.set_xlabel(r'$Temperature\ [K]$')
ax1.set_ylabel(r'$Saturation Vapor Pressure\ [mbar]$')
ax1.set_title(r'$P^{sat}_{vap}\ vs\ Temperature$')
ax1.legend()
ax2.plot(T,delta,label='$\delta$')
ax2.set_xlabel(r'$Temperature\ [K]$')
ax2.set_ylabel(r'$\delta\ [mbar\ K^{-1}]$')
ax2.set_title(r'$\delta\ vs\ Temperature$')
ax2.legend()
ax3.plot(T,delta,label='$ET_{0}$')
ax3.set_xlabel(r'$Temperature\ [K]$')
ax3.set_ylabel(r'$ET_{0}\ [mm\ d^{-1}]$')
ax3.set_title(r'$ET_{0}\ vs\ Temperature$')
ax3.legend()
plt.show()

# Exercise 2. ET0 vs. T
