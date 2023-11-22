# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- Fill in your team names --

Tutorial: Grass growth model analysis.
3. Temperature index
"""
# TODO: Import the required packages 
import numpy as np
import matplotlib.pyplot as plt

# Temperature index:
# TI = ( (T_max - T)/(T_max - T_opt)) * ((T - T_min)/(T_max - T_opt))^((T_opt - T_min)/(T_max - T_opt)) )^z
def func_TI(T, T_max, T_min, T_opt, z):
    # Define difference variables to simplify computation
    DT_max = T_max - T
    DT_min = T - T_min
    DT_a = T_max - T_opt
    DT_b = T_opt - T_min

    #Define power for the operands
    DT_pow = DT_b/DT_a

    # Define Temperature Index Calculation
    TI = np.power((DT_max/DT_a)*np.power((DT_min/DT_b),DT_pow),z)
    return TI

# TODO: Define the values for the TI parameters
T_min = [-10.0, 0.0, 5.0]
T_max = [32.0, 42.0, 52.0]
T_opt = [10, 20.0, 30.0]
dt = 0.05
z = 1.33
#output = {"T":[],"TI":[],"Tmax":[], "Tmin":[], "Topt":[]}
plt.style.use('ggplot')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# TODO: Define a sensible array for values of T
for tmin in T_min:
    for tmax in T_max:
        temp = np.linspace(tmin, tmax, 1 + int((tmax-tmin)/dt))
        index = 0
        for topt in T_opt:
            #output["T"].append(temp)
            ti = func_TI(temp,tmax,tmin,topt,z)
            # output["Tmax"].append(tmax)
            # output["Tmin"].append(tmin)
            # output["Topt"].append(topt)
            if index == 0:
                ax1.plot(temp, ti, label='$T_{max} = ' + repr(tmax) + '$, '+'$T_{min} = ' + repr(tmin) + '$ ')
                index+=1
            elif index == 1:
                ax2.plot(temp, ti, label='$T_{max} = ' + repr(tmax) + '$, '+'$T_{min} = ' + repr(tmin) + '$ ')
                index+=1
            elif index == 2:
                ax3.plot(temp, ti, label='$T_{max} = ' + repr(tmax) + '$, ' + '$T_{min} = ' + repr(tmin) + '$ ')
                index = 0
            else:
                print("Whoops this shouldn't happen")
# TODO: (Optional) Define support variables DTmin, DTmax, DTa, DTb


# TODO: Define TI


# TODO: Make a plot for TI vs T
ax1.set_xlabel(r'$Temperature\ [' + u'\u2103' + ']$')
ax1.set_ylabel(r'$Temperature\ Index\ [-]$')
ax1.title.set_text(r'$T_{opt}\ = ' + repr(T_opt[0])+ u'\u2103$')
ax1.legend()
ax2.set_xlabel(r'$Temperature\ [' + u'\u2103' + ']$')
ax2.set_ylabel(r'$Temperature\ Index\ [-]$')
ax2.title.set_text(r'$T_{opt}\ = ' + repr(T_opt[1])+ u'\u2103$')
ax2.legend()
ax3.set_xlabel(r'$Temperature\ [' + u'\u2103' + ']$')
ax3.set_ylabel(r'$Temperature\ Index\ [-]$')
ax3.title.set_text(r'$T_{opt}\ = ' + repr(T_opt[2])+ u'\u2103$')
ax3.legend()
plt.show()