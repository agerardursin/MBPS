# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   --- Fill in your team names ---

Tutorial: Grass growth model analysis.
1. Light intensity over leaves.
"""
import numpy as np
import matplotlib.pyplot as plt

### 1. Light intensity over leaves
# TODO: Define a function f_Il
# which takes positional arguments l, k, m, I0
# and returns an array for Il
def f_Il (l, k, m, I0):
    Il = (k/(1-m))*I0*np.exp(-k*l)
    return Il

# TODO: define an array with sensible values of leaf area index (l)
# Use the function np.linspace
l = np.linspace(0,2,10001)

# TODO: Code the figures for:
    # 1) Il vs. l, with three test values for k 
    # 2) Il vs. l, with three test values for m
    # 3) Il vs. l, with three test values for I0
k = [0.5, 0.25, 1]
m = [0.1, 0.05, 0.5]
I0 = [100, 50, 1000]
plt.style.use('ggplot')
plt.figure(1)
Il_0 = f_Il(l, k[0], m[0], I0[0])
Il_k2 = f_Il(l, k[1], m[0], I0[0])
Il_k3 = f_Il(l, k[2], m[0], I0[0])

# Il_1 = f_Il(l, k[0], m[0], I0[0])
Il_m2 = f_Il(l, k[0], m[1], I0[0])
Il_m3 = f_Il(l, k[0], m[2], I0[0])

Il_1 = f_Il(l, k[0], m[0], I0[0])
Il_I02 = f_Il(l, k[0], m[0], I0[1])
Il_I03 = f_Il(l, k[0], m[0], I0[2])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(l,Il_k2,label='k = 0.25')
ax1.plot(l,Il_0,label='k = 0.5')
ax1.plot(l,Il_k3,label='k = 1')
ax1.set_xlabel(r'$leaf\ area\ index\ [-]$')
ax1.set_ylabel(r'$Intensity\ over\ leaf\ [W\ m^{-2}]$')
ax1.legend()
ax2.plot(l,Il_m2,label='m = 0.05')
ax2.plot(l,Il_0,label='m = 0.1')
ax2.plot(l,Il_m3,label='m = 0.5')
ax2.set_xlabel(r'$leaf\ area\ index\ [-]$')
ax2.set_ylabel(r'$Intensity\ over\ leaf\ [W\ m^{-2}]$')
ax2.legend()
ax3.plot(l,Il_I02,label='$I_{0} = 50$')
ax3.plot(l,Il_0,label='$I_{0} = 100$')
ax3.plot(l,Il_I03,label='$I_{0} = 1000$')
ax3.set_xlabel(r'$leaf\ area\ index\ [-]$')
ax3.set_ylabel(r'$Intensity\ over\ leaf\ [W\ m^{-2}]$')
ax3.legend()
plt.show()