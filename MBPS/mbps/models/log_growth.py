#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Class for logistic growth model
"""
import numpy as np
import time

from MBPS_startcourse.mbps.classes.module import Module
from MBPS_startcourse.mbps.functions.integration import fcn_euler_forward, fcn_rk4

class LogisticGrowth(Module):
    """ Module for logistic growth (differential equation)
    
    Parameters
    ----------
    tsim : array
        Sequence of time points for the simulation
    dt : float
        Time step size [d]
    x0 : dictionary
        Initial conditions of the state variables \n
        * m : mass [gDM]
    p : dictionary of scalars
        Model parameters \n
        * r : relative growth rate [d-1]
        * K : maximum carrying capacity [gDM m-2]
    
    Returns
    -------
    m : array_like
        Time series for mass growth
    """
    # Initialize object. Inherit methods from object Module
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)
    
    # Define differential equation of the model
    def diff(self,_t,_y0):
        # State variable
        m = _y0
        # Parameters
        r, K = self.p['r'], self.p['K']
        # Differential equation
        dm_dt = r*m*(1-m/K)
        return dm_dt
        
    # Define model outputs from numerical integration of differential equations
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with sistem of differential equations
        m0 = self.x0['m']   # initial conditions
        # Initial conditions
        # Numerical integration
        # (for numerical integration, y0 must be numpy array,
        # even for a single state variable)
        y0 = np.array([m0,])
        y_ef = fcn_euler_forward(diff,tspan,y0,h=dt)    # Euler-forward
        y_rk = fcn_rk4(diff,tspan,y0,h=dt)              # Runge-Kutta
        # Retrieve results from numerical integration output
        t_ef = y_ef['t']        # time
        m_ef = y_ef['y'][0,:]   # first output (row 0)
        t_rk = y_rk['t']        # time
        m_rk = y_rk['y'][0,:]   # first output (row 0)
        return {'t':t_ef, 'm':m_ef,
                 't_rk':t_rk, 'm_rk':m_rk}