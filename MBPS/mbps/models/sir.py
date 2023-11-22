#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Your team names

Class for disease SIR model
"""
import numpy as np

from classes.module import Module
from functions.integration import fcn_euler_forward

class SIR(Module):
    """ Module for disease spread
    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I
    Parameters
    tsim : array
        Sequence of time points for the simulation
    dt : float
        Time step size [d]
    x0 : dictionary
        Initial conditions of the state variables \n
        * S : Susceptible Population
        * I : Infected Population
        * R : Recovered Population
    p : dictionary of scalars
        Model parameters \n
        * beta : Transmission Rate [d-1]
        * gamma : Recovery Rate [d-1]
    Add here the parameters required to initialize your object
    
    Returns
    -------
    Add here the model outputs returned by your object
    
    """
    # Initialize object. Inherit methods from object Module
    # TODO: fill in the required code
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)
    
    # Define system of differential equations of the model
    # TODO: fill in the required code.
    '''Explanation
    Notice that for the function diff, we use _t and _y0.
    This underscore (_) notation is used to define internal variables,
    which are only used inside the function.
    It is useful here to represent the fact that _t and _y0 are changing
    iteratively, every time step during the numerical integration
    (in this case, called by 'fcn_euler_forward')
    '''
    def diff(self,_t,_y0):
        # State variables
        S_x1, I_x2, R_x3 = _y0[0], _y0[1], _y0[2]
        # Parameters
        beta = self.p['beta']
        gamma = self.p['gamma']
        # Differential equations
        dS_dt = -beta*S_x1*I_x2 #[Suceptible d-1]
        dI_dt = beta*S_x1*I_x2 - gamma*I_x2
        dR_dt = gamma*I_x2
        return np.array([dS_dt,dI_dt,dR_dt])
        
    # Define model outputs from numerical integration of differential equations
    # This function is called by the Module method 'run'.
    # The model does not use disturbances (d), nor control inputs (u).
    # TODO: fill in the required code
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt
        diff = self.diff

        # initial conditions
        s0 = self.x0['S']
        i0 = self.x0['I']
        r0 = self.x0['R']        # Numerical integration
        # (for numerical integration, y0 must be numpy array,
        # even for a single state variable)
        y0 = np.array([s0, i0, r0])
        y_int = fcn_euler_forward(diff, tspan, y0, h=dt)
        # Retrieve results from numerical integration output
        t = y_int['t']
        s, i, r = y_int['y'][0,:],y_int['y'][1,:],y_int['y'][2,:]
        return {'t':t, 'S':s, 'I':i, 'R':r}
