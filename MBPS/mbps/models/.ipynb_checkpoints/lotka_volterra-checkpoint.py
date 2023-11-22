# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Class for Lotka-Volterra model, used for predator-prey population dynamics
"""
import numpy as np
from scipy.integrate import solve_ivp
from mbps.classes.module import Module
from mbps.functions.integration import fcn_euler_forward

# Model definition
class LotkaVolterra(Module):
    """ Module for Lotka-Volterra predator prey dynamics::

        dx1/dt = p1*x1 - p2*x1*x2
        dx2/dt = p3*x1*x2 - p4*x2
    
    Parameters
    ----------
    tsim : array
        Sequence of time points for the simulation
    dt : float
        Time step size [d]
    x0 : dictionary
        Initial conditions of the state variables \n
        * pred : number of predators
        * prey : number of preys
    p : dictionary of scalars
        Model parameters \n
        * p1 : Birth rate of preys [d-1]
        * p2 : Death rate of preys due to predation [pred-1 d-1]
        * p3 : Birth rate of predators facilitated by predation [prey-1 d-1]
        * p4 : Death rate of predators [d-1]
        
    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays ('prey', 'pred'), 
        and the evaluation time 't'.
    """
    # Initialize object. Inherit methods from object Module
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)

    # Define system of differential equations of the model
    def diff(self,_t,_y0):
        # State variables
        x1, x2 = _y0[0], _y0[1]
        # Parameters
        p1, p2 = self.p['p1'], self.p['p2']
        p3, p4 = self.p['p3'], self.p['p4']
        # Differential equations
        dx1_dt = p1*x1 - p2*x1*x2   # [prey d-1]
        dx2_dt = p3*x1*x2 - p4*x2   # [pred d-1]
        return np.array([dx1_dt,dx2_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        prey0 = self.x0['prey'] # initial condition
        pred0 = self.x0['pred'] # initial condiiton
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([prey0,pred0])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # TODO: add a second integration output from solve_ivp
        # Note: you must import the function solve_ivp from scipy,
        # at the top of this file.
        # See the help of solve_ivp, and pay special attention to
        # the argument t_eval.
        y_int2 = solve_ivp(diff, tspan, y0, t_eval=y_int['t'], method='RK45')
        
        # Retrieve results from numerical integration output
        # TODO: retrieve the results from y_int2
        t = y_int['t']              # time
        prey = y_int['y'][0,:]      # first output (row 0, all columns)
        pred = y_int['y'][1,:]      # second output (row 1, all columns)

        t_2 = y_int2['t']
        prey2 = y_int2['y'][0,:]
        pred2 = y_int2['y'][1,:]

        # TODO: add the model outputs from y_int2 (solve_ivp)
        return {'t':t, 'prey':prey, 'pred':pred, 't_sp':t_2, 'prey_sp':prey2, 'pred_sp':pred2}