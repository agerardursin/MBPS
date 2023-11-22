#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad
"""
import numpy as np

def fcn_euler_forward(diff, t_span, y0, h=1.0):
    """ Function for Euler Forward numerical integration.
    Based on the syntax of scipy.integrate.solve_ivp
    
    This function numerically integrates a system of ordinary differential
    equations, given an initial value::

        dy/dt = f(t, y)
        y(t0) = y0

    Here, t is a one-dimensional independent variable (time), y(t) is an
    n-dimensional vector-valued function (state), and an n-dimensional
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.
    
    Parameters
    ----------
    diff : callable
        Function to calculate the value of d_dt
        (the right-hand side of the system).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    h : float
        Step size for the integration.
    
    Returns
    -------
    Dictionary with the following:\n
    t : ndarray, shape (n_points,)
        Time vector for desired evaluation (based on `t_span` and `h`).
    y : ndarray, shape (n_outputs, n_points)
        Values of the solution at `t`.
    """
    # Number of elements for integration time and model outputs
    nt = int((t_span[1]-t_span[0])/h) + 1
    # Vectors for integration time and model outputs
    tint = np.linspace(t_span[0], t_span[1], nt)
    yint = np.zeros((y0.size, tint.size))
    # Assign initial condition to first element of output vector
    yint[:,0] = y0
    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tint[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Model outputs at next time instant (Euler forward)
        yint[:,idx+1] = y0 + diff(ti,y0)*h
        # Update initial condition for next iteration
        y0 = yint[:,idx+1]
    return {'t':tint, 'y':yint}

def fcn_rk4(diff, t_span, y0, h=1.0):
    """ Function for Runge-Kutta numerical integration.
    Based on the syntax of scipy.integrate.solve_ivp
    
    This function numerically integrates a system of ordinary differential
    equations, given an initial value::

        dy/dt = f(t, y)
        y(t0) = y0

    Here, t is a one-dimensional independent variable (time), y(t) is an
    n-dimensional vector-valued function (state), and an n-dimensional
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.
    
    Parameters
    ----------
    diff : callable
        Function to calculate the value of d_dt
        (the right-hand side of the system).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    h : float
        Step size for the integration.
    
    Returns
    -------
    Dictionary with the following:\n
    t : ndarray, shape (n_points,)
        Time vector for desired evaluation (based on `t_span` and `h`).
    y : ndarray, shape (n_outputs, n_points)
        Values of the solution at `t`.
    """
    # Number of elements for integration time and model outputs
    nt = int((t_span[1]-t_span[0])/h) + 1
    # Vectors for integration time and model outputs
    tint = np.linspace(t_span[0], t_span[1], nt)
    yint = np.zeros((y0.size, tint.size))
    # Assign initial condition to first element of output vector
    yint[:,0] = y0
    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tint[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Slopes
        # TODO: Write down and uncomment the equations for the slopes 
        k1 = diff(ti,y0)
        k2 = diff(ti + h/2.0, y0 + h*k1/2.0)
        k3 = diff(ti + h/2.0, y0 + h*k2/2.0)
        k4 = diff(ti + h, y0 + h*k3)
        # Model outputs at next time instant
        # TODO: Write down and uncomment the equation
        # for the numerical solution
        yint[:,idx+1] = y0 + (h/6)*(k1+2*k2+2*k3+k4)
        # Update initial condition for next iteration
        y0 = yint[:,idx+1]
    return {'t':tint, 'y':yint}
        