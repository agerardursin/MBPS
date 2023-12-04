# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Class for soil water model
"""
import numpy as np
import sys
sys.path.append('../mbps/classes/')
sys.path.append('../mbps/functions/')
from classes.module import Module
from functions.integration import fcn_euler_forward

class Water(Module):
    ''' 
    Water balance in soil as modelled by Castellaro et al. (2010).
    Model description in [mm] for water volume, and [d] for time.
    
    Assumptions
    -----------
    * 3 layers of soil
    * Priestley-Taylor model for reference evapotranspiration
    * Model for dPvap,sat/dT follows equation provided (unknown source)
    * Albedo is constant and represents average crop-soil
    * mulch cover is constant
    * Initial conditions L0 must be in the range [pwp,fc]
    
    Parameters
    ----------
    t : array
        A sequence of time points for the simulation [d]
    dt : scalar
        Time step for the numerical integration [d]
    x0 : dictionary of floats
        Initial conditions of the state variables
        
        ======  =============================================================
        key     meaning
        ======  =============================================================
        'L1'    [mm] Water level in soil layer 1
        'L2'    [mm] Water level in soil layer 2
        'L3'    [mm] Water level in soil layer 3
        'DSD'   [d] Days since damping
        ======  =============================================================
        
    p : dictionary of floats
        Model parameters
        
        ============  =========================================================
        key           meaning
        ============  =========================================================
        'S'           [mm d-1] parameter of precipitation retention
        'alpha'       [mm J-1] Priestley-Taylor parameter
        'gamma'       [mbar °C-1] Psychrometric constant
        'alb'         [-] Albedo (assumed constant crop & soil)
        'kcrop'       [-] Evapotransp coefficient, range (0.85-1.0)
        'WAIc'        [-] WAI critical, range (0.5-0.8)
        'theta_fc1'   [-] Field capacity of soil layer 1 in volume fraction
        'theta_fc2'   [-] Field capacity of soil layer 2 in volume fraction
        'theta_fc3'   [-] Field capacity of soil layer 3 in volume fraction
        'theta_pwp1'  [-] Permanent wilting point of layer 1, vol. fraction
        'theta_pwp2'  [-] Permanent wilting point of layer 2, vol. fraction
        'theta_pwp3'  [-] Permanent wilting point of layer 3, vol. fraction
        'D1'          [mm] Depth of Soil layer 1
        'D2'          [mm] Depth of Soil layer 2
        'D3'          [mm] Depth of Soil layer 3
        'krf1'        [-] Rootfraction layer 1
        'krf2'        [-] Rootfraction layer 2
        'krf3'        [-] Rootfraction layer 3
        'mlc'         [-] Fraction of soil covered by mulch
        ============  =========================================================
    
    d : dictionary of floats or arrays
        Model disturbances (required for method 'run'),
        of shape (len(t_d),2) for time and disturbance.
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'I_glb'  [J m-2 d-1] Global irradiance
        'T'      [°C] Environment average daily temperature
        'f_prc'  [mm d-1] Precipitation
        'LAI'    [-] Leaf area index of crop
        =======  ============================================================
        
    u : dictionary of 2D arrays
        Controlled inputs (required for method 'run'),
        of shape (len(t_d),2) for time and controlled input.
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'f_Irg'  [mm d-1] Irrigation
        =======  ============================================================ 
        
    Returns
    -------
    y : dictionary of arrays
        Model outputs
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'L1'     [mm] Water content of soil layer 1
        'L2'     [mm] Water content of soil layer 2
        'L3'     [mm] Water content of soil layer 3
        'DSD'    [d] Days since damp
        'WAI'    [-] Water availability index (for use in crop models)
        =======  ============================================================
        
    f : dictionary of arrays
        Volumetric flow rates (generated by method 'run')
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'f_Pe'   [mm d-1] Effective precipitation
        'f_Ev'   [mm d-1] Evaporation
        'f_Irg'  [mm d-1] Irrigation
        'f_Tr1'  [mm d-1] Transpiration from top layer
        'f_Tr2'  [mm d-1] Transpiration from mid layer
        'f_Tr3'  [mm d-1] Transpiration from bottom layer
        'f_Dr1'  [mm d-1] Drainage from top layer
        'f_Dr2'  [mm d-1] Drainage from mid layer
        'f_Dr3'  [mm d-1] Drainage from bottom layer
        =======  ============================================================
        
    References
    ----------    
    * Castellaro, G., Morales, L., Ahumada, M., & Barozzi, A. (2010).
      Simulation of dry matter productivity and water dynamics in a Chilean
      Patagonian range.
      Chilean Journal of Agricultural Research, 70(3), 417-427.
      
    * Mohtar, R.H., Buckmaster, D.R., & Fales, S.L. (1997),
      A grazing simulation model: GRASIM A: Model development.
      Transactions of the ASAE, 40(5), 1483-1493.
    '''
    def __init__(self, tsim, dt, x0, p):
        Module.__init__(self, tsim, dt, x0, p)
        # Initialize dictionary of flows
        self.f = {}
        self.f_keys = ('f_Pe', 'f_Tr1', 'f_Tr2', 'f_Tr3',
                       'f_Ev', 'f_Dr1', 'f_Dr2', 'f_Dr3', 'f_Irg')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
    
    def diff(self, _t, _x0):
        # -- State variables (initial conditions)
        L1 = _x0[0]     # [mm] Water in soil layer 1
        L2 = _x0[1]     # [mm] Water in soil layer 2
        L3 = _x0[2]     # [mm] Water in soil layer 3
        DSD = _x0[3]    # [d] Days since damping
        L_arr = np.array([L1,L2,L3])
        
        # -- Parameters
        S = self.p['S']           # [mm d-1] parameter of precipitation retention
        alpha = self.p['alpha']   # [mm J-1] Priestley-Taylor parameter
        gamma = self.p['gamma']   # [mbar °C-1] Psychrometric constant
        alb = self.p['alb']       # [-] Albedo of soil
        kcrop = self.p['kcrop']   # [-] Evapotranspiration coefficient
        WAIc = self.p['WAIc']     # [-] Critical water value for water availability index
        theta_fc1 = self.p['theta_fc1']   # [-] Field capacity of soil layer 1
        theta_fc2 = self.p['theta_fc2']   # [-] Field capacity of soil layer 2
        theta_fc3 = self.p['theta_fc3']   # [-] Field capacity of soil layer 3
        theta_pwp1 = self.p['theta_pwp1'] # [-] Permanent wilting point of soil layer 1 
        theta_pwp2 = self.p['theta_pwp2'] # [-] Permanent wilting point of soil layer 2
        theta_pwp3 = self.p['theta_pwp3'] # [-] Permanent wilting point of soil layer 3
        D1 = self.p['D1']         # [mm] Depth of Soil layer 1
        D2 = self.p['D2']         # [mm] Depth of soil layer 2
        D3 = self.p['D3']         # [mm] Depth of soil layer 3
        krf1 = self.p['krf1']     # [-] Rootfraction layer 1
        krf2 = self.p['krf2']     # [-] Rootfraction layer 2
        krf3 = self.p['krf3']     # [-] Rootfraction layer 3
        mlc = self.p['mlc']     # [-] Fraction of soil covered by mulch
        
        # -- Disturbances at instant _t
        I_glb, LAI = self.d['I_glb'], self.d['LAI']
        T, f_prc = self.d['T'], self.d['f_prc']
        # Global irradiance [J m-2 d-1] 
        _I_glb = np.interp(_t, I_glb[:,0], I_glb[:,1])
        # Environment temperature [°C] 
        _T = np.interp(_t, T[:,0], T[:,1])
        # Precipitation [mm d-1]
        _f_prc = np.interp(_t, f_prc[:,0], f_prc[:,1])  
        # Leaf area index [-]
        _LAI = np.interp(_t, LAI[:,0], LAI[:,1])
        
        # -- Controlled inputs
        f_Irg = self.u['f_Irg']    # [mm d-1] Irrigation
        
        # -- Supporting equations
        # [mm] Field capacities
        fc1, fc2, fc3 = theta_fc1*D1, theta_fc2*D2, theta_fc3*D3
        fc_arr = np.array([fc1, fc2, fc3])
        # [mm] Permanent wilting points
        pwp1, pwp2, pwp3 = theta_pwp1*D1, theta_pwp2*D2, theta_pwp3*D3  
        pwp_arr = np.array([pwp1, pwp2, pwp3])
        
        # - Effective precipitation
        f_Ru = 0.0 # Runoff
        if _f_prc + f_Irg > 0.2*S:
            f_Ru = (_f_prc +f_Irg - 0.2*S)**2/(_f_prc + f_Irg + 0.8*S)
        f_Pe = _f_prc - f_Ru + f_Irg
        
        # - Reference and potential evapotranspiration
        # Priestley-Taylor equation
        Rn = 0.408*_I_glb*(1.0-alb)   # [J m-2 d-1 ] Net irradiance
        Tk = _T + 273.15 # [K] Temperature env. from [°C] to [K]
        Delta = 5304/Tk**2 * np.exp(21.3 - 5304/Tk) # [mbar °C-1] dPvapsat/dT
        ET0 = alpha*Rn*Delta/(Delta+gamma)  # [mm d-1] Reference evapotransp
        ETp = kcrop*ET0 # [mm d-1] Potential evapotranspiration
        
        # - Plant transpiration
        kra = 0.0408*np.exp(0.19*_T) # [-] Root activity
        # Restriction of transpiration (krt) per layer
        WAI_arr = (L_arr-pwp_arr)/(fc_arr-pwp_arr)
        WAI_arr = np.maximum(WAI_arr,np.zeros(3,))
        krt_arr = np.ones((3,))
        krt_arr[WAI_arr<WAIc] = WAI_arr[WAI_arr<WAIc]/WAIc
        # Root fractions (krf)
        krf_arr = np.array([krf1, krf2, krf3])
        # Potential transpiration
        kTr = 1.0 - np.exp(-0.6*_LAI) # [-] Transpiration fraction from ETp
        Tp = kTr*ETp # [mm d-1] Potential transpiration
        # Real transpiration (if Li > pwpi)
        f_Tr_arr = (L_arr>pwp_arr)*kra*krt_arr*krf_arr*Tp # [mm d-1]
        
        # - Soil evaporation
        dt = self.dt
        Ep = min(ETp-Tp, ETp*(1-mlc)) # [mm d-1] Evaporation potential
        f_Ev = Ep/DSD**0.5  # [mm d-1] Real evaporation (if L1 > pwp1)
        # Initial estimation of next water level
        L1_hat = L_arr[0] + dt*(f_Pe - f_Ev - f_Tr_arr[0])
        Ev_switch = L1_hat > 0.0
        f_Ev = Ev_switch*f_Ev
        
        # - Drainage (if Li_final > fc)
        Dr1_switch = L1_hat > fc_arr[0]
        f_Dr1 = Dr1_switch*(L1_hat - fc_arr[0])/dt
        
        Dr2_switch = L_arr[1] + dt*(f_Dr1 - f_Tr_arr[1]) > fc_arr[1]
        f_Dr2 = Dr2_switch*((L_arr[1]-fc_arr[1])/dt + f_Dr1 - f_Tr_arr[1])
        
        Dr3_switch = L_arr[2] + dt*(f_Dr2 - f_Tr_arr[2]) > fc_arr[2]
        f_Dr3 = Dr3_switch*((L_arr[2]-fc_arr[2])/dt + f_Dr2 - f_Tr_arr[2])
        
        f_Dr_arr = np.array([f_Dr1,f_Dr2,f_Dr3])
        
        # Water availability index
        WAI = (L_arr.sum()-pwp_arr.sum()) / (fc_arr.sum()-pwp_arr.sum())
        WAI = max(WAI,0)
        
        # Differential equations [mm d-1]
        dL1_dt = f_Pe - f_Ev - f_Tr_arr[0] - f_Dr_arr[0]
        dL2_dt = f_Dr_arr[0] - f_Tr_arr[1] - f_Dr_arr[1]
        dL3_dt = f_Dr_arr[1] - f_Tr_arr[2] - f_Dr_arr[2]
        # dDSD_dt: slope in [d/d] to be multiplied by dt in fcn_euler_forward
        dDSD_dt = 1 - (f_Pe >= 1.5*Ep)*DSD
        
        # Store flows
        idx = np.isin(np.round(self.t,8), np.round(_t,8))
        self.f['f_Pe'][idx] = f_Pe
        self.f['f_Tr1'][idx] = f_Tr_arr[0]
        self.f['f_Tr2'][idx] = f_Tr_arr[1]
        self.f['f_Tr3'][idx] = f_Tr_arr[2]
        self.f['f_Ev'][idx] = f_Ev
        self.f['f_Dr1'][idx] = f_Dr_arr[0]
        self.f['f_Dr2'][idx] = f_Dr_arr[1]
        self.f['f_Dr3'][idx] = f_Dr_arr[2]
        self.f['f_Irg'][idx] = f_Irg
        
        return np.array([dL1_dt, dL2_dt, dL3_dt, dDSD_dt])
    
    def output(self, tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        L10 = self.x0['L1'] # initial condition
        L20 = self.x0['L2'] # initial condiiton
        L30 = self.x0['L3'] # initial condiiton
        DSD0 = self.x0['DSD'] # initial condiiton
        D1, D2, D3 = self.p['D1'], self.p['D2'], self.p['D3']
        pwp1, fc1 = D1*self.p['theta_pwp1'], D1*self.p['theta_fc1']
        pwp2, fc2 = D2*self.p['theta_pwp2'], D2*self.p['theta_fc2']
        pwp3, fc3 = D3*self.p['theta_pwp3'], D3*self.p['theta_fc3']

        # Numerical integration
        y0 = np.array([L10, L20, L30, DSD0])
        y_int = fcn_euler_forward(diff, tspan, y0, h=dt)
        
        # Model outputs
        t = y_int['t']
        L1 = y_int['y'][0,:]
        L2 = y_int['y'][1,:]
        L3 = y_int['y'][2,:]
        DSD = y_int['y'][3,:]
        WAI = ((L1+L2+L3)-(pwp1+pwp2+pwp3)) / ((fc1+fc2+fc3)-(pwp1+pwp2+pwp3))
        
        return {
            't':t,          # [d] integration time
            'L1':L1,        # [mm] soil water level top layer
            'L2':L2,        # [mm] soil water level mid layer
            'L3':L3,        # [mm] soil water level bottom layer
            'DSD':DSD,      # [d] days since damp
            'WAI':WAI       # [-] water availability index
        }
