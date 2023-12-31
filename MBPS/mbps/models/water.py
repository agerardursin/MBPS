# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR

@authors:   Nynke - write your team names --

Class for soil water model
"""
import numpy as np

from MBPS.mbps.classes.module import Module
from MBPS.mbps.functions.integration import fcn_euler_forward

class Water(Module):
    ''' 
    Water balance in soil as modelled by Castellaro et al. (2010).
    Model description in [mm] for water volume, and [d] for time.
        
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
        ### TODO: Define all necessary supporting equations
        
        # Runoff equation
        
        f_Ru = np.maximum(_f_prc-0.2*S,0)**2/(_f_prc + 0.8*S)
        
        """
        f_Ru = []
        for prc in f_prc:
            fru = (prc-0.2*S)**2/(prc + 0.8*S)
            if prc <= 0.2*S:
                f_Ru.append(0)
            else:
                f_Ru.append(fru)
        """
        """
        if f_prc > 0.2*S:
            f_Ru = (f_prc-0.2*S)**2/(f_prc + 0.8*S)
        else:
            f_Ru = 0
        """
        #Effective precipitation
        f_Pe = _f_prc - f_Ru

        #Evapotranspiration
        Rn = 0.408 * _I_glb * (1 - alb)    # [J m-2 d-1] net radiation
        delta = (5304/(_T+273)**2) * np.exp(21.3 - 5304/(_T+273))    #[mbar degrees C-1] slope of p_sat_vap at T_env
        ET_0 = alpha * Rn * delta / (delta + gamma)     # [mm d-1] reference evapotranspiration
        ET_p = kcrop * ET_0        # [mm d-1] Evapotranspiration potential

        #Transpiration
        WAI1 = max((L1/D1 - theta_pwp1), 0)/(theta_fc1-theta_pwp1) # [-] Water availability index of layer 1
        WAI2 = max((L2/D2 - theta_pwp2), 0)/(theta_fc2-theta_pwp2)
        WAI3 = max((L3/D3 - theta_pwp3), 0)/(theta_fc3-theta_pwp3)
        
        #k_rt1
        if WAI1 < WAIc:
            k_rt1 = WAI1 / WAIc
        else: 
            k_rt1 = 1                    # [-] restriction of transpiration by soil
       #k_rt2
        if WAI2 < WAIc:
            k_rt2 = WAI2 / WAIc
        else: 
            k_rt2 = 1  
            
        #k_rt3
        if WAI3 < WAIc:
            k_rt3 = WAI3 / WAIc
        else: 
            k_rt3 = 1  
        
        k_ra = 0.0408 *np.exp(0.19*(_T))    # [-] root activity
        k_tr = 1 - np.exp(-0.6*_LAI)     # [-] transpiration fraction from ETp   
        T_p = k_tr*ET_p                  # [mm d-1] potential transpiration
        
        f_Tr1 = k_rt1 * k_ra * krf1 * T_p
        f_Tr2 = k_rt2 * k_ra * krf2 * T_p
        f_Tr3 = k_rt3 * k_ra * krf3 * T_p
        

        # Evaporation
        Ep = np.minimum((ET_p-T_p), (ET_p*(1-mlc)))    #[mm d-1] Evaporation potential
        f_Ev = Ep / np.sqrt(DSD)    # [mm d-1] Evaporation

        #Drainage layer 1
        if L1 > theta_fc1*D1:
            f_Dr1 = L1-theta_fc1*D1
        else:
            f_Dr1 = 0
            
        #Drainage layer 2
        if L2 > theta_fc2*D2:
            f_Dr2 = L2-theta_fc2*D2
        else:
            f_Dr2 = 0
            
        #Drainage layer 3
        if L3 > theta_fc3*D3:
            f_Dr3 = L3-theta_fc3*D3
        else:
            f_Dr3 = 0

        # -- Differential equations [mm d-1]
        ### TODO: define the differential equations
        ### (use the flow names indicated below)
        dL1_dt = f_Pe - f_Tr1 - f_Ev - f_Dr1
        dL2_dt = f_Dr1 - f_Tr2 - f_Dr2
        dL3_dt = f_Dr2 - f_Tr3 - f_Dr3
        dDSD_dt = 1 - (f_Pe >= 1.5*Ep)*DSD
        
        # Store flows
        idx = np.isin(self.t, _t)
        self.f['f_Pe'][idx] = f_Pe
        self.f['f_Tr1'][idx] = f_Tr1
        self.f['f_Tr2'][idx] = f_Tr2
        self.f['f_Tr3'][idx] = f_Tr3
        self.f['f_Ev'][idx] = f_Ev
        self.f['f_Dr1'][idx] = f_Dr1
        self.f['f_Dr2'][idx] = f_Dr2
        self.f['f_Dr3'][idx] = f_Dr3
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
        }
