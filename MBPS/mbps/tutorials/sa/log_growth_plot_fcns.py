# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR

@author:    Daniel Reyes Lastiri

Normalized sensitivities for the logistic growth model
"""
import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.models import ColumnDataSource, Slider, formatters
from bokeh.layouts import column, gridplot

from mbps.models.log_growth import LogisticGrowth

# General module settings
# simulation time, time step, initial conditions, and model parameters
tsim = np.linspace(0.0, 10.0, 10+1)
dt = 0.1
tspan = (tsim[0], tsim[-1])
x0 = {'m':1.0}
p_ref = (1.2, 100.0)

def get_data_lg(r,K):
    p = {'r':r, 'K':K}
    lg = LogisticGrowth(tsim,dt,x0,p)
    ns_df = lg.ns(x0,p)
    return ns_df

# Modify document (e.g. rows or columns layout, not figure alone)
def modify_doc_lg(doc):
    # Object for source of data (initialized instance of call for get_data)
    source=ColumnDataSource(data=get_data_lg(p_ref[0], p_ref[1]))
    
    # Plot for model output
    ply = figure(width=400, height=250, tools="", toolbar_location=None)
    ply.line(x='index', y='m_ref_ref', source=source, line_width=2,
           line_color=Category10[3][0])
    ply.xaxis.axis_label = "time [d]"
    ply.yaxis.axis_label = "m [gDM m-2]"
    ply.y_range.start = 0
    ply.y_range.end = 200
    
    # Plots for normalized sensitivities
    plns = figure(width=400, height=300, tools="", toolbar_location=None)
    plns.line(x='index', y='m_r_-', source=source, line_width=2,
              line_color='navy', line_dash='dashed', legend_label='r-')
    plns.line(x='index', y='m_r_+', source=source, line_width=2,
             line_color='navy', legend_label='r+')
    plns.line(x='index', y='m_K_-', source=source, line_width=2,
             line_color='firebrick', line_dash='dashed', legend_label='K-')
    plns.line(x='index', y='m_K_+', source=source, line_width=2,
             line_color='firebrick', legend_label='K+')
    plns.yaxis.axis_label = "NS [-]"
    plns.xaxis.axis_label = "time [d]"
    plns.legend.location='top_left'
    
    # Add sliders
    fmt = formatters.BasicTickFormatter(use_scientific=True, power_limit_low=-1)
    sl1 = Slider(start=0.1, end=2.0, value=1.2, step=0.1,
                 title="r [d-1]", format=fmt)
    sl2 = Slider(start=10, end=200, value=100, step=10,
                 title="K [kgDM m-2]", format=fmt)
    
    # Function to call get_data when slider value changes
    def update_data(attrname,old,new):
        source.data = get_data_lg(sl1.value, sl2.value)
    
    # Call assign_data when slider value changes
    sl1.on_change('value',update_data)
    sl2.on_change('value',update_data)
    
    # Create document (layout columns in this case)
    sliders = column(sl1,sl2)
    grid = gridplot([[sliders, ply], [plns, None]])
    doc.add_root(grid)
