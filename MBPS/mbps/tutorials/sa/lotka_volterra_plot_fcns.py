# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR

@author:   Daniel Reyes Lastiri

Functions to generate interactive bokeh plots for the Lotka-Volterra model
"""
import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.models import ColumnDataSource, Slider, formatters
from bokeh.layouts import column, gridplot

from mbps.models.lotka_volterra import LotkaVolterra

# Get data to plot by running model
tsim = np.arange(0, 365, 1)
dt = 1
x0 = {'prey':50, 'pred':50}
tspan = (tsim[0], tsim[-1])
p_ref = (1/30, 0.02/30, 0.01/30, 1/30)

def get_data(p1,p2,p3,p4):
    p = {'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4}
    pop = LotkaVolterra(tsim,dt,x0,p)
    ns_df = pop.ns(x0,p)
    return ns_df

# Modify document (e.g. rows or columns layout, not figure alone)
def modify_doc(doc):
    # Object for source of data (initialized instance of call for get_data)
    source=ColumnDataSource(data=get_data(p_ref[0],p_ref[1],p_ref[2],p_ref[3]))
    
    # Plot for model outputs
    ply = figure(width=400, height=300, tools="",
                 toolbar_location=None, min_border=20)
    ply.line(x='index', y='prey_ref_ref', source=source, line_width=2,
            line_color=Category10[3][0], legend_label='prey')
    ply.line(x='index', y='pred_ref_ref', source=source, line_width=2,
            line_color=Category10[3][1], legend_label='pred')
    ply.legend.location='top_left'
    ply.xaxis.axis_label = "time [d]"
    ply.yaxis.axis_label = "Population [#]"
    ply.y_range.start = 0
    ply.y_range.end = 500
    
    # Plots for normalized sensitivities
    # y1 to p1 and p2
    ply1p1p2 = figure(width=400, height=250, tools="",
                      toolbar_location=None, min_border=ply.min_border)
    ply1p1p2.line(x='index', y='prey_p1_-', source=source, line_width=2,
                  line_color='navy', line_dash='dashed', legend_label='p1-')
    ply1p1p2.line(x='index', y='prey_p1_+', source=source, line_width=2,
                  line_color='navy', legend_label='p1+')
    ply1p1p2.line(x='index', y='prey_p2_-', source=source, line_width=2,
                  line_color='firebrick', line_dash='dashed', legend_label='p2-')
    ply1p1p2.line(x='index', y='prey_p2_+', source=source, line_width=2,
                  line_color='firebrick', legend_label='p2+')
    ply1p1p2.yaxis.axis_label = "NS prey [-]"
    # y1 to p3 and p4
    ply1p3p4 = figure(width=400, height=250, tools="",
                      toolbar_location=None, min_border=ply.min_border)
    ply1p3p4.line(x='index', y='prey_p3_-', source=source, line_width=2,
                  line_color='navy', line_dash='dashed', legend_label='p3-')
    ply1p3p4.line(x='index', y='prey_p3_+', source=source, line_width=2,
                  line_color='navy', legend_label='p3+')
    ply1p3p4.line(x='index', y='prey_p4_-', source=source, line_width=2,
                  line_color='firebrick',line_dash='dashed', legend_label='p4-')
    ply1p3p4.line(x='index', y='prey_p4_+', source=source, line_width=2,
                  line_color='firebrick', legend_label='p4+')
    ply1p3p4.yaxis.axis_label = "NS prey [-]"
    ply1p3p4.xaxis.axis_label = "time [d]"
    
    # y2 to p1 and p2
    ply2p1p2 = figure(width=400, height=250, tools="",
                      toolbar_location=None, min_border=ply.min_border)
    ply2p1p2.line(x='index', y='pred_p1_-', source=source, line_width=2,
                  line_color='navy', line_dash='dashed', legend_label='p1-')
    ply2p1p2.line(x='index', y='pred_p1_+', source=source, line_width=2,
                  line_color='navy', legend_label='p1+')
    ply2p1p2.line(x='index', y='pred_p2_-', source=source, line_width=2,
                  line_color='firebrick', line_dash='dashed', legend_label='p2-')
    ply2p1p2.line(x='index', y='pred_p2_+', source=source, line_width=2,
                  line_color='firebrick', legend_label='p2+')
    ply2p1p2.yaxis.axis_label = "NS pred [-]"
    # y2 to p3 and p4
    ply2p3p4 = figure(width=400, height=250, tools="",
                      toolbar_location=None, min_border=ply.min_border)
    ply2p3p4.line(x='index', y='pred_p3_-', source=source, line_width=2,
                  line_color='navy', line_dash='dashed', legend_label='p3-')
    ply2p3p4.line(x='index', y='pred_p3_+', source=source, line_width=2,
                  line_color='navy', legend_label='p3+')
    ply2p3p4.line(x='index', y='pred_p4_-', source=source, line_width=2,
                  line_color='firebrick', legend_label='p4-')
    ply2p3p4.line(x='index', y='pred_p4_+', source=source, line_width=2,
                  line_color='firebrick', line_dash='dashed', legend_label='p4+')
    ply2p3p4.yaxis.axis_label = "NS pred [-]"
    ply2p3p4.xaxis.axis_label = "time [d]"
    
    # Sliders
    fmt = formatters.BasicTickFormatter(use_scientific=True, power_limit_low=-1)
    slider1 = Slider(start=0, end=2/30, value=1/30, step=0.1/30,
                      title="p1 [d-1]", format=fmt)
    slider2 = Slider(start=0, end=0.04/30, value=0.02/30, step=0.002/30,
                      title="p2 [pred-1 d-1]", format=fmt)
    slider3 = Slider(start=0, end=0.02/30, value=0.01/30, step=0.001/30,
                      title="p3 [prey-1 d-1]", format=fmt)
    slider4 = Slider(start=0, end=2/30, value=1/30, step=0.1/30,
                      title="p4 [d-1]", format=fmt)
    
    # Function to call get_data when slider value changes
    def update_data(attrname,old,new):
        source.data = get_data(slider1.value, slider2.value,
                               slider3.value, slider4.value)
    
    # Call assign_data when slider value changes
    slider1.on_change('value',update_data)
    slider2.on_change('value',update_data)
    slider3.on_change('value',update_data)
    slider4.on_change('value',update_data)
    
    # Create document (layout columns in this case)
    #layout = row(column(slider1,slider2,slider3,slider4, pl),column(pl1a,pl1b))
    #doc.add_root(layout)
    sliders = column(slider1,slider2,slider3,slider4)
    grid = gridplot([[sliders, ply], [ply1p1p2, ply2p1p2], [ply1p3p4, ply2p3p4]])
    doc.add_root(grid)
