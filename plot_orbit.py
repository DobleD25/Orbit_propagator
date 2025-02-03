# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:28:56 2025

@author: ddiaz.beca
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import ScalarFormatter
plt.style.use('dark_background')

def setup_3d_plot():
    "Setup 3d Plot"
    fig=plt.figure(figsize=(18,6))
    ax=fig.add_subplot(111, projection='3d', computed_zorder=False)
    return fig, ax
def plot_3D(ax, trajectories, body_radius, colors, labels, save_plot=True):
    """
    Plot several orbits.
    Args:
        ax: 3D axis
        trajectories: Array lists with trajectories (N, 3)
        body_radius: central body radius
        colors: Colors list for every trajectory
        labels: labels list for every trajectory
        """
    
    
    
    #plot central body
    _u, _v =np.mgrid[0:2*np.pi:40j , 0:np.pi:20j]
    _x=body_radius.value*np.cos(_u)*np.sin(_v)
    _y=body_radius.value*np.sin(_u)*np.sin(_v)
    _z=body_radius.value*np.cos(_v)
    ax.plot_surface(_x,_y,_z,cmap='Blues',zorder=0)
    # Plot each trajectory
    max_val = 0
    for i, rs in enumerate(trajectories):
        ax.plot(rs[:, 0], rs[:, 1], rs[:, 2], color=colors[i], label=labels[i], zorder=2)
        ax.plot([rs[0, 0]], [rs[0, 1]], [rs[0, 2]], 'o', color=colors[i], zorder=2.01)
        current_max = np.max(np.abs(rs))
        max_val = current_max if current_max > max_val else max_val
    #Common configuration
    l=body_radius.value*2
    x, y, z = [[0, 0, 0], [0,0,0], [0, 0, 0]]
    u, v, w =[[l, 0, 0], [0, l, 0], [0, 0, l]]
    ax.quiver(x, y, z, u, v, w, color ='k')
    
    #Graph limits
    max_val=np.max(np.abs(rs))
    
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    #Labels
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    
    ax.set_title('Orbit Propagator', zorder= 3)
    
    plt.legend()
    if save_plot:
        plt.savefig('Orbit3D.png', dpi=300)
    #plt.show()
    
        
def setup_coes_plots(n_orbits):
    """Setup for COEs plot"""
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    fig.suptitle('COEs', fontsize=20)
    return fig, axs
def plot_coes(axs, time_arrays, coes_arrays, colors, labels, time_unit='seconds', save_plot=True, title='COEs', show_plot=True):
    """
    Plot comparison of classical orbital elements
    
    Args:
        axs: Array of matplotlib axes (2x3 grid)
        time_arrays: List of time arrays for each orbit
        coes_arrays: List of COEs arrays for each orbit
        colors: List of plot colors
        labels: List of trajectory labels
        time_unit: Time unit for x-axis ('seconds', 'hours' or 'days')
    """
    # Input validation
    assert len(coes_arrays) == len(colors) == len(labels), "Mismatched number of orbits"
    
    # Time scaling configuration
    time_factors = {
        'seconds': 1,
        'hours': 1/3600,
        'days': 1/(3600*24)
    }
    xlabel = f'Time ({time_unit})'
    # Initialize min and max values for each COE to set common limits later
    min_values = np.min(np.array([np.min(coes_array[:, :6], axis=0) for coes_array in coes_arrays]), axis=0)
    max_values = np.max(np.array([np.max(coes_array[:, :6], axis=0) for coes_array in coes_arrays]), axis=0)
    
    for orbit_idx, (times, coes) in enumerate(zip(time_arrays, coes_arrays)):
        # Convert time units
        scaled_times = times * time_factors[time_unit]
        color = colors[orbit_idx]
        label = labels[orbit_idx]
        formatter = ScalarFormatter(useMathText=True)  # Habilita formato científico
        formatter.set_useOffset(False)                  # Elimina el offset (ej: 1e4)
        #formatter.set_powerlimits((4, 4))               # Fuerza notación científica para exponente 4
        formatter.set_scientific(True)                  # Activa notación científica

        # Ajustar precisión a 3 decimal (manualmente, ya que no hay set_precision)
        formatter._format = "%3e"                      # Formato de 3 decimal
        # True Anomaly
        axs[0,0].yaxis.set_major_formatter(formatter)
        axs[0,0].plot(scaled_times, coes[:, 5], color = color)
        axs[0,0].set_title('True Anomaly')
        axs[0,0].grid(True)
        axs[0,0].set_ylabel('Angle (degrees)')
        #axs[0, 0].set_ylim([coes[:, 5].min() - 1, coes[:, 5].max() + 1])
        
        # Semi-Major Axis
        axs[1,0].yaxis.set_major_formatter(formatter)
        axs[1,0].plot(scaled_times, coes[:, 0], color=color)
        axs[1,0].set_title('Semi-Major Axis')
        axs[1,0].grid(True)
        axs[1,0].set_ylabel('a (km)')
       
        # Eccentricity
        axs[0,1].yaxis.set_major_formatter(formatter)
        axs[0,1].plot(scaled_times, coes[:, 1], color=color)
        axs[0,1].set_title('Eccentricity')
        axs[0,1].grid(True)
        #axs[0,1].set_ylim([coes[:, 1].min() - 0.01, coes[:, 1].max() + 0.01])
        #axs[0,1].plot(scaled_times, coes[:,1], color=color)
        
        # Argument of Periapsis
        axs[0,2].plot(scaled_times, coes[:,4], color=color)
        axs[0,2].set_title('Argument of Periapsis')
        axs[0,2].grid(True)
        axs[0,2].set_xlabel(xlabel)
        axs[0,2].set_ylabel('Angle (degrees)')
        axs[0,2].yaxis.set_major_formatter(formatter)
        #axs[0, 2].set_ylim([coes[:, 4].min() - 0.1, coes[:, 4].max() + 0.1])
        # Inclination
        axs[1,1].plot(scaled_times, coes[:,2], color=color)
        axs[1,1].set_title('Inclination')
        axs[1,1].grid(True)
        axs[1,1].set_ylabel('Angle (degrees)')
        
        #axs[1,1].set_ylim([coes[:, 2].min() - 0.1, coes[:, 2].max() + 0.1])
        axs[1,1].yaxis.set_major_formatter(formatter)
        # RAAN
        axs[1,2].set_title('RAAN')
        axs[1,2].grid(True)
        
        axs[1,2].set_ylabel('Angle (degrees)')
        axs[1,2].plot(scaled_times, coes[:,3], color=color)
        axs[1,2].yaxis.set_major_formatter(formatter)
                
    # Set common limits for each subplot based on min and max values across all orbits
    axs[0,0].set_ylim([min_values[5] - 5, max_values[5] + 5]) #True anomaly
    axs[1,0].set_ylim([min_values[0] - 10, max_values[0] + 10]) #Semi-major axis
    axs[0,1].set_ylim([min_values[1] - 0.01, max_values[1] + 0.01]) #eccentricy
    axs[0,2].set_ylim([min_values[4] - 5, max_values[4] + 5]) #argument of periapsis
    axs[1,1].set_ylim([min_values[2] - 1, max_values[2] + 1]) #inclination
    axs[1,2].set_ylim([min_values[3] - 5, max_values[3] + 5]) #RAAN
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # Common formatting
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel(xlabel)
    
    # Special formatting for semi-major axis
    #axs[1,0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    
    # Unified legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig = axs[0,0].figure
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.9))
    if save_plot:
        plt.savefig(title+'.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
        
    
        """
def plot_coes(ts, coes, hours= False, days=False, show_plot=True, , title= 'COEs', figsize=(16,8)):
    print('Plotting COEs...')
    #create figure and axes instances
    fig, axs=plt.subplots(nrows=2, ncols=3, figsize=figsize)
    
    #figure tittle
    fig.suptitle(title, fontsize=20)
    
    #x axis
    if hours:
        ts=ts/3600.0
        xlabel='Time Elapsed (hours)'
    elif days:
        ts=ts/3600.0/24.0
        xlabel='Time Elapsed (days)'
    else: 
        ts=ts
        xlabel='Time Elapsed (seconds)'
    
    #Plot True anomaly
    axs[0,0].plot(ts, coes[:, 5])
    axs[0,0].set_title('True Anomaly vs. Time')
    axs[0,0].grid(True)
    axs[0,0].set_ylabel('Angle (degrees)')
    axs[0, 0].set_ylim([coes[:, 5].min() - 1, coes[:, 5].max() + 1])
    
    #plot semi major axis
    axs[1,0].plot(ts, coes[:, 0])
    axs[1,0].set_title('Semi-Major Axis vs. Time')
    axs[1,0].grid(True)
    axs[1,0].set_ylabel('Semi-Major Axis (km)')
    axs[1,0].set_xlabel(xlabel)
    # Aplicar formato con 1 decimal en notación científica 
    formatter = ScalarFormatter(useMathText=True)  # Habilita formato científico
    formatter.set_useOffset(False)                  # Elimina el offset (ej: 1e4)
    #formatter.set_powerlimits((4, 4))               # Fuerza notación científica para exponente 4
    formatter.set_scientific(True)                  # Activa notación científica

    # Ajustar precisión a 3 decimal (manualmente, ya que no hay set_precision)
    formatter._format = "%3e"                      # Formato de 3 decimal

    axs[1,0].yaxis.set_major_formatter(formatter)
    
   # Ajustar los límites del eje y para el semi-major axis
    axs[1, 0].set_ylim([coes[:, 0].min()-1, coes[:, 0].max()+1])
    

    #plot eccentricity
    
    axs[0,1].plot(ts, coes[:, 1])
    axs[0,1].set_title('Eccentricity vs. Time')
    axs[0,1].grid(True)
    
    axs[0,1].set_ylim([coes[:, 1].min() - 0.01, coes[:, 1].max() + 0.01])
    
    #plot argument of periapsis    
    axs[0,2].plot(ts, coes[:, 4  ])
    axs[0,2].set_title('Argument of Periapsis vs. Time')
    axs[0,2].grid(True)
    axs[0,2].set_ylabel('Angle (degrees)')
    
    axs[0, 2].set_ylim([coes[:, 4].min() - 0.1, coes[:, 4].max() + 0.1])
    #Plot inclination
    axs[1,1].plot(ts, coes[:, 2 ])
    axs[1,1].set_title('Inclination vs. Time')
    axs[1,1].grid(True)
    axs[1,1].set_ylabel('Angle (degrees)')
    axs[1,1].set_xlabel(xlabel)
    axs[1,1].set_ylim([coes[:, 2].min() - 0.1, coes[:, 2].max() + 0.1])
#Plot RAAN
    axs[1,2].plot(ts, coes[:, 3  ])
    axs[1,2].set_title('RAAN vs. Time')
    axs[1,2].grid(True)
    axs[1,2].set_ylabel('Angle (degrees)')
    axs[1,2].set_xlabel(xlabel)
    axs[1,2].set_ylim([coes[:, 3].min() - 0.1, coes[:, 3].max() + 0.1])
    
"""