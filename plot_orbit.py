# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:28:56 2025

@author: ddiaz.beca
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import spiceypy as spice
plt.style.use('dark_background')

def setup_3d_plot():
    "Setup 3d Plot"
    fig=plt.figure(figsize=(18,6))
    ax=fig.add_subplot(111, projection='3d', computed_zorder=False)
    return fig, ax
def plot_3D(ax, trajectories, body_radius, colors, labels, df_states, eclipsing_bodies, save_plot=True):
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
        partial_eclipse_color='orange'
        total_eclipse_color='red'
        label = labels[i]
        default_color = colors[i]
        default_rgb = mcolors.to_rgb(default_color)
        partial_rgb = (default_rgb[0] * 0.7, default_rgb[1] * 0.7, default_rgb[2] * 0.7)  # Factor de 0.7 para oscurecer
        total_rgb= (default_rgb[0] * 0.4, default_rgb[1] * 0.4, default_rgb[2] * 0.4)
        # Graficar por segmentos, considerando eclipses
        start_index = 0
        for j in range(1, len(rs)):
            # ***Lógica simplificada para detectar cambios de eclipse***
            current_eclipse = False  # Asume que no hay eclipse al principio
            current_partial = False
            current_total = False
            for ecb in eclipsing_bodies:
                eclipse_column = f'Eclipse_status_{ecb}'
                if eclipse_column in df_states.columns:
                    if df_states.loc[j, eclipse_column] == 1:
                        current_eclipse = True
                        current_partial = True
                    elif df_states.loc[j, eclipse_column] == 2:
                        current_eclipse = True
                        current_total = True
                    

            if j > 0:  # Asegura que haya un punto anterior para comparar
                previous_eclipse = False
                previous_partial = False
                previous_total = False
                for ecb in eclipsing_bodies:
                    eclipse_column = f'Eclipse_status_{ecb}'
                    if eclipse_column in df_states.columns:
                        if df_states.loc[j - 1, eclipse_column] == 1:
                            previous_eclipse = True
                            previous_partial = True
                        elif df_states.loc[j - 1, eclipse_column] == 2:
                            previous_eclipse = True
                            previous_total = True
                            
                if current_eclipse != previous_eclipse or current_partial != previous_partial or current_total != previous_total: #If the eclipse status has changed
                    # Grafica el segmento anterior en cuanto cambia el status del eclipse
                    segment_color = default_color
                    if previous_partial:
                        segment_color = partial_rgb
                    elif previous_total:
                        segment_color = total_rgb

                    ax.plot(rs[start_index:j, 0], rs[start_index:j, 1], rs[start_index:j, 2], color=segment_color, zorder=2)
                    start_index = j

        # Graficar el último segmento
        partial_eclipse = False
        total_eclipse = False
        for ecb in eclipsing_bodies:
            eclipse_column = f'Eclipse_status_{ecb}'
            if eclipse_column in df_states.columns:
                if (df_states.loc[start_index:, eclipse_column] == 1).any():
                    partial_eclipse = True
                if (df_states.loc[start_index:, eclipse_column] == 2).any():
                    total_eclipse = True

        if total_eclipse:
            segment_color = total_rgb
        elif partial_eclipse:
            segment_color = partial_rgb
        else:
            segment_color = default_color
        if len(rs)>0:
          ax.plot(rs[start_index:, 0], rs[start_index:, 1], rs[start_index:, 2], color=segment_color, zorder=2)
        
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
        plt.savefig('output/'+'Orbit3D.png', dpi=300)
    #plt.show()
    
        
def setup_coes_plots(n_orbits):
    """Setup for COEs plot"""
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    fig.suptitle('COEs', fontsize=20)
    return fig, axs
def plot_coes(axs, fig, time_arrays, coes_arrays, colors, labels, ets_last, df_states, eclipsing_bodies, datetimes, time_unit='seconds', save_plot=True, title='COEs', show_plot=True):
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
    """
    # Input validationcd ..
    assert len(coes_arrays) == len(colors) == len(labels), "Mismatched number of orbits"
    if  3600*10 <= ets_last < 3600*24*5: #Si el numero de segundos es más que un valor arbitario (10h) lo representamos como horas
        time_unit='hours'
    elif ets_last>= 3600*24*5: #si es mayor que 5 días lo representamos como dias.
        time_unit= 'days'
    # Time scaling configuration
    time_factors = {
        'seconds': 1,
        'hours': 1/3600,
        'days': 1/(3600*24)
    }
    xlabel = f'Time ({time_unit})'
    """
    # Initialize min and max values for each COE to set common limits later
    min_values = np.min(np.array([np.min(coes_array[:, :6], axis=0) for coes_array in coes_arrays]), axis=0)
    max_values = np.max(np.array([np.max(coes_array[:, :6], axis=0) for coes_array in coes_arrays]), axis=0)
    
    for orbit_idx, (times, coes) in enumerate(zip(time_arrays, coes_arrays)):
        # Convert time units
        #scaled_times = times * time_factors[time_unit]
        default_color = colors[orbit_idx]
        default_rgb = mcolors.to_rgb(default_color)
        partial_rgb = (default_rgb[0] * 0.8, default_rgb[1] * 0.8, default_rgb[2] * 0.8)  
        total_rgb= (default_rgb[0] * 0.6, default_rgb[1] * 0.6, default_rgb[2] * 0.6)
        label = labels[orbit_idx]
        formatter = ScalarFormatter(useMathText=True)  # Habilita formato científico
        formatter.set_useOffset(False)                  # Elimina el offset (ej: 1e4)
        #formatter.set_powerlimits((4, 4))               # Fuerza notación científica para exponente 4
        formatter.set_scientific(True)                  # Activa notación científica

        # Ajustar precisión a 3 decimal (manualmente, ya que no hay set_precision)
        formatter._format = "%3e"                      # Formato de 3 decimal
        for subplot_row, subplot_col, coe_index, title_str, ylabel in [
            (0, 0, 5, 'True Anomaly', 'Angle (degrees)'),
            (1, 0, 0, 'Semi-Major Axis', 'a (km)'),
            (0, 1, 1, 'Eccentricity', ''),
            (0, 2, 4, 'Argument of Periapsis', 'Angle (degrees)'),
            (1, 1, 2, 'Inclination', 'Angle (degrees)'),
            (1, 2, 3, 'RAAN', 'Angle (degrees)')
        ]:
            #Format y-axis
            ax = axs[subplot_row, subplot_col]
            ax.yaxis.set_major_formatter(formatter)
            ax.set_title(title_str)
            ax.grid(True)
            ax.set_ylabel(ylabel)
            
            # Graficar por segmentos, considerando eclipses
            start_index = 0
            current_eclipse_status = {}  # Store eclipse status for each eclipsing body
            previous_eclipse_status = {}
            for ecb in eclipsing_bodies:
                eclipse_column = f'Eclipse_status_{ecb}'
                if eclipse_column in df_states.columns:
                        current_eclipse_status[ecb] = df_states.loc[0, eclipse_column]
                        previous_eclipse_status[ecb] = current_eclipse_status[ecb]
            for j in range(1, len(times)):
                current_eclipse_status = {} #Reset the dictionary for each step
                
                        

                for ecb in eclipsing_bodies:
                    eclipse_column = f'Eclipse_status_{ecb}'
                    if eclipse_column in df_states.columns:
                        current_eclipse_status[ecb] = df_states.loc[j, eclipse_column]

                if current_eclipse_status != previous_eclipse_status:  # Any eclipse status change?
                    segment_color = default_color
                    partial_eclipse = any(status == 1 for status in previous_eclipse_status.values())
                    total_eclipse = any(status == 2 for status in previous_eclipse_status.values())

                    if total_eclipse:
                        segment_color = total_rgb
                    elif partial_eclipse:
                        segment_color = partial_rgb

                    ax.plot(datetimes[start_index:j], coes[start_index:j, coe_index], color=segment_color) #Plot with datetimes 
                    start_index = j
                previous_eclipse_status = current_eclipse_status.copy()

            # Graficar el último segmento
            segment_color = default_color
            partial_eclipse = any(status == 1 for status in current_eclipse_status.values())
            total_eclipse = any(status == 2 for status in current_eclipse_status.values())

            if total_eclipse:
                segment_color = total_rgb
            elif partial_eclipse:
                segment_color = partial_rgb
            ax.plot(datetimes[start_index:], coes[start_index:, coe_index], color=segment_color) #Plot with datetimes 
            #ax.plot(scaled_times[start_index:], coes[start_index:, coe_index], color=segment_color)
            # --- Formateo del eje X con matplotlib.dates DESPUÉS de graficar ---
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis))
            xlabel='UTC Time'
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Formato opcional  
            # --- Establecer límites del eje X explícitamente ---
            start_datetime = datetimes[0]  # Tiempo inicial en datetime
            end_datetime = datetimes[-1] # Tiempo final en datetime
            ax.set_xlim(start_datetime, end_datetime)  # Establecer límites
            
    fig.autofmt_xdate() #Auto fix of the date axis.
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
        # Establece los límites del eje X EXPLICITAMENTE para evitar el factor extraño
        #min_x = np.min(np.array(time_arrays) * time_factors[time_unit])  # Calcula el mínimo valor de x
        #max_x = np.max(np.array(time_arrays) * time_factors[time_unit])  # Calcula el máximo valor de x
        #ax.set_xlim([min_x, max_x])  # Establece los límites del eje X
    # Special formatting for semi-major axis
    #axs[1,0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    
    # Unified legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.9))
    if save_plot:
        plt.savefig('output/'+title+'.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
        
def setup_vector_plot():   
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    return fig, axes
# Función para graficar los vectores en un plano 2D
def plot_vectors(vector_list, ax, i, color, vector_name, title, margin, save_plot=True):
    
    vector_array = np.array(vector_list)
    # Graficar los vectores (CORREGIDO)
    x = vector_array[:, 0]  # Extrae todas las coordenadas x
    y = vector_array[:, 1]  # Extrae todas las coordenadas y
    ax.scatter(x[0], y[0], color='green', s=50, label='First point'if i == 0 else None) #Primer punto. Leyenda si es la primera orbita
    ax.scatter(x[-1], y[-1], color='red', s=50, label='Last point' if i == 0 else None) #ultimo punto. Leyenda si es la primera orbita
    ax.plot(x, y, color=color, linewidth=0.5, label=vector_name) # Un solo scatter para todos los puntos
    


    
    
    # Personalizar el gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid()
    ax.legend()
    
    min_value_x=np.min(x)
    max_value_x=np.max(x)
    min_value_y=np.min(y)
    max_value_y=np.max(y)
    ax.set_aspect('equal')  # Para que los ejes tengan la misma escala
    ax.set_xlim([min_value_x-margin, max_value_x+margin])  # Ajustar límites si es necesario
    ax.set_ylim([min_value_y-margin, max_value_y+margin])  # Ajustar límites si es necesario
    
    

    
def plot_i_and_e_vectors(fig, axes, i_vec_all, e_vec_all, orbit_params, save_plot=True, show_plot=True, title='e-i_vectors'):
    """Grafica los vectores i y e en subplots separados."""


    # Valores iniciales para los límites (se calculan dentro de la función)
    min_xi, min_xe = float('inf'), float('inf')
    max_xi, max_xe = float('-inf'), float('-inf')
    min_yi, min_ye = float('inf'), float('inf')
    max_yi, max_ye = float('-inf'), float('-inf')

    # Graficar los vectores i
    for i, i_vec_orbit_list in enumerate(i_vec_all):
        i_vec_orbit_array = np.array(i_vec_orbit_list)
        plot_vectors(i_vec_orbit_array, axes[1], i, orbit_params['colors'][i], f'i_vec orbit {i}', title=f'Inclination vector', margin=0.01)
        # Actualizar los límites
        min_xi = min(min_xi, np.min(i_vec_orbit_array[:, 0]))
        max_xi = max(max_xi, np.max(i_vec_orbit_array[:, 0]))
        min_yi = min(min_yi, np.min(i_vec_orbit_array[:, 1]))
        max_yi = max(max_yi, np.max(i_vec_orbit_array[:, 1]))

    # Graficar los vectores e
    for i, e_vec_orbit_list in enumerate(e_vec_all):
        e_vec_orbit_array = np.array(e_vec_orbit_list)
        plot_vectors(e_vec_orbit_array, axes[0], i, orbit_params['colors'][i], f'e_vec orbit {i}', title=f'Excentricity vector', margin=0.001)
        min_xe = min(min_xe, np.min(e_vec_orbit_array[:, 0]))
        max_xe = max(max_xe, np.max(e_vec_orbit_array[:, 0]))
        min_ye = min(min_ye, np.min(e_vec_orbit_array[:, 1]))
        max_ye = max(max_ye, np.max(e_vec_orbit_array[:, 1]))

    # Establecer los límites del gráfico
    margin_i = 0.001
    margin_e = 0.0001
    axes[0].set_xlim([min_xe - margin_e, max_xe + margin_e])
    axes[0].set_ylim([min_ye - margin_e, max_ye + margin_e])
    axes[1].set_xlim([min_xi - margin_i, max_xi + margin_i])
    axes[1].set_ylim([min_yi - margin_i, max_yi + margin_i])
    
    plt.tight_layout()
    if save_plot:
        plt.savefig('output/'+title+'.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
