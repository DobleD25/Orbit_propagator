# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:28:56 2025
This script contains functions for plotting orbit propagation results, including 3D orbits,
classical orbital elements (COEs), and eccentricity/inclination vectors.
@author: ddiaz.beca
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates

plt.style.use("dark_background")


def call_plots(
    datetimes,
    df_states,
    trajectories,
    labels,
    states_kepl_all,
    i_vec_all,
    e_vec_all,
    orbit_params,
    body_params,
    ets,
    t_eval,
    sph_bodyfixed_all,
    current_dir,
    maneuver_datetimes
):
    """
    Calls functions to generate all plots.

    Args:
        datetimes (list): List of datetime objects corresponding to time steps.
        df_states (pandas.DataFrame): DataFrame containing state information.
        trajectories (list): List of trajectory arrays.
        labels (list): List of orbit labels.
        states_kepl_all (list): List of Keplerian state arrays.
        i_vec_all (list): List of inclination vector lists (a temporal list per propagation).
        e_vec_all (list): List of eccentricity vector lists (a temporal list per propagation).
        orbit_params (dict): Dictionary containing orbit parameters.
        body_params (dict): Dictionary containing body parameters.
        ets (numpy.ndarray): Array of ephemeris times relatives to the initial epoch.
        t_eval (numpy.ndarray): Array of evaluation times.
    """
    # Plotting 3D orbit
    fig_3d, ax_3d = setup_3d_plot()
    plot_3D(
        ax_3d,
        trajectories,
        body_params["radius"],
        orbit_params["colors"],
        labels,
        df_states,
        body_params["ecb"],
        )
    fig_3d.savefig(os.path.join("output", "Orbit3D.png"), dpi=300, bbox_inches="tight")

    # Plotting groundtracks:
    lat_lon = sph_bodyfixed_all[:, :, 1:3]
    coastlines_coordinates_file = os.path.join(current_dir, "Data", "coastlines.csv")
    args = {
    "figsize": (18, 9),
    "markersize": 1,
    "labels": [""] * len(lat_lon),
    "colors": ["c", "r", "b", "g", "w", "y"],
    "grid": True,
    "title": "Groundtracks",
    "show": False,
    "filename": False,
    "dpi": 300,
    "legend": True,
    "surface_image": False,
    "surface_body": "earth",
    "plot_coastlines": True,
    }
    plot_groundtracks(lat_lon, args, coastlines_coordinates_file)
    plt.savefig(os.path.join("output", "Groundtracks.png"), dpi=300, bbox_inches="tight")

    # Plotting longitude variation
    lon = np.rad2deg(sph_bodyfixed_all[:, :, 1])
    lon_vel = vel_lon(lon, t_eval)
    plot_lon(lon, lon_vel, t_eval, datetimes)
    plt.savefig(os.path.join("output", "longitude.png"), dpi=300, bbox_inches="tight")

# Plotting the COEs
    fig_coes, axs_coes = setup_coes_plots(len(orbit_params["initial_states"]))
    plot_coes(
    axs_coes,
    fig_coes,
    [ets] * len(orbit_params["initial_states"]),
    states_kepl_all,
    orbit_params["colors"],
    labels,
    t_eval[-1],
    df_states,
    body_params["ecb"],
    datetimes,
    maneuver_datetimes
    )
    fig_coes.savefig(os.path.join("output", "COEs.png"), dpi=300, bbox_inches="tight")

    # Plotting e and i vector:
    fig_vectors, axes_vectors = setup_vector_plot()
    plot_i_and_e_vectors(fig_vectors, axes_vectors, i_vec_all, e_vec_all, orbit_params)
    fig_vectors.savefig(os.path.join("output", "e-i_vectors.png"), dpi=300, bbox_inches="tight")

    plt.show()

def setup_3d_plot():
    "Setup 3d Plot"
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    return fig, ax


def plot_3D(
    ax,
    trajectories,
    body_radius,
    colors,
    labels,
    df_states,
    eclipsing_bodies,
    title="Orbit3D",
    save_plot=True,
):
    """
    Plot several orbits in 3D.

    Args:
        ax (Axes3D): 3D axis object.
        trajectories (list): List of trajectory arrays.
        body_radius (astropy.units.quantity.Quantity): Radius of the central body.
        colors (list): List of colors for each trajectory.
        labels (list): List of labels for each trajectory.
        df_states (pandas.DataFrame): DataFrame containing state information.
        eclipsing_bodies (list): List of eclipsing body names.
        save_plot (bool, optional): Whether to save the plot. Defaults to True.
    """

    # plot central body
    _u, _v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    _x = body_radius.value * np.cos(_u) * np.sin(_v)
    _y = body_radius.value * np.sin(_u) * np.sin(_v)
    _z = body_radius.value * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap="Blues", zorder=0)

    # Plot each trajectory
    max_val = 0
    for i, rs in enumerate(trajectories):
        partial_eclipse_color = "orange"
        total_eclipse_color = "red"
        label = labels[i]
        default_color = colors[i]
        default_rgb = mcolors.to_rgb(default_color)
        partial_rgb = (
            default_rgb[0] * 0.7,
            default_rgb[1] * 0.7,
            default_rgb[2] * 0.7,
        )  # shadow with 0.4 factor
        total_rgb = (default_rgb[0] * 0.3, default_rgb[1] * 0.3, default_rgb[2] * 0.3)
        # Plot by segments, considering eclipses
        start_index = 0
        for j in range(1, len(rs)):
            current_eclipse = False  # Assume no eclipse at the beginning
            current_partial = False
            current_total = False
            for ecb in eclipsing_bodies:
                eclipse_column = f"Eclipse_status_{ecb}"
                if eclipse_column in df_states.columns:
                    if df_states.loc[j, eclipse_column] == 1:
                        current_eclipse = True
                        current_partial = True
                    elif df_states.loc[j, eclipse_column] == 2:
                        current_eclipse = True
                        current_total = True

            if j > 0:  # Ensure there is a previous point to compare
                previous_eclipse = False
                previous_partial = False
                previous_total = False
                for ecb in eclipsing_bodies:
                    eclipse_column = f"Eclipse_status_{ecb}"
                    if eclipse_column in df_states.columns:
                        if df_states.loc[j - 1, eclipse_column] == 1:
                            previous_eclipse = True
                            previous_partial = True
                        elif df_states.loc[j - 1, eclipse_column] == 2:
                            previous_eclipse = True
                            previous_total = True

                if (
                    current_eclipse != previous_eclipse
                    or current_partial != previous_partial
                    or current_total != previous_total
                ):  # If the eclipse status has changed
                    # Plot the previous segment when the eclipse status changes
                    segment_color = default_color
                    if previous_partial:
                        segment_color = partial_rgb
                    elif previous_total:
                        segment_color = total_rgb

                    ax.plot(
                        rs[start_index:j, 0],
                        rs[start_index:j, 1],
                        rs[start_index:j, 2],
                        color=segment_color,
                        zorder=2,
                        
                    )
                    start_index = j

        # Plot the last segment
        partial_eclipse = False
        total_eclipse = False
        for ecb in eclipsing_bodies:
            eclipse_column = f"Eclipse_status_{ecb}"
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
        if len(rs) > 0:
            ax.plot(
                rs[start_index:, 0],
                rs[start_index:, 1],
                rs[start_index:, 2],
                color=segment_color,
                zorder=2,
                
            )

        ax.plot(
            rs[:, 0], rs[:, 1], rs[:, 2], color=colors[i], label=labels[i], zorder=2
        )
        ax.plot([rs[0, 0]], [rs[0, 1]], [rs[0, 2]], "o", color=colors[i], zorder=2.01)
        current_max = np.max(np.abs(rs))
        max_val = current_max if current_max > max_val else max_val
    # Common configuration
    l = body_radius.value * 2
    x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
    ax.quiver(x, y, z, u, v, w, color="k")

    # Graph limits
    max_val = np.max(np.abs(rs))

    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Labels
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    
    ax.set_title("Orbit 3D", zorder=3)

    plt.legend()
    
    


def setup_coes_plots(n_orbits):
    """Setup for COEs plot"""
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    fig.suptitle("COEs", fontsize=20)
    return fig, axs


def plot_coes(
    axs,
    fig,
    time_arrays,
    coes_arrays,
    colors,
    labels,
    ets_last,
    df_states,
    eclipsing_bodies,
    datetimes,
    maneuver_datetimes,  # Added parameter for maneuver datetimes
    time_unit="seconds",
    save_plot=True,
    title="COEs",
    show_plot=True,
):
    """
    Plot comparison of classical orbital elements with maneuver lines.

    Args:
        axs: Array of matplotlib axes (2x3 grid)
        time_arrays: List of time arrays for each orbit
        coes_arrays: List of COEs arrays for each orbit
        colors: List of plot colors
        labels: List of trajectory labels
        maneuver_datetimes (list): List of datetime objects representing maneuver times.
        time_unit: Time unit for x-axis ('seconds', 'hours' or 'days')
    """

    # Initialize min and max values for each COE to set common limits later
    min_values = np.min(
        np.array([np.min(coes_array[:, :6], axis=0) for coes_array in coes_arrays]),
        axis=0,
    )
    max_values = np.max(
        np.array([np.max(coes_array[:, :6], axis=0) for coes_array in coes_arrays]),
        axis=0,
    )

    for orbit_idx, (times, coes) in enumerate(zip(time_arrays, coes_arrays)):
        default_color = colors[orbit_idx]
        default_rgb = mcolors.to_rgb(default_color)
        partial_rgb = (default_rgb[0] * 0.8, default_rgb[1] * 0.8, default_rgb[2] * 0.8)
        total_rgb = (default_rgb[0] * 0.6, default_rgb[1] * 0.6, default_rgb[2] * 0.6)
        label = labels[orbit_idx]
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_useOffset(False)
        formatter.set_scientific(True)
        formatter._format = "%3e"

        for subplot_row, subplot_col, coe_index, title_str, ylabel in [
            (0, 0, 5, "True Anomaly", "Angle (degrees)"),
            (1, 0, 0, "Semi-Major Axis", "a (km)"),
            (0, 1, 1, "Eccentricity", ""),
            (0, 2, 4, "Argument of Periapsis", "Angle (degrees)"),
            (1, 1, 2, "Inclination", "Angle (degrees)"),
            (1, 2, 3, "RAAN", "Angle (degrees)"),
        ]:
            ax = axs[subplot_row, subplot_col]
            ax.yaxis.set_major_formatter(formatter)
            ax.set_title(title_str)
            ax.grid(True)
            ax.set_ylabel(ylabel)

            start_index = 0
            current_eclipse_status = {}
            previous_eclipse_status = {}
            for ecb in eclipsing_bodies:
                eclipse_column = f"Eclipse_status_{ecb}"
                if eclipse_column in df_states.columns:
                    current_eclipse_status[ecb] = df_states.loc[0, eclipse_column]
                    previous_eclipse_status[ecb] = current_eclipse_status[ecb]
            for j in range(1, len(times)):
                current_eclipse_status = {}
                for ecb in eclipsing_bodies:
                    eclipse_column = f"Eclipse_status_{ecb}"
                    if eclipse_column in df_states.columns:
                        current_eclipse_status[ecb] = df_states.loc[j, eclipse_column]

                if current_eclipse_status != previous_eclipse_status:
                    segment_color = default_color
                    partial_eclipse = any(status == 1 for status in previous_eclipse_status.values())
                    total_eclipse = any(status == 2 for status in previous_eclipse_status.values())

                    if total_eclipse:
                        segment_color = total_rgb
                    elif partial_eclipse:
                        segment_color = partial_rgb

                    ax.plot(
                        datetimes[start_index:j],
                        coes[start_index:j, coe_index],
                        color=segment_color,
                    )
                    start_index = j
                previous_eclipse_status = current_eclipse_status.copy()

            segment_color = default_color
            partial_eclipse = any(status == 1 for status in current_eclipse_status.values())
            total_eclipse = any(status == 2 for status in current_eclipse_status.values())

            if total_eclipse:
                segment_color = total_rgb
            elif partial_eclipse:
                segment_color = partial_rgb
            ax.plot(
                datetimes[start_index:],
                coes[start_index:, coe_index],
                color=segment_color,
            )

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis))
            xlabel = "UTC Time"
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            start_datetime = datetimes[0]
            end_datetime = datetimes[-1]
            ax.set_xlim(start_datetime, end_datetime)

            # Add maneuver lines
            """
            if maneuver_datetimes:
                for maneuver_time in maneuver_datetimes:
                    # Normalize timezones
                    if start_datetime.tzinfo is not None and maneuver_time.tzinfo is not None:
                        maneuver_time = maneuver_time.astimezone(start_datetime.tzinfo)
                    elif start_datetime.tzinfo is None and maneuver_time.tzinfo is not None:
                        maneuver_time = maneuver_time.replace(tzinfo=None)

                    if start_datetime <= maneuver_time <= end_datetime:
                        ax.axvline(maneuver_time, color='red', linestyle='--')
"""
        fig.autofmt_xdate()
        axs[0, 0].set_ylim([min_values[5] - 5, max_values[5] + 5])
        axs[1, 0].set_ylim([min_values[0] - 10, max_values[0] + 10])
        axs[0, 1].set_ylim([min_values[1] - 0.01, max_values[1] + 0.01])
        axs[0, 2].set_ylim([min_values[4] - 5, max_values[4] + 5])
        axs[1, 1].set_ylim([min_values[2] - 1, max_values[2] + 1])
        axs[1, 2].set_ylim([min_values[3] - 5, max_values[3] + 5])

        plt.tight_layout()
        for ax in axs.flat:
            ax.grid(True)
            ax.set_xlabel(xlabel)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.9))
        #if save_plot:
         #   plt.savefig(os.path.join("output", f"{title}" + ".png"), dpi=300, bbox_inches="tight")
        
            


def setup_vector_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    return fig, axes


# Función para graficar los vectores en un plano 2D
def plot_vectors(vector_list, ax, i, color, vector_name, title, margin, save_plot=True):
    """
    Plots a list of vectors on a 2D axis.

    Args:
    vector_list (list): List of vectors to plot.
    ax (matplotlib.axes.Axes): Axis object to plot on.
    i (int): Index of the vector list (for legend purposes).
    color (str): Color of the vectors.
    vector_name (str): Name of the vector set (for legend purposes).
    title (str): Title of the plot.
    margin (float): Margin for the plot limits.
    save_plot (bool, optional): Whether to save the plot. Defaults to True.
    """

    vector_array = np.array(vector_list)
    # Plot the vectors
    x = vector_array[:, 0]  # Extract all x coordinates
    y = vector_array[:, 1]  # Extract all y coordinates
    ax.scatter(
        x[0], y[0], color="green", s=50, label="First point" if i == 0 else None
    )  # First point. Legend if it's the first orbit
    ax.scatter(
        x[-1], y[-1], color="red", s=50, label="Last point" if i == 0 else None
    )  # Last point. Legend if it's the first orbit
    ax.plot(
        x, y, color=color, linewidth=0.5, label=vector_name
    )  # Single scatter for all points

    # Customize the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid()
    ax.legend()

    min_value_x = np.min(x)
    max_value_x = np.max(x)
    min_value_y = np.min(y)
    max_value_y = np.max(y)
    ax.set_aspect("equal")  # To make the axes have the same scale
    ax.set_xlim(
        [min_value_x - margin, max_value_x + margin]
    )  # Adjust limits if necessary
    ax.set_ylim(
        [min_value_y - margin, max_value_y + margin]
    )  # Adjust limits if necessary


def plot_i_and_e_vectors(
    fig,
    axes,
    i_vec_all,
    e_vec_all,
    orbit_params,
    save_plot=True,
    show_plot=True,
    title="e-i_vectors",
):
    """Plots the i and e vectors in separate subplots."""

    # Initial values for the limits (calculated within the function)
    min_xi, min_xe = float("inf"), float("inf")
    max_xi, max_xe = float("-inf"), float("-inf")
    min_yi, min_ye = float("inf"), float("inf")
    max_yi, max_ye = float("-inf"), float("-inf")

    # Plot the i vectors
    for i, i_vec_orbit_list in enumerate(i_vec_all):
        i_vec_orbit_array = np.array(i_vec_orbit_list)
        plot_vectors(
            i_vec_orbit_array,
            axes[1],
            i,
            orbit_params["colors"][i],
            f"i_vec mission {i}",
            title=f"Inclination vector",
            margin=0.01,
        )
        # Update the limits
        min_xi = min(min_xi, np.min(i_vec_orbit_array[:, 0]))
        max_xi = max(max_xi, np.max(i_vec_orbit_array[:, 0]))
        min_yi = min(min_yi, np.min(i_vec_orbit_array[:, 1]))
        max_yi = max(max_yi, np.max(i_vec_orbit_array[:, 1]))

    # Plot the e vectors
    for i, e_vec_orbit_list in enumerate(e_vec_all):
        e_vec_orbit_array = np.array(e_vec_orbit_list)
        plot_vectors(
            e_vec_orbit_array,
            axes[0],
            i,
            orbit_params["colors"][i],
            f"e_vec mission {i+1}",
            title=f"Excentricity vector",
            margin=0.001,
        )
        # Limits:
        min_xe = min(min_xe, np.min(e_vec_orbit_array[:, 0]))
        max_xe = max(max_xe, np.max(e_vec_orbit_array[:, 0]))
        min_ye = min(min_ye, np.min(e_vec_orbit_array[:, 1]))
        max_ye = max(max_ye, np.max(e_vec_orbit_array[:, 1]))

    # Set the plot limits
    margin_i = 0.001
    margin_e = 0.0001
    axes[0].set_xlim([min_xe - margin_e, max_xe + margin_e])
    axes[0].set_ylim([min_ye - margin_e, max_ye + margin_e])
    axes[1].set_xlim([min_xi - margin_i, max_xi + margin_i])
    axes[1].set_ylim([min_yi - margin_i, max_yi + margin_i])

    plt.tight_layout()


    
        

def vel_lon(lon, t_eval):
    
    lon_vel = np.diff(lon, axis=1) / np.diff(t_eval)
    
    return lon_vel

def setup_lon_plot():
    """Sets up the figure and axes for the longitude plots."""
    fig_lon, axes_lon = plt.subplots(2, 1, figsize=(12, 12))  # Increased figure height
    return fig_lon, axes_lon
def plot_lon(lon, lon_vel, t_eval, datetimes, args=None):
    """
    Plots the longitude and its velocity variation in two subplots.

    Args:
        lon (numpy.ndarray): Array of longitude values.
        lon_vel (numpy.ndarray): Array of longitude velocity variation values.
        t_eval (numpy.ndarray): Array of evaluation times.
        args (dict, optional): Dictionary containing plot arguments. Defaults to None.
    """

    _args = {
        "markersize": 1,
        "labels": [""] * len(lon),
        "colors": ["c", "r", "b", "g", "w", "y"],
        "grid": True,
        "show": True,
        "filename": True,
        "dpi": 300,
        "legend": True,
    }

    if args:
        _args.update(args)

    fig_lon, axes_lon = setup_lon_plot()

    for i in range(lon.shape[0]):
        # Plot 1: Longitude speed variation vs. Longitude
        axes_lon[0].plot(lon[i, 1:], lon_vel[i, :], label=f'Mission {i+1}', color=_args["colors"][i % len(_args["colors"])])
        axes_lon[0].set_xlabel('Longitude (degrees)')
        axes_lon[0].set_ylabel('Drift rate (degrees/s)')
        axes_lon[0].set_title('Drift rate vs. Longitude')

        # Plot 2: Longitude vs. Time
        axes_lon[1].plot(datetimes, lon[i, :], label=f'Mission {i+1}', color=_args["colors"][i % len(_args["colors"])])
        axes_lon[1].set_xlabel('Time')
        axes_lon[1].set_ylabel('Longitude (degrees)')
        axes_lon[1].set_title('Longitude vs. Time')

    # Common formatting
    for ax in axes_lon:
        if _args["legend"]:
            ax.legend()
        if _args["grid"]:
            ax.grid(linestyle="dotted")

    if _args["filename"]:
        plt.savefig(os.path.join("output", "Longitude_plots.png"), dpi=300, bbox_inches="tight")

    
        
def plot_groundtracks(coords, args, coastlines_coordinates_file):
    _args = {
        "figsize": (18, 9),
        "markersize": 1,
        "labels": [""] * len(coords),
        "colors": ["c", "r", "b", "g", "w", "y"],
        "grid": True,
        "title": "Groundtracks",
        "show": True,
        "filename": True,
        "dpi": 300,
        "legend": True,
        "surface_image": False,
        "surface_body": "earth",
        "plot_coastlines": True,
    }
    for key in args.keys():
        _args[key] = args[key]

    plt.figure(figsize=_args["figsize"])

    
    for i in range(coords.shape[0]):  #
        lon = np.rad2deg(coords[i, :, 0])  # lon in degrees
        lat = np.rad2deg(coords[i, :, 1])  # lat in degrees

    # Different color por each propagation
    color = _args["colors"][i % len(_args["colors"])]  # Use cyclic colors

    # Graph the points
    plt.plot(
        lon,
        lat,
        "o",
        color=color,
        markersize=_args["markersize"],
        label=f"Mission {i+1}",
    )
    if _args["plot_coastlines"]:
        coast_coords = np.genfromtxt(coastlines_coordinates_file, delimiter=",")

        plt.plot(coast_coords[:, 0], coast_coords[:, 1], "mo", markersize=0.3)


    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(range(-180, 200, 20))
    plt.yticks(range(-90, 100, 10))
    plt.xlabel(r"Longitude (degrees $^\circ$)")
    plt.ylabel(r"Latitude (degrees $^\circ$)")
    plt.tight_layout()

    if _args["legend"]:
        plt.legend()

    if _args["grid"]:
        plt.grid(linestyle="dotted")

    
        


