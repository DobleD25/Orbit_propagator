# -*- coding: utf-8 -*-
"""
@author: ddiaz.beca
This script performs orbit propagation using numerical integration methods,
handles perturbations, calculates eclipse statuses, and generates plots.

"""
import os

# importing sys
import sys

# 3rd party libraries
import numpy as np
import time
import spiceypy as spice


# Directory route:
current_dir = os.path.dirname(os.path.abspath(__file__))
# build routes to script folders
Transformation_path = os.path.join(current_dir, "src", "Transformations")
Tools_path = os.path.join(current_dir, "src", "Tools")
Perturbation_path = os.path.join(current_dir, "src", "Perturbations")
Math_path = os.path.join(current_dir,"src", "Math")
Data_path = os.path.join(current_dir, "Data")
# adding folders to the system path
sys.path.insert(0, Transformation_path)
sys.path.insert(0, Tools_path)
sys.path.insert(0, Perturbation_path)
sys.path.insert(0, Math_path)
sys.path.insert(0, Data_path)

# Own libraries
import coord_conversion as cc
import plot_orbit as po
import n_methods as nm
import geopotential_model as gm
import read as rd
import init
import tools as tl


# Integration of the ODE
def solve_orbit(
    method,
    steps,
    ets,
    states,
    orbit_params,
    perturbation_params,
    body_params,
    man_epoch_chem_list,
    man_epoch_elec_list,
    coeffs,
    max_order,
    spacecraft_params,
):
    """
    Integrates the orbit using the specified method.

    Input:
        method: integration method (e.g., "1" for RK4, "2" for Adams-Bashforth-Moulton)
        steps: number of steps
        ets: array of time steps (relative to epoch)
        states: initial state vector (position and velocity)
        orbit_params: dictionary with orbit parameters (e.g., dt, epoch)
        perturbation_params: dictionary with perturbation parameters (e.g., perturbations, coef_pot)
        body_params: dictionary with body parameters (e.g., radius, name, ecb)
        man_epoch_chem_list: list of chemical maneuver epochs
        man_epoch_elec_list: list of electrical maneuver epochs
        coeffs: potential model coefficients
        max_order: maximum order of the potential model
        spacecraft_params: dictionary with spacecraft parameters

    Output:
        states: array of state vectors at each time step
        ets: array of time steps
        eclipse_statuses: array of eclipse statuses at each time step and for each eclipsing body
    """
    f_prev = [None] * 4  # To store previous function evaluations
    eclipse_statuses = []  # List to store eclipse status for each step
    dt = orbit_params["dt"]
    epoch = orbit_params["epoch"]
    pert = perturbation_params["perturbations"]
    coef_pot = perturbation_params["coef_pot"]
    cb_radius = body_params["radius"]
    name_cb = body_params["name"]
    # mu_cb=getattr(planets, name_cb.lower())['mu']
    mu_cb = 398600.4418  # mu_hardcodeado para compararlo con GMAT
    ecb_list = body_params["ecb"]

    epoch_et = spice.str2et(epoch)  # Tiempo absoluto del epoch inicial
    t_eval = [0.0] * steps
    num_maniobras_electricas = len(man_epoch_elec_list)
    deltaV_electric_maneuvers = np.zeros((num_maniobras_electricas, 3))
    # Eclipsis status:
    # 0: No eclipse
    # 1: Partial eclipse
    # 2: Total eclipse
    eclipse_statuses = np.zeros(
        (steps, len(ecb_list)), dtype=int
    )  # Array to store eclipse status for each step and eclipsing body
    if method == "1":

        # Call RK4 method
        states, t_eval, eclipse_statuses = nm.call_rk4(
            epoch_et,
            ets,
            steps,
            states,
            pert,
            coef_pot,
            cb_radius,
            mu_cb,
            ecb_list,
            body_params,
            man_epoch_chem_list,
            man_epoch_elec_list,
            deltaV_electric_maneuvers,
            orbit_params,
            coeffs,
            max_order,
            perturbation_params,
            spacecraft_params,
            dt,
            eclipse_statuses,
        )
        """
    elif method == "2":

        # Call Adams Bashford-Moulton Method
        states, t_eval, eclipse_statuses = nm.call_adamsBM(
            epoch_et,
            ets,
            steps,
            states,
            pert,
            coef_pot,
            cb_radius,
            mu_cb,
            ecb_list,
            body_params,
            man_epoch_chem_list,
            man_epoch_elec_list,
            deltaV_electric_maneuvers,
            orbit_params,
            coeffs,
            max_order,
            perturbation_params,
            spacecraft_params,
            dt,
            eclipse_statuses,
            f_prev,
        )
        """
    elif method == "2":

        # Call RK56 method
        states, t_eval, eclipse_statuses = nm.call_rk56(
            epoch_et,
            ets,
            steps,
            states,
            pert,
            coef_pot,
            cb_radius,
            mu_cb,
            ecb_list,
            body_params,
            man_epoch_chem_list,
            man_epoch_elec_list,
            deltaV_electric_maneuvers,
            orbit_params,
            coeffs,
            max_order,
            perturbation_params,
            spacecraft_params,
            dt,
            eclipse_statuses,
        )
        
    else:
        print("Incorrect value")

    return states, t_eval, eclipse_statuses, ecb_list


if __name__ == "__main__":

    # Read Input
    body_params, orbit_params, perturbation_params, spacecraft_params = rd.read_json(
        "Input.json"
    )
    # Read maneuver data
    orbit_params = rd.read_maneuvers(orbit_params)
    # Read EGM96 model coefficients
    coeffs, max_order = gm.egm96_model_bool(perturbation_params)
    # Calculate mu in km
    mu = tl.mu_km(tl.G_km(), body_params["mass"])
    # Initialization of arrays and lists used in the propagation:
    (
        trajectories,
        states_kepl_all,
        sph_bodyfixed_all,
        e_vec,
        i_vec,
        i_vec_all,
        e_vec_all,
        labels,
    ) = init.global_prop_init(orbit_params)

    for idx, initial_state in enumerate(orbit_params["initial_states"]):
        # Initialization for orbit number idx:
        (
            steps,
            states,
            sph_bodyfixed,
            ets,
            method,
            start_time,
            man_epoch_chem_list,
            man_epoch_elec_list,
            i_vec_orbit,
            e_vec_orbit,
        ) = init.orbit_prop_init(idx, initial_state, orbit_params)

        # Propagation of the orbit
        states, t_eval, eclipse_statuses, ecb_list = solve_orbit(
            method,
            steps,
            ets,
            states,
            orbit_params,
            perturbation_params,
            body_params,
            man_epoch_chem_list,
            man_epoch_elec_list,
            coeffs,
            max_order,
            spacecraft_params,
        )
        # Calculate and print execution time.
        execution_time = tl.fun_execution_time(start_time, time)

        # Save trajectories:
        trajectories.append(states[:, :3])

        # Conversion to Keplerian elements
        states_kepl = np.apply_along_axis(cc.cart_2_kep, 1, states, mu)
        states_kepl_all.append(states_kepl)
        # Conversion to spherical coordinates in body-fixed frame:
        for i in range(len(states)):
            state = states[i]
            epoch = t_eval[i]
            spherical_coords = cc.proyect_lat_lon(state, epoch)
            sph_bodyfixed.append(spherical_coords)
        sph_bodyfixed_all.append(sph_bodyfixed)
        sph_bodyfixed_all = np.array(sph_bodyfixed_all)  # Convert to np array
        for state_kepl in states_kepl:  # Iterate over rows of the coordinates array
            cc.fun_i_vec(state_kepl, i_vec_orbit)
            cc.fun_e_vec(state_kepl, e_vec_orbit)
        i_vec_all.append(i_vec_orbit)
        e_vec_all.append(e_vec_orbit)

        # Save CSV file with ephemerides in cartesians and eclipse status, and get states dataframes and datetimes
        df_states, datetimes = tl.save_csv(
            t_eval, states, eclipse_statuses, body_params, idx
        )
        # Save Save CSV file with ephemerides in cartesians and eclipse status
        tl.save_csv_kepler(t_eval, states_kepl, eclipse_statuses, body_params, idx)

#List of maneuver datetimes,for plotting
maneuver_datetimes=tl.get_maneuver_dates(man_epoch_chem_list, man_epoch_elec_list)
# Call plotting functions.
po.call_plots(
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
    maneuver_datetimes,
    ecb_list
)


spice.kclear()
