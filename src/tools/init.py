# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:24:52 2025

@author: ddiaz.beca
"""
import numpy as np
import time


def global_prop_init(orbit_params):
    """
    Initializes global variables for orbit propagation.

    Args:
        orbit_params (dict): Dictionary containing orbit parameters.

    Returns:
        tuple: A tuple containing initialized lists and variables:
            - trajectories (list): List to store trajectory data.
            - states_kepl_all (list): List to store Keplerian state vectors.
            - e_vec (list): Initial eccentricity vector.
            - i_vec (list): Initial inclination vector.
            - i_vec_all (list): List to store inclination vectors for all orbits.
            - e_vec_all (list): List to store eccentricity vectors for all orbits.
            - labels (list): Labels for each orbit.
    """
    # init trajectories:
    trajectories = []
    states_kepl_all = []

    # spherical coordinates body-fixed:
    sph_bodyfixed = []
    # e, i vectors:
    e_vec = [0, 0]
    i_vec = [0, 0]
    i_vec_all = []
    e_vec_all = []
    # Labels of the different orbits
    labels = [f"Mission {i+1}" for i in range(len(orbit_params["initial_states"]))]
    return (
        trajectories,
        states_kepl_all,
        sph_bodyfixed,
        e_vec,
        i_vec,
        i_vec_all,
        e_vec_all,
        labels,
    )


def orbit_prop_init(idx, initial_state, orbit_params):
    """
    Initializes variables for a specific orbit propagation.

    Args:
        idx (int): Index of the orbit.
        initial_state (numpy.ndarray): Initial state vector for the orbit.
        orbit_params (dict): Dictionary containing orbit parameters.

    Returns:
        tuple: A tuple containing initialized variables for the orbit:
            - steps (int): Number of integration steps.
            - states (numpy.ndarray): Array to store state vectors.
            - ets (numpy.ndarray): Array of time steps.
            - method (str): Selected integration method.
            - start_time (float): Start time of the propagation.
            - man_epoch_chem_list (list): List of chemical maneuver epochs.
            - man_epoch_elec_list (list): List of electrical maneuver epochs.
            - i_vec_orbit (list): List to store inclination vectors for the orbit.
            - e_vec_orbit (list): List to store eccentricity vectors for the orbit.
    """
    steps = int(orbit_params["tspan"] / orbit_params["dt"])
    states = np.zeros(
        (steps, 6), dtype=np.float64
    )  # Inicializamos states con formato float64
    states[0] = initial_state

    r_sph_bodyfixed = []
    ets = np.linspace(0, (steps - 1) * orbit_params["dt"], steps)  # Time steps

    # Propagation
    method = input(
        f"""Select number of the resolution method for orbit {idx+1}:
                   1- Runge-Kutta 4th order 
                   2- Runge-Kutta 5(6)th order
                   \n"""
    )
    start_time = time.time()  # Execution time start
    # Definición de maniobras:
    man_epoch_chem_list = orbit_params["man_epoch_chem_lists"][idx]
    man_epoch_elec_list = orbit_params["man_epoch_elec_lists"][idx]

    # Inicialization of i_vec y e_vec para la orbita:
    i_vec_orbit = []
    e_vec_orbit = []
    return (
        steps,
        states,
        r_sph_bodyfixed,
        ets,
        method,
        start_time,
        man_epoch_chem_list,
        man_epoch_elec_list,
        i_vec_orbit,
        e_vec_orbit,
    )
