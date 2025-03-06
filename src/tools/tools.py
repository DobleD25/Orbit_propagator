# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:30:38 2025

@author: ddiaz.beca
"""
import astropy.constants as const
import astropy.units as u
import spiceypy as spice
import pandas as pd
import os


def G_km():
    """
    Calculates the gravitational constant G in km^3 / (kg * s^2).

    Returns:
        astropy.units.quantity.Quantity: Gravitational constant in specified units.
    """
    G_km = const.G.to(u.km**3 / (u.kg * u.s**2))
    return G_km


def mu_km(G_km, mass):
    """
    Calculates the standard gravitational parameter mu in km^3 / s^2.

    Args:
        G_km (astropy.units.quantity.Quantity): Gravitational constant in km^3 / (kg * s^2).
        mass (astropy.units.quantity.Quantity): Mass of the central body in kg.

    Returns:
        float: Standard gravitational parameter mu in km^3 / s^2.
    """
    mu = (G_km * mass).value
    return mu


def fun_execution_time(start_time, time):
    """
    Calculates and prints the execution time of a function.

    Args:
        start_time (float): Start time of the function (e.g., time.time()).
        time (module): The time module.

    Returns:
        float: Execution time in seconds.
    """
    end_time = time.time()
    # Execution time:
    execution_time = end_time - start_time
    print(
        f"Execution finished in {execution_time:.3f}s. Generating state vectors files and plots."
    )
    return execution_time


def save_csv(t_eval, states, eclipse_statuses, body_params, idx):
    """
    Saves the state vectors and eclipse statuses to a CSV file.

    Args:
        t_eval (numpy.ndarray): Array of evaluation times (ET).
        states (numpy.ndarray): Array of state vectors.
        eclipse_statuses (numpy.ndarray): Array of eclipse statuses.
        body_params (dict): Dictionary containing body parameters.
        idx (int): Index of the orbit.

    Returns:
        tuple: DataFrame containing state vectors and datetimes.
    """
    datetimes = [
        spice.et2datetime(et) for et in t_eval
    ]  # absolute SPICE Time to datetime
    # Save files
    df_states = pd.DataFrame(states, columns=["X", "Y", "Z", "Vx", "Vy", "Vz"])
    df_states.insert(0, "Time", t_eval)
    df_states.insert(1, "Datetime", datetimes)
    for i, ecb in enumerate(body_params["ecb"]):
        df_states.insert(
            len(df_states.columns), f"Eclipse_status_{ecb}", eclipse_statuses[:, i]
        )  # Adds a column for each eclipsing body.
    df_states.to_csv(
        os.path.join(os.getcwd(), "output", f"State_vectors_orbit_{idx+1}.csv"),
        index_label="Step",
    )
    return df_states, datetimes


def save_csv_kepler(t_eval, kep_states, eclipse_statuses, body_params, idx):
    """
    Saves the keplerian state vectors and eclipse statuses to a CSV file.

    Args:
        t_eval (numpy.ndarray): Array of evaluation times (ET).
        states (numpy.ndarray): Array of keplerian state vectors.
        eclipse_statuses (numpy.ndarray): Array of eclipse statuses.
        body_params (dict): Dictionary containing body parameters.
        idx (int): Index of the orbit.

    """
    datetimes = [
        spice.et2datetime(et) for et in t_eval
    ]  # absolute SPICE Time to datetime
    # Save files (a, e, i, RAAN, aop, nu)
    df_states = pd.DataFrame(kep_states, columns=["a", "e", "i", "RAAN", "aop", "ta"])
    df_states.insert(0, "Time", t_eval)
    df_states.insert(1, "Datetime", datetimes)
    for i, ecb in enumerate(body_params["ecb"]):
        df_states.insert(
            len(df_states.columns), f"Eclipse_status_{ecb}", eclipse_statuses[:, i]
        )  # Adds a column for each eclipsing body.
    df_states.to_csv(
        os.path.join(os.getcwd(), "output", f"Kepler_State_vectors_orbit_{idx+1}.csv"),
        index_label="Step",
    )
