# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:26:19 2025

@author: ddiaz.beca
"""

from astropy.constants import R_sun
import spice_tool as st
import numpy as np
import planetary_data as pd


def apparent_r(ocb, body_params, epoch, r):
    """
    Calculates apparent radii and related coefficients for eclipse calculations.

    Parameters:
    ----------
    ocb : str
        Name of the occulting body (e.g., "Moon").
    body_params : dict
        Parameters of the central body (from input.json).
    epoch : float
        Time in ephemeris time.
    r : numpy.ndarray
        Position vector of the spacecraft.

    Returns:
    -------
    tuple:
        A tuple containing:
        - a (float): Apparent radius of the Sun.
        - b (float): Apparent radius of the occulting body.
        - c_ (float): Apparent distance between the center of the occulting body and the Sun.
    """
    r_ocb = st.n_body(
        body_params["name"], ocb, epoch
    )  # distance between the ocultting body and the Earth (Earth centered system)
    r_ocb = r_ocb[:3]  # save only the position vector
    r_sat2ocb = r_ocb - r  # distance between the satellite and the occulting body
    r_cb2sun = st.n_body(
        body_params["name"], "Sun", epoch
    )  # distance between the central body and the Sun.
    r_cb2sun = r_cb2sun[:3]
    r_sat2sun = r_cb2sun - r  # distance between the sat and the sun
    a = np.arcsin(R_sun.value / 1000 / (np.linalg.norm(r_sat2sun)))

    ecb_radius = getattr(pd, ocb.lower())["radius"]
    b = np.arcsin(ecb_radius / (np.linalg.norm(r_sat2ocb)))

    cos_c = np.dot(-r_sat2ocb, r_sat2sun) / (
        np.linalg.norm(r_sat2ocb) * np.linalg.norm(r_sat2sun)
    )
    c_ = np.arccos(np.clip(cos_c, -1.0, 1.0))  # Clip to [-1, 1]

    return a, b, c_


def eclipse(ocb, body_params, epoch, state):
    """
    Determines the eclipse status and calculates the shadow function (f).

    Parameters:
    ----------
    ocb : str
        Name of the occulting body.
    body_params : dict
        Parameters of the central body.
    epoch : float
        Time in ephemeris time.
    state : numpy.ndarray
        State vector of the spacecraft.

    Returns:
    -------
    tuple:
        A tuple containing:
        - eclipse_status (int): 0 (no eclipse), 1 (partial eclipse), 2 (total eclipse).
        - f(float): Shadow function.
    """

    r = state[:3]  # position vector s/c

    a, b, c_ = apparent_r(ocb, body_params, epoch, r)  # Apparent radius

    eclipse_status = 0
    if (a + b) <= c_:  # No ocultation
        f = 1.0
        eclipse_status = 0

    elif np.linalg.norm(a - b) < c_ < a + b:  # partial ocultation
        x = (c_**2 + a**2 - b**2) / (2 * c_)
        y = np.sqrt(a**2 - x**2)
        A = a**2 * np.arccos(x / a) + b**2 * np.arccos((c_ - x) / b) - c_ * y
        f = 1 - A / (np.pi * a**2)
        eclipse_status = 1
    elif c_ < a - b:  # partial ocultation
        f = 1 - (b / a) ** 2
        eclipse_status = 1
    elif c_ < b - a:  # Full ocultation
        f = 0.0
        eclipse_status = 2

    else:
        print("Error in the geometry of the eclipse")
        f = 1.0

    return eclipse_status, f


def calculate_nu(state, ecb_list, body_params, epoch_et):
    """
    Calculates the shadow function (f) for a given state and list of eclipsing bodies.

    Parameters:
    ----------
    state : numpy.ndarray
        State vector of the spacecraft.
    ecb_list : list
        List of eclipsing bodies.
    body_params : dict
        Parameters of the central body.
    epoch_et : float
        Time in ephemeris time.

    Returns:
    -------
    float:
        The calculated shadow function f.
    """
    f = None  # Inicializa nu

    for i, ecb in enumerate(ecb_list):
        _, f = eclipse(
            ecb, body_params, epoch_et, state
        )  # Calculate ffor the current time and state
    return f
