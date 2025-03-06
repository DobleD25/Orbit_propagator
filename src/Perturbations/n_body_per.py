# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:26:59 2025

@author: ddiaz.beca
"""
import numpy as np
import spice_tool as st
import planetary_data as pd


def n_body_a(perturbation_params, body_params, epoch, r):
    """
    Calculates the acceleration due to N-body perturbations.

    Args:
        perturbation_params (dict): Dictionary containing perturbation parameters, including the list of N-body perturbers.
        body_params (dict): Dictionary containing parameters of the central body.
        epoch (float): Ephemeris time (ET) epoch.
        r (numpy.ndarray): Position vector of the spacecraft in the central body's frame.

    Returns:
        numpy.ndarray: Acceleration vector due to N-body perturbations.
    """
    body_list = perturbation_params["N-body"][0]["list"]

    r_nbodies = []
    a_nbody = np.array([0.0, 0.0, 0.0])
    for nbody in body_list:

        # vector pointing from central body to n-body
        state_cb2nb = st.n_body(body_params["name"], nbody, epoch)
        r_cb2nb = state_cb2nb[:3]

        # r_nbodies.append(r_cb2nb)
        try:
            mu_nbody = getattr(pd, nbody.lower())[
                "mu"
            ]  # Use getattr for dynamic access

        except AttributeError:
            raise ValueError(f"Unknown n-body: {nbody}")
        # Vector pointing from satellite to n-body:
        r_sat2nb = r_cb2nb - r
        # Acceleration of the n-body in the sat.
        a_nbody += mu_nbody * (
            r_sat2nb / np.linalg.norm(r_sat2nb) ** 3
            - r_cb2nb / np.linalg.norm(r_cb2nb) ** 3
        )
    return a_nbody
