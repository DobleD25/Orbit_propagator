# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:44:01 2025

@author: ddiaz.beca
"""

import spiceypy as spice


def n_body(central_body, n_body, time):
    """
    Retrieves the state vector of a third body relative to a central body at a given time with SPICE

    Args:
        central_body (str): Name of the central body.
        n_body (str): Name of the third body.
        time (float): Ephemeris time (ET).

    Returns:
        numpy.ndarray: State vector (position and velocity) of the n body relative to the central body.
    """
    # et=spice.str2et(time)
    state, lt = spice.spkezr(
        n_body.upper(), time, "J2000", "NONE", central_body.upper()
    )
    return state
