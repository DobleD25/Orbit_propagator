# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:14:44 2025

@author: ddiaz.beca
"""

from astropy.constants import L_sun, R_sun, R_earth, c

c_light = c
import spice_tool as st
import numpy as np
import coord_conversion as cc
from astropy import units as u


def apparent_r(ocb, body_params, epoch, r):
    """
    Calculates the apparent radii and related coefficients for occultation.

    Args:
        ocb (str): Name of the occulting body.
        body_params (dict): Parameters of the central body (from input.json).
        epoch (float): Time in ephemeris seconds.
        r (numpy.ndarray): Position vector of the spacecraft.

    Returns:
        tuple: Apparent radii and related coefficients (a, b, c_).
            a: Apparent radius of the Sun.
            b: Apparent radius of the occulting body.
            c_: Apparent distance between the center of the occulting body and the center of the Sun.
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
    r_sat2sun = r_cb2sun - r
    a = np.arcsin(R_sun.value / 1000 / (np.linalg.norm(r_sat2sun)))
    b = np.arcsin(R_earth.value / 1000 / (np.linalg.norm(r_sat2ocb)))
    cos_c = np.dot(-r_sat2ocb, r_sat2sun) / (
        np.linalg.norm(r) * np.linalg.norm(r_sat2sun)
    )
    c_ = np.arccos(np.clip(cos_c, -1.0, 1.0))  # Clip to [-1, 1]

    return a, b, c_


def F_srp(P, Cr, Area_m2, r_unit_sun):
    """
    Calculates the force due to solar radiation pressure.

    Args:
        P (astropy.units.quantity.Quantity): Solar radiation pressure.
        Cr (float): Radiation pressure coefficient.
        Area_m2 (astropy.units.quantity.Quantity): Spacecraft area in m^2.
        r_unit_sun (astropy.units.quantity.Quantity): Unit vector pointing from the spacecraft to the Sun.

    Returns:
        numpy.ndarray: Force vector in Newtons.
    """
    F = -P * Cr * Area_m2 * r_unit_sun
    return F


def SRP_cannonball(body_params, sc_params, epoch, state, f):
    """
    Calculates the acceleration due to solar radiation pressure using the cannonball model.

    Args:
        body_params (dict): Parameters of the central body (from input.json).
        sc_params (dict): Parameters of the spacecraft.
        epoch (float): Time in ephemeris seconds.
        state (numpy.ndarray): State vector of the spacecraft.
        f (float): Shadow function.

    Returns:
        numpy.ndarray: Acceleration vector due to SRP in km/s^2.
    """
    r_cb2sun = st.n_body(body_params["name"], "Sun", epoch)
    r_cb2sun = r_cb2sun[:3]  # position vector occulting body to sun
    r = state[:3]  # position vector s/c
    r_sat2sun = r_cb2sun - r  # position vector satellite to sun (km)
    r_sat2sun_meters = r_sat2sun * 1000 * u.m  # to S.I units (m)
    r_sat2sun_mnorm = np.linalg.norm(r_sat2sun_meters)
    # Solar pressure, N/m2
    P = L_sun / (4 * np.pi * c * r_sat2sun_mnorm**2)  # r_sat2sun from km to meter
    # a, b, c_ = apparent_r(ocb, body_params, epoch, r)

    Area_m2 = sc_params["area"]  # Área en m^2
    Mass_kg = sc_params["mass"]  # Masa en kg
    Cr = sc_params["Cr"]  # Reflectivity coefficient
    r_hat_sun = r_sat2sun_meters / r_sat2sun_mnorm

    # Initialization of F:
    F = np.array([0.0, 0.0, 0.0])
    # Calculation of F with the shadow function
    F = f * F_srp(P, Cr, Area_m2, r_hat_sun)

    a_srp_m = np.array([0.0, 0.0, 0.0])
    a_srp_m = F / (Mass_kg)  # acceleration of the perturbation (m/s2)
    a_srp_km = np.array([0.0, 0.0, 0.0])
    a_srp_km = a_srp_m / 1000  # m/s2 to km/s


def SRP_realistic(body_params, sc_params, epoch, state, f):
    """
    Calculates the acceleration due to solar radiation pressure taking in count the declination of the Sun.

    Args:
        body_params (dict): Parameters of the central body (from input.json).
        sc_params (dict): Parameters of the spacecraft.
        epoch (float): Time in ephemeris seconds.
        state (numpy.ndarray): State vector of the spacecraft.
        f (float): Shadow function.

    Returns:
        numpy.ndarray: Acceleration vector due to SRP in km/s^2.
    """
    r_cb2sun = st.n_body(body_params["name"], "Sun", epoch)
    r_cb2sun = r_cb2sun[:3]  # position vector occulting body to sun
    r = state[:3]  # position vector s/c
    r_sat2sun = r_cb2sun - r  # position vector satellite to sun (km)
    r_sat2sun_meters = r_sat2sun * 1000 * u.m  # to S.I units (m)
    r_sat2sun_mnorm = np.linalg.norm(r_sat2sun_meters)
    #Declination of the sun in radians
    _, _, decl= cc.cartesian_to_spherical(r_sat2sun)
    # Solar pressure, N/m2
    P = L_sun / (4 * np.pi * c * r_sat2sun_mnorm**2)  # r_sat2sun from km to meter
    

    Area_m2 = sc_params["area"]
    Mass_kg = sc_params["mass"]
    Cr = sc_params["Cr"]  # Absortion coeficient
    r_hat_sun = r_sat2sun_meters / r_sat2sun_mnorm

    # Initialization of F:
    F = np.array([0.0, 0.0, 0.0])
    # Calculation of F with the shadow function
    F = f * F_srp(P, Cr, Area_m2, r_hat_sun)*np.cos(decl)

    a_srp_m = np.array([0.0, 0.0, 0.0])
    a_srp_m = F / (Mass_kg)  # acceleration of the perturbation (m/s2)
    a_srp_km = np.array([0.0, 0.0, 0.0])
    a_srp_km = a_srp_m / 1000  # Convert to km/s^2
    return a_srp_km.value