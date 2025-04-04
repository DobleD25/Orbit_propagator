# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:27:56 2025

@author: ddiaz.beca
"""
import numpy as np
import astropy.units as u


def j2(state, J2, mu, body_radius):
    """
    Calculates the acceleration due to the J2 zonal harmonic.

    Args:
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        J2 (float): J2 zonal harmonic coefficient.
        mu (float): Standard gravitational parameter.
        body_radius (float): Radius of the central body.

    Returns:
        numpy.ndarray: Acceleration vector due to J2.
    """

    r = state[:3]
    norm_r = np.linalg.norm(r)
    z2 = r[2] ** 2
    r2 = norm_r**2

    tx = r[0] / norm_r * (5 * z2 / r2 - 1)
    ty = r[1] / norm_r * (5 * z2 / r2 - 1)
    tz = r[2] / norm_r * (5 * z2 / r2 - 3)

    a_J2 = 1.5 * J2 * mu * body_radius**2 / norm_r**4 * np.array([tx, ty, tz])

    return a_J2


def j3(state, J3, mu, body_radius):
    """
    Calculates the acceleration due to the J3 zonal harmonic.

    Args:
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        J3 (float): J3 zonal harmonic coefficient.
        mu (float): Standard gravitational parameter.
        body_radius (float): Radius of the central body.

    Returns:
        numpy.ndarray: Acceleration vector due to J3.
    """
    r = state[:3]
    norm_r = np.linalg.norm(r)
    z2 = r[2] ** 2
    z3 = r[2] ** 3
    r2 = norm_r**2

    tx = r[0] / norm_r * (3 * r[2] - 7 * z3 / r2)
    ty = r[1] / norm_r * (3 * r[2] - 7 * z3 / r2)
    tz = 6 * z2 - 7 * z2**2 / r2 - 3 * r2 / 5

    a_J3 = -(5 * J3 * mu * body_radius**3) / (2 * norm_r**7) * np.array([tx, ty, tz])

    return a_J3


def C22(state, C22, mu, body_radius):
    """
    Calculates the acceleration due to the C22 sectorial harmonic.

    Args:
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        C22 (float): C22 sectorial harmonic coefficient.
        mu (float): Standard gravitational parameter.
        body_radius (float): Radius of the central body.

    Returns:
        numpy.ndarray: Acceleration vector due to C22.
    """
    r = state[:3]
    norm_r = np.linalg.norm(r)
    tx = r[0] * (-2 * norm_r**2 + 5 * r[0] ** 2 - 5 * r[1] ** 2)
    ty = r[1] * (-2 * norm_r**2 + 5 * r[0] ** 2 - 5 * r[1] ** 2)
    tz = r[2] * (r[0] ** 2 - r[1] ** 2)

    a_C22 = np.array(
        [
            ((-3 * C22 * mu * body_radius**2) / (norm_r**7)) * tx,
            ((-3 * C22 * mu * body_radius**2) / (norm_r**7)) * ty,
            ((-15 * C22 * mu * body_radius**2) / (norm_r**7)) * tz,
        ]
    )
    return a_C22


def S22(state, S22, mu, body_radius):
    """
    Calculates the acceleration due to the S22 sectorial harmonic.

    Args:
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        S22 (float): S22 sectorial harmonic coefficient.
        mu (float): Standard gravitational parameter.
        body_radius (float): Radius of the central body.

    Returns:
        numpy.ndarray: Acceleration vector due to S22.
    """

    r = state[:3]
    norm_r = np.linalg.norm(r)
    tx = r[1] * (5 * r[0] ** 2 - norm_r**2)
    ty = r[0] * (5 * r[1] ** 2 - norm_r**2)
    tz = r[0] * r[1] * r[2]

    a_S22 = np.array(
        [
            ((-6 * S22 * mu * body_radius**2) / (norm_r**7)) * tx,
            ((-6 * S22 * mu * body_radius**2) / (norm_r**7)) * ty,
            ((-30 * S22 * mu * body_radius**2) / (norm_r**7)) * tz,
        ]
    )
    return a_S22
