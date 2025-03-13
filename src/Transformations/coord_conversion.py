# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:21:04 2025

@author: ddiaz.beca
"""


import numpy as np
import time_conversion as tc


def EA(EA_k, M, e):  # excentric anomaly, iterative solution
    """
    Calculates the eccentric anomaly using an iterative formula.

    Args:
        EA_k (float): Eccentric anomaly in the previous iteration.
        M (float): Mean anomaly.
        e (float): Eccentricity.

    Returns:
        float: The updated eccentric anomaly.
    """
    return EA_k - (EA_k - e * np.sin(EA_k) - M) / (1 - e * np.cos(EA_k))


def newton_iterative(M, e, tol=1e-8, max_iter=100):
    """
    Calculates the eccentric anomaly using Newton's iterative method.

    Args:
        M (float): Mean anomaly.
        e (float): Eccentricity.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: The eccentric anomaly.

    Raises:
        RuntimeError: If the method fails to converge.
    """
    EA_k = M  # initial:guess
    for _ in range(max_iter):
        EA_k1 = EA(EA_k, M, e)
        if abs(EA_k1 - EA_k) < tol:
            return EA_k1
        EA_k = EA_k1
    raise RuntimeError("Error in the calculus of EA. Could not converge")


def cart_2_kep(state_vector, mu):
    """
    Converts Cartesian coordinates to Keplerian elements.

    Args:
        state_vector (numpy.ndarray): Cartesian state vector (x, y, z, vx, vy, vz).
        mu (float): Standard gravitational parameter.

    Returns:
        list: Keplerian state vector (a, e, i, RAAN, aop, nu).
    """
    r_vec = state_vector[0:3]  # Position vector
    r_nor = np.linalg.norm(r_vec)
    v_vec = state_vector[3:6]  # Velocity vector
    v_nor = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_bar = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_bar)

    # Inclination
    i = np.arccos(h_bar[2] / h)

    # Eccentricity vector
    e_vec = ((v_nor**2 - mu / r_nor) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e_nor = np.linalg.norm(e_vec)

    # Node line
    N = np.cross([0, 0, 1], h_bar)
    N_nor = np.linalg.norm(N)

    if N_nor == 0:  # Handle equatorial orbit case (N_nor = 0)
        raan = 0.0  # Undefined value
        aop = np.arctan2(
            e_vec[1], e_vec[0]
        )  # Convention (https://academia-lab.com/enciclopedia/argumento-de-periapsis/#google_vignette)
    else:
        # Right ascension of ascending node (RAAN)
        cos_raan = N[0] / N_nor
        cos_raan = np.clip(cos_raan, -1.0, 1.0)
        raan = np.arccos(cos_raan)
        if N[1] < 0:
            raan = 2 * np.pi - raan  # Quadrant check

        # Argument of periapsis
        cos_aop = np.dot(N, e_vec) / (N_nor * e_nor)

        if e_nor == 0:  # Handle circular orbit case (e_nor = 0)
            aop = 0.0  # Undefined aop
        else:
            cos_aop = np.clip(cos_aop, -1.0, 1.0)
            aop = np.arccos(cos_aop)
            if e_vec[2] < 0:
                aop = 2 * np.pi - aop  # Quadrant check

    # True anomaly
    ta = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e_nor * r_nor), -1.0, 1.0))
    if np.dot(r_vec, v_vec) < 0:
        ta = 2 * np.pi - ta  # Quadrant check

    # Semi-major axis
    a = mu / (2 * mu / r_nor - v_nor**2)

    return [a, e_nor, np.rad2deg(i), np.rad2deg(raan), np.rad2deg(aop), np.rad2deg(ta)]


def kep_2_cart(state_vector, mu):
    """
    Converts Keplerian elements to Cartesian coordinates.

    Args:
        state vector(a, e, i, RAAN (Longitude of the Ascending Node), aop (Argument of Periapsis), nu (True anomaly))
        mu (float): Standard gravitational parameter.
    Return:
        state vector(x, y, z, vx, vy, vz)
    """
    a, e, i, RAAN, aop, nu = state_vector
    # Convert degrees to radians
    i = np.deg2rad(i)
    RAAN = np.deg2rad(RAAN)
    aop = np.deg2rad(aop)
    nu = np.deg2rad(nu)
    # 1

    # Mean anomaly (M) calculation
    M = (
        np.arctan2(-np.sqrt(1 - e**2) * np.sin(nu), -e - np.cos(nu))
        + np.pi
        - e * (np.sqrt(1 - e**2) * np.sin(nu) / (1 + e * np.cos(nu)))
    )
    
    # 2 Solve for eccentric anomaly (EA) using an iterative method
    EA = newton_iterative(M, e)
    # 4 Calculate the radius (r
    r = a * (1 - e * np.cos(EA))
    # 5 Calculate the specific angular momentum (h)
    h = np.sqrt(mu * a * (1 - e**2))
    # 6 Assign the angles
    Om = RAAN
    w = aop
    # Calculate the position coordinates (X, Y, Z)
    X = r * (np.cos(Om) * np.cos(w + nu) - np.sin(Om) * np.sin(w + nu) * np.cos(i))
    Y = r * (np.sin(Om) * np.cos(w + nu) + np.cos(Om) * np.sin(w + nu) * np.cos(i))
    Z = r * (np.sin(i) * np.sin(w + nu))

    # 7 Calculate the semi-latus rectum (p)
    p = a * (1 - e**2)
    # Calculate the velocity components (V_X, V_Y, V_Z)
    V_X = (X * h * e / (r * p)) * np.sin(nu) - (h / r) * (
        np.cos(Om) * np.sin(w + nu) + np.sin(Om) * np.cos(w + nu) * np.cos(i)
    )
    V_Y = (Y * h * e / (r * p)) * np.sin(nu) - (h / r) * (
        np.sin(Om) * np.sin(w + nu) - np.cos(Om) * np.cos(w + nu) * np.cos(i)
    )
    V_Z = (Z * h * e / (r * p)) * np.sin(nu) + (h / r) * (np.cos(w + nu) * np.sin(i))

    return [X, Y, Z, V_X, V_Y, V_Z]


def cartesian_to_spherical(state):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Args:
        state (numpy.ndarray): Cartesian state vector (x, y, z).

    Returns:
        tuple: Spherical coordinates (r, az, el).
    """
    x = state[0]
    y = state[1]
    z = state[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)  # longitud
    el = np.arcsin(z / r)  # latitud
    return r, az, el


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Args:
        r (float): Radius.
        theta (float): Azimuth angle (longitude).
        phi (float): Elevation angle (latitude).

    Returns:
        tuple: Cartesian coordinates (x, y, z).
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def e_vector(e, raan, aop):
    """
    Defines the two-dimensional eccentricity vector.

    Args:
        e (float): Eccentricity.
        raan (float): Right ascension of ascending node (RAAN).
        aop (float): Argument of periapsis.

    Returns:
        list: Eccentricity vector [e_x, e_y].
    """

    e_vec = [e * np.cos(raan + aop), e * np.sin(raan + aop)]

    return e_vec


def i_vector(i, raan):
    """
    Defines the 2-dimensional inclination vector.

    Args:
        i (float): Inclination.
        raan (float): Right ascension of ascending node (RAAN).

    Returns:
        list: Inclination vector [i_x, i_y].
    """
    i_vec = [i * np.sin(raan), -i * np.cos(raan)]

    return i_vec


def J2000_to_bodyfixed(state, epoch):
    """
    Transforms a position vector from J2000 frame to Body-Fixed frame.

    Args:
        state (numpy.ndarray): State vector in J2000 frame.
        epoch (float): Epoch in ephemeris time.

    Returns:
        numpy.ndarray: Position vector in Body-Fixed frame.
    """
    r_j2000 = state[:3]  # Position in J2000 frame

    # 1. Calculate Greenwich Hour Angle for the given epoch
    gha_radians, gha_degrees = tc.spiceET2GHA(epoch)

    # 2. Construct rotation matrix T to rotate from Body-Fixed to J2000

    T_BF_to_J2000 = np.array(
        [
            [np.cos(gha_radians), -np.sin(gha_radians), 0],
            [np.sin(gha_radians), np.cos(gha_radians), 0],
            [0, 0, 1],
        ]
    )

    # 3. Calculate the inverse rotation matrix T_J2000_to_BF to rotate from J2000 to Body-Fixed
    T_J2000_to_BF = T_BF_to_J2000.T  # Transpose for inverse rotation

    # 4. Transform position vector from J2000 to Body-Fixed frame
    r_body_fixed_cartesian = np.dot(T_J2000_to_BF, r_j2000)

    return r_body_fixed_cartesian


def bodyfixed_to_J2000(state, epoch):
    """
    Transforms a position vector from Body-Fixed frame to J2000 frame.

    Args:
        state (numpy.ndarray): State vector in Body-Fixed frame.
        epoch (float): Epoch in ephemeris time.

    Returns:
        numpy.ndarray: Position vector in J2000 frame.
    """
    r_bodyfixed = state[:3]
    # 1. Calculate Greenwich Hour Angle for the given epoch
    gha_radians, gha_degrees = tc.spiceET2GHA(epoch)
    T_BF_to_J2000 = np.array(
        [
            [np.cos(gha_radians), -np.sin(gha_radians), 0],
            [np.sin(gha_radians), np.cos(gha_radians), 0],
            [0, 0, 1],
        ]
    )

    r_J2000 = np.dot(T_BF_to_J2000, r_bodyfixed)
    return r_J2000


def proyect_lat_lon(state_J2000, epoch):
    """
    Calculates spherical coordinates in Body-Fixed frame.

    Args:
        state_J2000 (numpy.ndarray): State vector in J2000 frame.
        epoch (float or numpy.ndarray): Epoch in ephemeris time.

    Returns:
        numpy.ndarray: Spherical coordinates in Body-Fixed frame.
    """
    r_J2000 = state_J2000[:3]

    if isinstance(epoch, (list, np.ndarray)):  # Handle array of epochs
        if isinstance(state_J2000, np.ndarray) and state_J2000.ndim == 1:
            raise ValueError("state must be a 2d array when epoch is a array")

        spherical_coords = []
        for i in range(len(epoch)):
            r_body_fixed_cart = J2000_to_bodyfixed(r_J2000, epoch[i])
            r, lon, lat = cartesian_to_spherical(r_body_fixed_cart)
            spherical_coords.append([r, lon, lat])
        return np.array(spherical_coords)
    else:  # Handle single epoch
        r_body_fixed_cart = J2000_to_bodyfixed(r_J2000, epoch)
        r, lon, lat = cartesian_to_spherical(r_body_fixed_cart)
        return np.array([r, lon, lat])


def fun_i_vec(state_kepl, i_vec_orbit):
    """
    Calculates and appends the inclination vector to the orbit's ephemeris list.

    Args:
        state_kepl (list): Keplerian state vector.
        i_vec_orbit (list): List to store inclination vectors.

    Returns:
        list: Updated inclination vector ephemeris list.
    """
    i_vec = i_vector(
        np.deg2rad(state_kepl[2]), np.deg2rad(state_kepl[3])
    )  # input: i, raan in RADIANS
    i_vec_orbit.append(i_vec)  # append to the i_vec ephemerids list for that orbit

    return i_vec_orbit  # i_vec ephemerids list for orbit idx


def fun_e_vec(state_kepl, e_vec_orbit):
    """
    Calculates and appends the eccentricity vector to the orbit's ephemeris list.

    Args:
        state_kepl (list): Keplerian state vector.
        e_vec_orbit (list): List to store eccentricity vectors.

    Returns:
        list: Updated eccentricity vector ephemeris list.
    """
    e_vec = e_vector(
        state_kepl[1], np.deg2rad(state_kepl[3]), np.deg2rad(state_kepl[4])
    )  # input: e, raan, aop in radians
    e_vec_orbit.append(e_vec)  # append to the e_vec ephemerids list for that orbit
    return e_vec_orbit  # e_vec ephemerids list for orbit idx
