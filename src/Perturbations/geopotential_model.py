# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:30:02 2025

@author: ddiaz.beca
"""
import numpy as np
import coord_conversion
from scipy.special import lpmn
import math
import time_conversion as tc
from astropy.time import Time
import read as rd
import os


# import sympy as sp
def read_egm96_coeffs(file_path):
    """
    Reads the EGM96 model coefficients from an .ascii file.

    Parameters:
    - file_path: Path to the .ascii file containing the coefficients.

    Returns:
    - coeffs: Matrix of coefficients (Nx4) with columns [n, m, Cnm, Snm].
    """
    # Read the file, ignoring the standard deviation columns (columns 4 and 5)
    data = np.loadtxt(file_path, usecols=(0, 1, 2, 3))

    # Ensure the data is in float64 format for better precision
    coeffs = data.astype(np.float64)

    return coeffs


def egm96_model_bool(perturbation_params):
    """
    Determines if the EGM96 model should be used based on perturbation parameters.

    Args:
        perturbation_params (dict): Dictionary containing perturbation parameters.

    Returns:
        tuple: A tuple containing the coefficients and maximum order if the model is used, otherwise (None, None).
    """
    # Read egm96 coeffs if needed
    Non_spherical_body_bool = perturbation_params["perturbations"][
        "Non_spherical_body"
    ][0].get("value", False)
    egm96_model_bool = perturbation_params["perturbations"]["Non_spherical_body"][
        0
    ].get("EGM96_model", False)

    if egm96_model_bool and Non_spherical_body_bool:
        coeffs = rd.read_EGM96_coeffs(
            os.path.join(os.getcwd(), "data", "egm96_to360.ascii")
        ).astype(float)
        max_order = int(
            input("Enter the maximum degree for the EGM96 coefficients. \n")
        )

        return coeffs, max_order
    else:
        return None, None  # coeffs, max_order nulos


# EL MODELO EGM96 TIENE COEF NORMALIZADOS, LOS POLINOMIOS DEBEN ESTARLO TAMBIEN
def normalize_legendre_polynomials(n, x):
    """
    Normalize the associated Legendre polynomials (Pnm) for use with normalized coefficients.

    Parameters:
    - n (int): maximum degree of the polynomial.
    - x (float): input value (sin(el), where el is the latitude or colatitude).

    Returns:
    - tuple: A tuple containing the normalized associated Legendre polynomials (Pnm) and their derivatives (Pdnm).
    """
    # Get the associated Legendre polynomials (without normalization)
    P, Pd = lpmn(n, n, x)

    P = P.T  # scipy uses (m, n) indexing
    Pd = Pd.T
    # Initialize the normalized polynomials
    P_normalized = np.zeros_like(P)
    Pd_normalized = np.zeros_like(Pd)
    # Normalize the polynomials
    for n_idx in range(n + 1):

        for m_idx in range(n_idx + 1):
            # For m=0
            normalization_factor = np.sqrt(
                (2 * n_idx + 1)
                * math.factorial(n_idx - m_idx)
                / math.factorial(n_idx + m_idx)
            )

            if m_idx > 0:

                # Apply normalization factor
                normalization_factor *= np.sqrt(2)

            P_normalized[n_idx, m_idx] = P[n_idx, m_idx] * normalization_factor
            Pd_normalized[n_idx, m_idx] = Pd[n_idx, m_idx] * normalization_factor

    return P_normalized, Pd_normalized


def perturbation_potential_2(state, coeffs, mu, body_radius, max_order, epoch):
    """
    Calculation of the acceleration due to the non-symmetrical body using Normalized Legendre Polynomials and harmonic coefficients
    from the EGM96 model.

    Parameters:
    ----------
    state (numpy.ndarray): Position state vector [x, y, z, vx, vy, vz].
    coeffs (numpy.ndarray): Normalized harmonic coefficients from EGM96 model.
    mu (float): mu of the central body.
    body_radius (float): radius of the central body.
    max_order (int): Maximum order to have in count in the EGM96 model.
    epoch (float): Epoch in ephemeris time.

    Returns:
    -------
    numpy.ndarray: perturbed acceleration vector.
    """
    rJ2000 = state[:3]
    r_bodyfixed = coord_conversion.J2000_to_bodyfixed(rJ2000, epoch)
    _, az, el = coord_conversion.cartesian_to_spherical(r_bodyfixed)
    x, y, z = r_bodyfixed
    r_norm = np.linalg.norm(r_bodyfixed)

    U = 0  # terms w/o longitude dependence
    nrows, ncols = coeffs.shape
    term_spherical = np.zeros(3)
    gpert_spherical = np.zeros(3)  # Initialize acceleration in spherical coordinates
    if ncols > 1:

        if max_order is None:
            N = int(np.max(coeffs[:, 0]))  # Use all coefficients
        else:
            N = max_order  # Limit calculation to the specified order

        # Filter coefficients up to the maximum order
        coeffs_filtered = coeffs[coeffs[:, 0] <= N]
        # Normalized Legendre Polynomials
        P_norm, Pd_norm = normalize_legendre_polynomials(N, np.sin(el))

        for row in coeffs_filtered:
            n, m, C, S = (
                int(row[0]),
                int(row[1]),
                row[2],
                row[3],
            )  # Extract degree, order, and coefficients from the row
            Pnm = P_norm[
                n, m
            ]  # Normalized Legendre Polynomial P_norm[n, m] specific to (n, m)
            Pnmd = Pd_norm[m, n]
            harmonic_term = Pnm * (
                C * np.cos(m * az) + S * np.sin(m * az)
            )  # Angular harmonic term
            U += (
                body_radius / r_norm
            ) ** n * harmonic_term  # Add the term to the total potential
            term_spherical[0] += (
                -(n + 1)
                * (body_radius / r_norm) ** n
                * Pnm
                * (C * np.cos(m * az) + S * np.sin(m * az))
            )
            term_spherical[1] += (
                (body_radius / r_norm) ** n
                * Pnmd
                * (C * np.cos(m * az) + S * np.sin(m * az))
            )
            cos_el_abs = abs(np.cos(el))
            pole_threshold = 1e-6  # Threshold to consider being near a pole
            if cos_el_abs > pole_threshold:
                term_spherical[2] += (
                    (body_radius / r_norm) ** n
                    * m
                    * (Pnm / np.cos(el))
                    * (S * np.cos(m * az) - C * np.sin(m * az))
                )
            else:
                term_spherical[2] += 0
        # acceleration perturbation in spherical:
        gpert_spherical = (mu / r_norm**2) * term_spherical
        # Rotation matrix to transform from spherical to Cartesian

        rot = np.array(
            [
                [np.cos(el) * np.cos(az), -np.sin(el) * np.cos(az), -np.sin(az)],
                [np.cos(el) * np.sin(az), -np.sin(el) * np.sin(az), np.cos(az)],
                [np.sin(el), np.cos(el), 0],
            ]
        )
        gpert_bodyfixed_cartesian = np.dot(rot, gpert_spherical)
        # Now we must convert it to the inertial frame
        gpert_J2000 = coord_conversion.bodyfixed_to_J2000(
            gpert_bodyfixed_cartesian, epoch
        )

    return gpert_J2000
