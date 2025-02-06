# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:30:02 2025

@author: ddiaz.beca
"""
import numpy as np
import coord_conversion
from scipy.special import lpmn
import math
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

#EL MODELO EGM96 TIENE COEF NORMALIZADOS, LOS POLINOMIOS DEBEN ESTARLO TAMBIEN
def normalize_legendre_polynomials(n, x):
    """
    Normalize the associated Legendre polynomials (Pnm) for use with normalized coefficients.

    Parameters:
    - n: maximum degree of the polynomial (int)
    - x: input value (sin(el), where el is the latitude or colatitude)

    Returns:
    - P_normalized: normalized associated Legendre polynomials (Pnm)
    """
    # Get the associated Legendre polynomials (without normalization)
    P, Pd = lpmn(n, n, x)
    
    P=P.T #scipy uses (m, n) indexing
    # Initialize the normalized polynomials
    P_normalized = np.zeros_like(P)
   
    # Normalize the polynomials
    for n_idx in range(n+1):
        
        for m_idx in range(n_idx+1):
                # For m=0
                normalization_factor = np.sqrt((2*n_idx+1) * math.factorial(n_idx - m_idx) / math.factorial(n_idx + m_idx))
                
                if m_idx > 0:
                    
                # Apply normalization factor
                    normalization_factor *= np.sqrt(2)
                    
                P_normalized[n_idx, m_idx] = P[n_idx, m_idx] *normalization_factor

   
    return P_normalized, P



def perturbation_potential(r, coeffs, mu, body_radius, max_order):
    """
    Calculation of the perturbed potential using Normalized Legendre Polynomials and armonical coefficients 
    from the EGM96 model

    Parameters
    ----------
    r : position
    coeffs : Normalized armonical coefficientes from EGM96 model
    mu : mu of the central body
    body_radius : radius of the central body
    max_order : Maximum order to have in count in the EGM96 model.

    Returns
    -------
    U_per : perturbed potential

    """
    _, az, el = coord_conversion.cartesian_to_spherical(r)
    x, y, z = r
    r_norm = np.linalg.norm(r)
    
    U=0  # terms w/o longitude dependence
     #terms with longitude dependence
    nrows, ncols = coeffs.shape
    if ncols > 1:
        
        if max_order is None:
            N = int(np.max(coeffs[:, 0:2]))  # Use all coefficients
        else:
            N = max_order  # Limit calculation to the specified order

        # Filter coefficients up to the maximum order
        coeffs_filtered = coeffs[coeffs[:, 0] <= N]
        #Normalized Legendre Polynomials
        P_norm, P= normalize_legendre_polynomials(N, np.sin(el))
        
        for row in coeffs_filtered:
            n, m, C, S=int(row[0]), int(row[1]), row[2], row[3]
            sum_factor=0
            
            for m_idx in range(m+1):
                Pnm = P_norm[n, m_idx]
                #print(f"Pnm{n, m_idx}: {Pnm}")
                
                sum_factor+=Pnm*(C*np.cos(m_idx*az)+S*np.sin(m_idx*az)) 
            U += (body_radius/r_norm)**n *sum_factor
            #print(f"U:{U}")
    U_per=U*(mu/r_norm)
    return U_per
    
    
    #calculation of the gradient
def gradient(f, r, h=1e-9):
    grad = np.zeros_like(r)
    perturbations = np.eye(len(r)) * h
    for i in range(len(r)):
        grad[i] = (f(r + perturbations[i]) - f(r - perturbations[i])) / (2 * h)
    return grad
def acceleration(state, coeffs, mu, body_radius, max_order):
    r = state[:3]
    U= lambda r: perturbation_potential(r, coeffs, mu, body_radius, max_order)
    
    grad_U = gradient(U, r)
    return grad_U
"""
mu= 465332.196
body_radius=6378.1
max_order= 8
state=[-8312.2,	0,	8998.7,	-7.822,	0,	0]
r = state[:3]
coeffs = read_egm96_coeffs('egm96_to360.ascii').astype(float)
a_pert= acceleration(state, coeffs, mu, body_radius, max_order)
n = 5
x = 0.5
P_normalized, P = normalize_legendre_polynomials(n, x)
"""
