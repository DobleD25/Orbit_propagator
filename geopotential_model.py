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
#import sympy as sp
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
    Pd=Pd.T
    # Initialize the normalized polynomials
    P_normalized = np.zeros_like(P)
    Pd_normalized= np.zeros_like(Pd)
    # Normalize the polynomials
    for n_idx in range(n+1):
        
        for m_idx in range(n_idx+1):
                # For m=0
                normalization_factor = np.sqrt((2*n_idx+1) * math.factorial(n_idx - m_idx) / math.factorial(n_idx + m_idx))
                
                if m_idx > 0:
                    
                # Apply normalization factor
                    normalization_factor *= np.sqrt(2)
                    
                P_normalized[n_idx, m_idx] = P[n_idx, m_idx] *normalization_factor
                Pd_normalized[n_idx, m_idx] = Pd[n_idx, m_idx] *normalization_factor
   
    return P_normalized, Pd_normalized




def perturbation_potential_2(state, coeffs, mu, body_radius, max_order, epoch):
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
    rJ2000 = state[:3]
    r_bodyfixed=coord_conversion.J2000_to_bodyfixed(rJ2000, epoch)
    _, az, el = coord_conversion.cartesian_to_spherical(r_bodyfixed)
    x, y, z = r_bodyfixed
    r_norm = np.linalg.norm(r_bodyfixed)
    
    U=0  # terms w/o longitude dependence
     #terms with longitude dependence
    nrows, ncols = coeffs.shape
    term_spherical=np.zeros(3)
    gpert_spherical = np.zeros(3)  # Inicializar la aceleración en coordenadas esféricas
    if ncols > 1:
        
        if max_order is None:
            N = int(np.max(coeffs[:, 0]))  # Use all coefficients
        else:
            N = max_order  # Limit calculation to the specified order

        # Filter coefficients up to the maximum order
        coeffs_filtered = coeffs[coeffs[:, 0] <= N]
        #Normalized Legendre Polynomials
        P_norm, Pd_norm= normalize_legendre_polynomials(N, np.sin(el))
        """
        for row in coeffs_filtered:
            n, m, C, S=int(row[0]), int(row[1]), row[2], row[3]
            sum_factor=0
            
            for m_idx in range(n+1):
                Pnm = P_norm[n, m_idx]
                sum_factor+=Pnm*(C*np.cos(m_idx*az)+S*np.sin(m_idx*az)) 
            U += (body_radius/r_norm)**n *sum_factor
         """
        for row in coeffs_filtered:
            n, m, C, S = int(row[0]), int(row[1]), row[2], row[3] # Extraer grado, orden y coeficientes de la fila
            Pnm = P_norm[n, m] # Polinomio de Legendre Normalizado P_norm[n, m] específico para (n, m)
            Pnmd = Pd_norm[m, n]
            harmonic_term = Pnm * (C * np.cos(m * az) + S * np.sin(m * az)) # Término armónico angular
            U += (body_radius / r_norm)**n * harmonic_term # Sumar el término al potencial total
            term_spherical[0] += -(n+1)*(body_radius/r_norm)**n* Pnm * (C * np.cos(m * az) + S * np.sin(m * az))
            term_spherical[1] += (body_radius/r_norm)**n* Pnmd*(C * np.cos(m * az) + S * np.sin(m * az))
            cos_el_abs = abs(np.cos(el))
            pole_threshold = 1e-6  # Umbral para considerar que estamos cerca de un polo
            if cos_el_abs > pole_threshold:
                term_spherical[2] +=  (body_radius/r_norm)**n*m*(Pnm/ np.cos(el))*(S * np.cos(m * az) - C * np.sin(m * az))
            else: 
                term_spherical[2] += 0
               
    gpert_spherical=(mu/r_norm**2)*term_spherical
    U_per=U*(mu/r_norm)
    # Matriz de rotación para transformar de esféricas a cartesianas 
    
    rot = np.array([ [np.cos(el) * np.cos(az), -np.sin(el) * np.cos(az), -np.sin(az)], 
                    [np.cos(el) * np.sin(az), -np.sin(el) * np.sin(az), np.cos(az)],
                    [np.sin(el), np.cos(el),0 ]])
    gpert_bodyfixed_cartesian= np.dot(rot, gpert_spherical)
    #gpert_cartesian= coord_conversion.spherical_to_cartesian(gpert_spherical[0], gpert_spherical[1], gpert_spherical[2])
    #Ahora debemos pasarlo al marco inercial
    gpert_J2000=coord_conversion.bodyfixed_to_J200(gpert_bodyfixed_cartesian, epoch)
   
    
    #gpert_cartesian = astropy.coordinates.spherical_to_cartesian(gpert_spherical)
    #gpert_cartesian = coord_conversion.spherical_to_cartesian(gpert_spherical[0], gpert_spherical[1], gpert_spherical[2] )
    
    return gpert_J2000
"""
def acceleration_sym(state, coeffs, mu, body_radius, max_order):
    r = state[:3]
    
    x, y, z = sp.symbols('x y z')
    r_sym = sp.Matrix([x, y, z])
    U_sym = perturbation_potential(r_sym, coeffs, mu, body_radius, max_order)
    grad_U_sym = sp.Matrix([sp.diff(U_sym, var) for var in r_sym])
    grad_U_func = sp.lambdify((x, y, z), grad_U_sym)
    grad_U = grad_U_func(*r)
    #U= lambda r: perturbation_potential(r, coeffs, mu, body_radius, max_order)
    
    grad_U = grad_U_func(*r)
    return np.array(grad_U).astype(np.float64).flatten()
"""
"""
mu= 465332.196
epoch=790580819.1843239
body_radius=6378.1
max_order= 8
state=[0,	1000,	8998.7,	-7.822,	0,	0]
r = state[:3]
coeffs = read_egm96_coeffs('data/egm96_to360.ascii').astype(float)
a_pert= perturbation_potential_2(state, coeffs, mu, body_radius, max_order, epoch)
n = 5
x = 0.5
P_normalized, P = normalize_legendre_polynomials(n, x)
"""
