# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:21:04 2025

@author: ddiaz.beca
"""

from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import numpy as np




def EA(EA_k, M, e): #excentric anomaly, iterative solution
    return EA_k-(EA_k-e*np.sin(EA_k)-M)/(1-e*np.cos(EA_k))


def newton_iterative(M, e, tol=1e86, max_iter=100): 
    EA_k = M #initial:guess
    for _ in range(max_iter):
        EA_k1 = EA(EA_k, M, e)
        if abs(EA_k1 - EA_k) < tol:
            return EA_k1
        EA_k = EA_k1
    raise RuntimeError("Error in the calculus of EA. Could not converge")
    

def cart_2_kep(state_vector, mu):
    """
    Cartesians to Keplerians coordinates.
    Input: state vector (x, y, z, vx, vy, vz)
    Output: state vector(a, e, i, Omega_AN (Longitude of the Ascending Node), omega_per (Argument of Periapsis), nu (True anomaly))
    """
    r_vec = state_vector[0:3]   # Position vector
    r_nor = np.linalg.norm(r_vec)
    v_vec = state_vector[3:6]   # Velocity vector
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
    
    # Right ascension of ascending node (RAAN)
    raan = np.arccos(N[0] / N_nor)
    if N[1] < 0:
        raan = 2 * np.pi - raan  # Quadrant check
    
    # Argument of periapsis
    cos_aop = np.dot(N, e_vec) / (N_nor * e_nor)
    cos_aop=np.clip(cos_aop, -1, 1)
    aop = np.arccos(cos_aop)
    if e_vec[2] < 0:
        aop = 2 * np.pi - aop  # Quadrant check
    
    # True anomaly
    ta = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e_nor * r_nor), -1.0, 1.0))
    if np.dot(r_vec, v_vec) < 0:
        ta = 2 * np.pi - ta  # Quadrant check
    
    # Semi-major axis
    a = r_nor * (1 + e_nor * np.cos(ta)) / (1 - e_nor**2)
    
    
    
    return [a, e_nor, np.rad2deg(i), np.rad2deg(raan), np.rad2deg(aop), np.rad2deg(ta)]
def kep_2_cart(state_vector, mu):
    """
    Keplerians to Cartesians coordinates
    Input: state vector(a, e, i, Omega_AN (Longitude of the Ascending Node), omega_per (Argument of Periapsis), nu (True anomaly))
    Output: state vector(x, y, z, vx, vy, vz)
    """
    a, e, i, omega_AN, omega_PER, nu = state_vector
    # Convert degrees to radians
    i = np.deg2rad(i)
    omega_AN = np.deg2rad(omega_AN)
    omega_PER = np.deg2rad(omega_PER)
    nu = np.deg2rad(nu)
    #1
    
    # Mean anomaly (M) calculation
    M = np.arctan2(-np.sqrt(1 - e**2) * np.sin(nu), -e - np.cos(nu)) + np.pi - e * (np.sqrt(1 - e**2) * np.sin(nu) / (1 + e * np.cos(nu)))
    #2 Solve for eccentric anomaly (EA) using an iterative method
    EA=newton_iterative(M, e)
    #4 Calculate the radius (r
    r = a*(1 - e*np.cos(EA))
    #5 Calculate the specific angular momentum (h)
    h = np.sqrt(mu*a * (1 - e**2))
    #6 Assign the angles
    Om = omega_AN
    w =  omega_PER
    #Calculate the position coordinates (X, Y, Z)
    X = r*(np.cos(Om)*np.cos(w+nu) - np.sin(Om)*np.sin(w+nu)*np.cos(i))
    Y = r*(np.sin(Om)*np.cos(w+nu) + np.cos(Om)*np.sin(w+nu)*np.cos(i))
    Z = r*(np.sin(i)*np.sin(w+nu))

    #7 Calculate the semi-latus rectum (p)
    p = a*(1-e**2)
    #Calculate the velocity components (V_X, V_Y, V_Z)
    V_X = (X*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(Om)*np.sin(w+nu) + \
    np.sin(Om)*np.cos(w+nu)*np.cos(i))
    V_Y = (Y*h*e/(r*p))*np.sin(nu) - (h/r)*(np.sin(Om)*np.sin(w+nu) - \
    np.cos(Om)*np.cos(w+nu)*np.cos(i))
    V_Z = (Z*h*e/(r*p))*np.sin(nu) + (h/r)*(np.cos(w+nu)*np.sin(i))

    return [X,Y,Z,V_X,V_Y,V_Z]


def cartesian_to_spherical(state):
    x=state[0]
    y=state[1]
    z=state[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x) #longitud
    el = np.arcsin(z / r) #latitud 
    return r, az, el

def e_vector(e, raan, aop):
    """
    Definition of the two-dimensional eccentricity vector
    
    """
    
    e_vec=[e*np.cos(raan+aop), e*np.sin(raan+aop)]
    
    return e_vec

def i_vector(i, raan):
    """
    Definition of the 2-dimensional inclination vector

    """
    i_vec=[i*np.sin(raan), -i*np.cos(raan)]
    
    return i_vec

#Test vectors
"""
mu=G.value*M_earth.value
r_test = np.array([2660*1000, 0*1000, 1260*1000])
v_test = np.array([7.8*1000, 0, 0])
state_vector=np.hstack((r_test, v_test))
t = 0
state_kepl = cart_2_kep(state_vector, mu)

e_vec=e_vector(state_kepl[1], np.deg2rad(state_kepl[3]), np.deg2rad(state_kepl[4]))
state_vector_cart = kep_2_cart(state_vector_klep, mu)
print(state_vector_cart-state_vector)
"""

