# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:21:04 2025

@author: ddiaz.beca
"""

from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import numpy as np
import time_conversion as tc



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
    
    if N_nor == 0:  # Manejar el caso de órbita ecuatorial (N_nor = 0)
        raan = 0.0  # valor no definido
        aop = np.arctan2(e_vec[1], e_vec[0]) # convención (https://academia-lab.com/enciclopedia/argumento-de-periapsis/#google_vignette)
    else:
            # Right ascension of ascending node (RAAN)
        cos_raan = N[0] / N_nor
        cos_raan = np.clip(cos_raan, -1.0, 1.0)
        raan = np.arccos(cos_raan)
        if N[1] < 0:
            raan = 2 * np.pi - raan #Quadrant check

        # Argument of periapsis
        cos_aop = np.dot(N, e_vec) / (N_nor * e_nor)

        if e_nor == 0: # Manejar el caso de órbita circular (e_nor = 0)
            aop = 0.0 #aop no definido
        else:
            cos_aop = np.clip(cos_aop, -1.0, 1.0)
            aop = np.arccos(cos_aop)
            if e_vec[2] < 0:
                aop = 2 * np.pi - aop #quadrant check
    
    # True anomaly
    ta = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e_nor * r_nor), -1.0, 1.0))
    if np.dot(r_vec, v_vec) < 0:
        ta = 2 * np.pi - ta  # Quadrant check
    
    # Semi-major axis
    #a = r_nor * (1 + e_nor * np.cos(ta)) / (1 - e_nor**2)
    a = mu / (2*mu/r_nor - v_nor**2)
    
    
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
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z
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


def J2000_to_bodyfixed(state, epoch):
    r_j2000 = state[:3] # Position in J2000 frame

    # 1. Calculate Greenwich Hour Angle for the given epoch
    gha_radians, gha_degrees = tc.spiceET2GHA(epoch)

    # 2. Construct rotation matrix T to rotate from Body-Fixed to J2000
   
    T_BF_to_J2000 = np.array([[np.cos(gha_radians), -np.sin(gha_radians), 0],
                               [np.sin(gha_radians), np.cos(gha_radians), 0],
                               [0, 0, 1]])

    # 3. Calculate the inverse rotation matrix T_J2000_to_BF to rotate from J2000 to Body-Fixed
    T_J2000_to_BF = T_BF_to_J2000.T  # Transpose for inverse rotation

    # 4. Transform position vector from J2000 to Body-Fixed frame
    r_body_fixed_cartesian = np.dot(T_J2000_to_BF, r_j2000)
    
    return r_body_fixed_cartesian

def bodyfixed_to_J200(state, epoch):
    r_bodyfixed=state[:3]
    # 1. Calculate Greenwich Hour Angle for the given epoch
    gha_radians, gha_degrees = tc.spiceET2GHA(epoch)
    T_BF_to_J2000 = np.array([[np.cos(gha_radians), -np.sin(gha_radians), 0],
                               [np.sin(gha_radians), np.cos(gha_radians), 0],
                               [0, 0, 1]])
    
    r_J2000=np.dot(T_BF_to_J2000, r_bodyfixed)
    return r_J2000
    

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

