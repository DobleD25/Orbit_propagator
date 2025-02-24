# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:14:44 2025

@author: ddiaz.beca
"""

from astropy.constants import L_sun, R_sun, R_earth, c 

c_light=c
import spice_tool as st
import numpy as np
import planetary_data as pd
import spiceypy as spice
#TESTING
from astropy import units as u  
def solar_pressure(cb):
    R_cb=getattr(pd, cb.lower())['sma']
    P=L_sun/(4*np.pi*(c/1000)*R_cb)
    return P


def apparent_r(ocb, body_params, epoch, r) :
    """
    

    Parameters
    ----------
    ocb : occulting body
    body_params: paraneters of the central body (from input.json)
    epoch: time 
    r: possition vector of the S/C

    Returns
    -------
    Aparent radius and related coeficientes.
    a= apparent sun radius
    b=aparent occulting body radius
    c= apparent distance between center of ocb and the center of Sun
    """
    r_ocb=st.n_body(body_params["name"], ocb, epoch) #distance between the ocultting body and the Earth (Earth centered system)
    r_ocb=r_ocb[:3] #save only the position vector
    r_sat2ocb=r_ocb-r #distance between the satellite and the occulting body
    r_cb2sun=st.n_body(body_params["name"], "Sun", epoch) #distance between the central body and the Sun.
    r_cb2sun=r_cb2sun[:3]
    r_sat2sun=r_cb2sun-r
    a=np.arcsin(R_sun.value/1000/(np.linalg.norm(r_sat2sun)))
    b=np.arcsin(R_earth.value/1000/(np.linalg.norm(r_sat2ocb)))
    cos_c = np.dot(-r_sat2ocb, r_sat2sun) / (np.linalg.norm(r) * np.linalg.norm(r_sat2sun))
    c_ = np.arccos(np.clip(cos_c, -1.0, 1.0)) # Clip to [-1, 1]
    
    
    return a, b, c_


def F_srp(P, Cr, Area_m2, r_hat_sun):
    F = -P * Cr * Area_m2 * r_hat_sun
    return F


def SRP_a(ocb, body_params, sc_params, epoch, state, nu):
    
    """
    Calculation of the Solar Radiation Pressure using the canonball model
    
    Parameters
    ----------
    ocb : occulting body
    body_params: paraneters of the central body (from input.json)
    epoch: time (updated during the propagation 
    state: state vector of the S/C
    nu: shadow function 
    Returns
    -------
    a_srp : acceleration due to the SRP perturbation
    
    
    """
    r_cb2sun=st.n_body(body_params["name"], "Sun", epoch)
    r_cb2sun=r_cb2sun[:3] #position vector occulting body to sun
    r=state[:3] #position vector s/c
    r_sat2sun=r_cb2sun-r #position vector satellite to sun (km)
    r_sat2sun_meters = r_sat2sun*1000 * u.m #to S.I units (m)
    r_sat2sun_mnorm = np.linalg.norm(r_sat2sun_meters)
    #Solar pressure, N/m2
    P=L_sun/(4*np.pi*c*r_sat2sun_mnorm**2) #r_sat2sun from km to meter 
    a, b, c_ = apparent_r(ocb, body_params, epoch, r)
    
    Area_m2 = sc_params["area"]    #Área en m^2
    Mass_kg = sc_params["mass"] #Masa en kg
    Cr = sc_params["Cr"] #Coeficiente de reflectividad
    r_hat_sun = r_sat2sun_meters / r_sat2sun_mnorm
    
    #Inicialization of F:
    F=np.array([0.0, 0.0, 0.0])
    if (a + b) <= c_:
       F = F_srp(P, Cr, Area_m2, r_hat_sun)
    elif np.linalg.norm(a - b) < c_ < (a + b):  # ocultación parcial
       F = -nu * F_srp(P, Cr, Area_m2, r_hat_sun)
    elif c_ < (a - b): #ocultación total
       F = -nu * F_srp(P, Cr, Area_m2, r_hat_sun)
    elif c_ < (b - a): #umbra
       F = np.array([0.0, 0.0, 0.0])
        
    else: 
        print("Error in the geometry of the eclipse")
    a_srp_m = np.array([0.0, 0.0, 0.0])
    a_srp_m= F/(Mass_kg)    #acceleration of the perturbation (m/s2)
    a_srp_km = np.array([0.0, 0.0, 0.0])
    a_srp_km = a_srp_m/1000 # Convertir a km/s^2
    return a_srp_km.value
        
#TESTING
"""
spice.furnsh("data/naif0012.tls.pc")

    # solar system ephemeris kernel
spice.furnsh("data/de432s.bsp")
ocb="Earth"
body_params = {
    'name': 'Earth',
    'radius': 6378.1 * u.km,  # Correct: Quantity object directly
    'mass': 6.972e+24 * u.kg   # Correct: Quantity object directly
}

sc_params= {
    'Cr': 1,
    'mass': 1000,
    'area': 10
    }
epoch="2025-03-20T00:00:53"
et = spice.str2et(epoch)
state=[0, 9000, 0,	7.7, 0, 0]


a_srp=SRP_a(ocb, body_params, sc_params, et, state, nu=1)
nu=1.0
"""