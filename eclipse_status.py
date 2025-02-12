# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:26:19 2025

@author: ddiaz.beca
"""

from astropy.constants import L_sun, R_sun, R_earth, c 


import spice_tool as st
import numpy as np
import planetary_data as pd
import spiceypy as spice
#TESTING
from astropy import units as u  


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
    r_sat2sun=r_cb2sun-r #distance between the sat and the sun
    a=np.arcsin(R_sun.value/1000/(np.linalg.norm(r_sat2sun)))
    
    ecb_radius=getattr(pd, ocb.lower())['radius']
    b=np.arcsin(ecb_radius/(np.linalg.norm(r_sat2ocb)))
    
    cos_c = np.dot(-r_sat2ocb, r_sat2sun) / (np.linalg.norm(r_sat2ocb) * np.linalg.norm(r_sat2sun))
    c_ = np.arccos(np.clip(cos_c, -1.0, 1.0)) # Clip to [-1, 1]
    #c=np.arccos(np.dot(-r_sat2ocb, r_sat2sun)/(np.linalg.norm(r)*np.linalg.norm(r_sat2sun)))
    
    #x=(c_**2+a**2-b**2)/(2*c_)
    #y=np.sqrt(a**2-x**2)
    #Non occulted fraction
    #A=a**2*np.arccos(x/a)+b**2*np.arccos((c-x)/b)-c*y
    
    return a, b, c_


def eclipse(ocb, body_params, epoch, state):
    #r_cb2sun=st.n_body(ocb, "Sun", epoch)
    #r_cb2sun=r_cb2sun[:3] #position vector occulting body to sun
    r=state[:3] #position vector s/c
    #r_sat2sun=r_cb2sun-r #position vector satellite to sun
    #r_sat2sun_meters = r_sat2sun*1000 * u.m
    #r_sat2sun_mnorm = np.linalg.norm(r_sat2sun_meters)

    a, b, c_ = apparent_r(ocb, body_params, epoch, r) #Apparent radius
    
    
    eclipse_status=0
    if (a+b) <= c_:
        nu=1.0
        eclipse_status=0
       
    elif np.linalg.norm(a-b)< c_< a+b: #partial ocultation 
        x=(c_**2+a**2-b**2)/(2*c_)
        y=np.sqrt(a**2-x**2)
        A=a**2*np.arccos(x/a)+b**2*np.arccos((c_-x)/b)-c_*y
        nu=1-A/(np.pi*a**2)
        eclipse_status=1
    elif c_<a-b:
        nu=1-(b/a)**2
        eclipse_status=1
    elif c_<b-a:
        nu=0
        eclipse_status=2
        
    else: 
        print("Error in the geometry of the eclipse")
        nu=1.0
        
    
    
    return eclipse_status, nu


"""
-------------------TESTING--------------------------
ocb="Moon"
body_params = {
    'name': 'Earth',
    'radius': 6378.1 * u.km,  # Correct: Quantity object directly
    'mass': 6.972e+24 * u.kg   # Correct: Quantity object directly
}

sc_params= {
    'Cr': 1,
    'Mass': 1000,
    'Area': 10
    }
epoch="2025-03-20T00:00:53"
epoch_et = spice.str2et(epoch)
state=[6678, 0, 0,	7.7, 0, 0]

a, b, c_=apparent_r(ocb, body_params, epoch_et, state[:3])
eclipse_status, nu=eclipse(ocb, body_params, epoch_et, state)

"""