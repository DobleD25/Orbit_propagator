# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:10:15 2025

@author: ddiaz.beca
"""

import numpy as np
import json

import astropy.constants as const
import astropy.units as u
import pandas as pd
import coord_conversion as cc
def read_json(input_file):
    np.set_printoptions(precision=3, suppress=True)
    
    # Leer archivo de entrada:
    with open(input_file, 'r') as file: 
        input = json.load(file)
        
    # Leer parámetros de las órbitas:
    orbits = input['Orbits']
    initial_states = []
    colors = []
    # Parámetros del cuerpo central:
    central_body = input['Central_body'][0]  # Asumiendo que Central_body es una lista con un diccionario
    body_params = {
        "name": central_body["name"],
        "radius": central_body["radius"] * u.km,
        "mass": central_body["mass"] * u.kg,
        "ecb": central_body.get("eclipsing_bodies", [])
    }
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    mu = G * body_params["mass"]
    mu_value = mu.value
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    
    
    
    # Print central body parameters
    print(f"""
          Central Body Parameters:
              ------------------------
              Name            : {body_params['name']}
              Radius          : {body_params['radius']:.4e}
              Mass            : {body_params['mass']:.4e}
              Mu              : {mu:.4e} 
              eclipsing bodies: {body_params['ecb']}
              """)
              
              
        # Spacecraft parameters
    spacecraft_data = input.get('Spacecraft', [])  # Handle missing 'Spacecraft' key
    spacecraft_params = []
    for spacecraft in spacecraft_data:
        spacecraft_params.append({
            "name": spacecraft["name"],
            "mass": spacecraft["mass"] * u.kg,  # Add units
            "area": spacecraft["area"] * u.m**2, # Add units and convert area to m^2
            "Cr": spacecraft["Cr"]
        })
        
        
        # Print spacecraft parameters - similar to central body print
        print(f"""
           Spacecraft Parameters:
              -----------------------
              Name   : {spacecraft['name']}
              Mass   : {spacecraft['mass']:.4e}
              Area   : {spacecraft['area']:.4e}
              Cr     : {spacecraft['Cr']}
              """)
    for orbit in orbits:
        coord_sys_orbit = orbit['system'].lower()
        print(f"Coordinate system: {coord_sys_orbit}")
        color = orbit['color']
        epoch= orbit['epoch']
        colors.append(color)
        if coord_sys_orbit == 'cartesians':
            coords_cart = orbit['Cartesians_coordinates'][0]
            x, y, z = coords_cart['x'], coords_cart['y'], coords_cart['z']
            vx, vy, vz = coords_cart['vx'], coords_cart['vy'], coords_cart['vz']
            initial_state_cart = np.array([x, y, z, vx, vy, vz])
            initial_states.append(initial_state_cart)
            initial_state_kepl = cc.cart_2_kep(initial_state_cart, mu_value)
            print(f"Initial state vector (cartesians): {np.round(initial_state_cart, 3)}")
            print(f"Initial state vector (keplerians): {np.round(initial_state_kepl, 3)}")
        
        elif coord_sys_orbit == 'keplerians':
            coords_kepl = orbit['Keplerian_coordinates'][0]
            a, e, i, Omega_AN, omega_PER, nu = coords_kepl.values()
            initial_state_kepl = np.array([a, e, i, Omega_AN, omega_PER, nu])
            initial_state_cart = cc.kep_2_cart(initial_state_kepl, mu_value)
            initial_states.append(initial_state_cart)
            print(f"Initial state vector (cartesians): {np.round(initial_state_cart, 3)}")
            print(f"Initial state vector (keplerians): {np.round(initial_state_kepl, 3)}")
        else:
            raise ValueError("Invalid coordinate system specified")

    dt = orbits[0]['step_size']
    tspan = orbits[0]['time_span']
    
    # Leer perturbaciones:
    perturbations = input.get('Perturbations', [{}])[0]
    Geopotential_bool = perturbations.get("Non_spherical_body", [{}])[0].get("value", False)
    N_body_bool = perturbations.get("N-body", [{}])[0].get("value", False)
    SRP_bool = perturbations.get("SRP", [{}])[0].get("value", False)  # Correct way to access SRP value
    
    coefficients = perturbations["Non_spherical_body"][0].get("coefficients", [{}])[0]
    if Geopotential_bool:
        
        EGM96_model=perturbations["Non_spherical_body"][0].get("EGM96_model")
        
        coef_pot = {k: coefficients.get(k, 0) for k in ["J2", "J3", "C22", "S22"]}
    if N_body_bool:
        n_body_list = perturbations.get("N-body", [{}])[0].get('list', [])
        
    if SRP_bool:
        ecb_list= perturbations.get("Central_body", [{}])[0].get('eclipsing_bodies', [])
    #dictionary of parameters:
    orbit_params = {
        "coord_sys_orbit": coord_sys_orbit,
        "initial_states": initial_states,
        "dt": dt,
        "tspan": tspan,
        "colors": colors,
        "epoch": epoch
    }
    perturbation_params = {
        "perturbations": perturbations
    }
    if Geopotential_bool:
        perturbation_params.update({
        "coef_pot": coef_pot,
        "EGM96_model": EGM96_model
    })
    else:
        perturbation_params.update({
        "coef_pot": "0",
        "EGM96_model": "0"
    })
    if "N-body" in perturbations:
        perturbation_params["N-body"] = perturbations["N-body"]
    if N_body_bool:
        perturbation_params.update({
            "body_list": n_body_list
        })
    if SRP_bool:
        perturbation_params["SRP"] = perturbations["SRP"]
        perturbation_params.update({
            
            "eclipsing_bodies": ecb_list
        })
    return body_params, orbit_params, perturbation_params, spacecraft_params

def read_EGM96_coeffs(file_path):
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