# -*- coding: utf-8 -*-
"""
@author: ddiaz.beca

"""

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pandas as pd
from astropy.constants import G 
import astropy.constants as const
import astropy.units as u

import coord_conversion
import plot_orbit
import geopotential_a
import n_methods
import geopotential_model





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
        "mass": central_body["mass"] * u.kg
    }
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    mu = G * body_params["mass"]
    mu_value = mu.value
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    
    
    
    # Print central body parameters
    print(f"""
          Central Body Parameters:
              ------------------------
              Name   : {body_params['name']}
              Radius : {body_params['radius']:.4e}
              Mass   : {body_params['mass']:.4e}
              Mu     : {mu:.4e} 
              """)
    
    for orbit in orbits:
        coord_sys_orbit = orbit['system'].lower()
        print(f"Coordinate system: {coord_sys_orbit}")
        color = orbit['color']
        colors.append(color)
        if coord_sys_orbit == 'cartesians':
            coords_cart = orbit['Cartesians_coordinates'][0]
            x, y, z = coords_cart['x'], coords_cart['y'], coords_cart['z']
            vx, vy, vz = coords_cart['vx'], coords_cart['vy'], coords_cart['vz']
            initial_state_cart = np.array([x, y, z, vx, vy, vz])
            initial_states.append(initial_state_cart)
            initial_state_kepl = coord_conversion.cart_2_kep(initial_state_cart, mu_value)
            print(f"Initial state vector (cartesians): {np.round(initial_state_cart, 3)}")
            print(f"Initial state vector (keplerians): {np.round(initial_state_kepl, 3)}")
        
        elif coord_sys_orbit == 'keplerians':
            coords_kepl = orbit['Keplerian_coordinates'][0]
            a, e, i, Omega_AN, omega_PER, nu = coords_kepl.values()
            initial_state_kepl = np.array([a, e, i, Omega_AN, omega_PER, nu])
            initial_state_cart = coord_conversion.kep_2_cart(initial_state_kepl, mu_value)
            initial_states.append(initial_state_cart)
            print(f"Initial state vector (cartesians): {np.round(initial_state_cart, 3)}")
            print(f"Initial state vector (keplerians): {np.round(initial_state_kepl, 3)}")
        else:
            raise ValueError("Invalid coordinate system specified")

    dt = orbits[0]['step_size']
    tspan = orbits[0]['time_span']
    
    # Leer perturbaciones:
    perturbations = input.get('Perturbations', [{}])[0]
    coeficients = perturbations.get("coeficients", [{}])[0]
    EGM96_model=perturbations.get("EGM96_model")
    coef_pot = {k: coeficients.get(k, 0) for k in ["J2", "J3", "C22", "S22"]}
    #dictionary of parameters:
    orbit_params = {
        "coord_sys_orbit": coord_sys_orbit,
        "initial_states": initial_states,
        "dt": dt,
        "tspan": tspan,
        "colors": colors
    }
    perturbation_params = {
        "perturbations": perturbations,
        "coef_pot": coef_pot,
        "EGM96_model": EGM96_model
    }
    return body_params, orbit_params, perturbation_params

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
    
def two_body_ode(t, state, perts, coef_dict, body_radius, mu):
    """
    Newton's Universal Law of Gravitation
    """
    r = state[:3]
    a = -mu * r / np.linalg.norm(r)**3
    # Potential perturbations
    if perts.get('Non_spherical_body', False) and perturbation_params['EGM96_model']==False:
        
        
        a_j2=0
        a_j3=0
        a_C22=0
        a_S22=0
        if 'J2' in coef_dict:
            a_j2 = geopotential_a.j2(r, coef_dict['J2'], mu, body_radius)
        if 'J3' in coef_dict:
            a_j3 = geopotential_a.j3(r, coef_dict['J3'], mu, body_radius)
        if 'C22' in coef_dict:
            a_C22 = geopotential_a.C22(r, coef_dict['C22'], mu, body_radius) 
        if 'S22' in coef_dict:
            a_S22 = geopotential_a.S22(r, coef_dict['S22'], mu, body_radius) 
        a += a_j2+a_j3+a_C22+a_S22
    elif perturbation_params['EGM96_model']==True:
       a_pert=0
       a_pert = geopotential_model.acceleration(state, coeffs, mu, body_radius, max_order)
       
       a += a_pert
     
    return np.array([state[3], state[4], state[5], a[0], a[1], a[2]])

#Integration of the ODE
def solve_orbit(method, steps, ets, states, dt, pert, coef_pot, body_radius, mu):
    f_prev = [None] * 4  # To store previous function evaluations
    if method == "1":
        for step in range(steps - 1):
            states[step + 1] = n_methods.rk4_step(lambda t, y: two_body_ode(t, y, pert, coef_pot , body_radius.value, mu), ets[step], states[step], dt)
    elif method == "2":
        for step in range(steps - 1):
            if step < 3:
                # Use RK4 for the first three steps to get initial values
                states[step + 1] = n_methods.rk4_step(lambda t, y: two_body_ode(t, y, pert, coef_pot, body_radius.value, mu), ets[step], states[step], dt)
                f_prev[step] = two_body_ode(ets[step], states[step], pert, coef_pot, body_radius.value, mu)
            else:
                f_prev[3] = f_prev[2]
                f_prev[2] = f_prev[1]
                f_prev[1] = f_prev[0]
                f_prev[0] = two_body_ode(ets[step], states[step], pert, coef_pot, body_radius.value, mu)
                states[step + 1] = n_methods.adams_predictor_corrector(lambda t, y: two_body_ode(t, y, pert, coef_pot, body_radius.value, mu), ets[step], states[step], dt, f_prev)
    elif method == "3":
        states, ets = n_methods.lsoda_solver(lambda t, y: two_body_ode(t, y, pert, coef_pot, body_radius.value, mu), states, ets[0], ets[-1], dt)
    elif method == "4":
        states, ets = n_methods.zvode_solver(lambda t, y: two_body_ode(t, y, pert, coef_pot, body_radius.value, mu), states, ets[0], ets[-1], dt)
    else:
        print("Incorrect value")
    return states, ets


if __name__ == '__main__':
    
    # Read Input
    body_params, orbit_params, perturbation_params = read_json("Input.json")
    #Read egm96 coeffs if needed
    
    if perturbation_params['EGM96_model']:
        coeffs = read_EGM96_coeffs('egm96_to360.ascii').astype(float)
        max_order = int(input("""Enter the maximum degree for the EGM96 coefficients 
                              Danger: The calculation time increases highly with the degree
                              
                              """))
    
    #mu:
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    mu = (G * body_params['mass']).value
    
    #init trajectories:
    trajectories = []
    states_kepl_all = []
    # Orbit colors
    #colors = ['cyan']
    labels = [f'Orbit {i+1}' for i in range(len(orbit_params['initial_states']))]
    
    for idx, initial_state in enumerate(orbit_params['initial_states']):
        # Inicialización de vectores
        steps = int(orbit_params['tspan'] / orbit_params['dt'])
        states = np.zeros((steps, 6))
        states[0] = initial_state
        ets = np.linspace(0, (steps - 1) * orbit_params['dt'], steps)
    
        # Propagation
        method = input(f"""Select number of the resolution method for orbit {idx+1}:
                       1- Runge-Kutta 4th order 
                       2- Adams Predictor-Corrector
                       3- Lsoda (scipy method) 
                       4- Vode (scipy method)
                       """)
        start_time = time.time()  # Execution time start
        # Propagation of the orbit
        states, ets = solve_orbit(method, steps, ets, states, orbit_params['dt'], perturbation_params['perturbations'], perturbation_params['coef_pot'], body_params['radius'], mu)
        end_time = time.time()
        # Execution time:
        execution_time = end_time - start_time
        print(f"Execution finished in {execution_time:.3f}s. Generating state vectors files.")
        # Save trajectories:
        trajectories.append(states[:, :3])
        # Conversion to Keplerian elements
        states_kepl = np.apply_along_axis(coord_conversion.cart_2_kep, 1, states, mu)
        states_kepl_all.append(states_kepl)
    
        # Save files
        df_states = pd.DataFrame(states, columns=['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz'])
        df_states.insert(0, 'Time', ets)
        df_states.to_csv(f'State_vectors_orbit_{idx+1}.csv', index_label='Step')

# Plotting 3D orbit
fig_3d, ax_3d = plot_orbit.setup_3d_plot()
plot_orbit.plot_3D(ax_3d, trajectories, body_params['radius'], orbit_params['colors'], labels)

# Plotting the COEs
fig_coes, axs_coes = plot_orbit.setup_coes_plots(len(orbit_params['initial_states']))
plot_orbit.plot_coes(axs_coes, [ets] * len(orbit_params['initial_states']), states_kepl_all, orbit_params['colors'], labels)


    