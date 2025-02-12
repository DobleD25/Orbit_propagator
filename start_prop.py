# -*- coding: utf-8 -*-
"""
@author: ddiaz.beca

"""

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from astropy.constants import G 
import astropy.constants as const
import astropy.units as u

import coord_conversion as cc
import plot_orbit as po
import geopotential_1 as geop1
import n_methods as nm
import geopotential_model as gm
import n_body_per as nb
import SRP as srp
import eclipse_status as es
import spiceypy as spice
import read as rd

#spice.unload("data/naif0012.tls.pc")
#spice.unload("data/de432s.bsp")
spice.kclear()
    # Leap second kernel
spice.furnsh("data/naif0012.tls.pc")

    # solar system ephemeris kernel
spice.furnsh("data/de432s.bsp")


    
def two_body_ode(t, state, perts, coef_dict, cb_radius, mu_cb, epoch_t, nu):
    """
    Newton's Universal Law of Gravitation with the perturbations
    
    Input:
        t: time
        state: state vector
        perts: dictonary with the perturbations data
        coef_dict: dictionary with the potential coeffs.
        cb_radius: radius of the central body.
        mu_cb: mu of the central body
        epoch_t: epoch of the state vector
        nu: value of the shadow function in that state
    
    
    Output:
        vector: [v_x, v_y, v_z, a_x, a_y, a_z]
        
    """
    r = state[:3]
    a = -mu_cb * r / np.linalg.norm(r)**3 
    
    # Potential perturbations
    if perts.get('Non_spherical_body', False) and perturbation_params['EGM96_model']==False:
        
        a_j2=0
        a_j3=0
        a_C22=0
        a_S22=0
        if 'J2' in coef_dict:
            a_j2 = geop1.j2(r, coef_dict['J2'], mu_cb, cb_radius)
        if 'J3' in coef_dict:
            a_j3 = geop1.j3(r, coef_dict['J3'], mu_cb, cb_radius)
        if 'C22' in coef_dict:
            a_C22 = geop1.C22(r, coef_dict['C22'], mu_cb, cb_radius) 
        if 'S22' in coef_dict:
            a_S22 = geop1.S22(r, coef_dict['S22'], mu_cb, cb_radius) 
        a += a_j2+a_j3+a_C22+a_S22
    elif perturbation_params['EGM96_model']==True:
       
       a_pert = gm.acceleration(state, coeffs, mu_cb, cb_radius, max_order)
       
       a += a_pert
    if perturbation_params['N-body'][0]["value"]:
        #a_pert = np.array([0.0, 0.0, 0.0])
        a_pert=nb.n_body_a(perturbation_params, body_params, epoch_t, r)
        a += a_pert
    #srp = perturbation_params.get('SRP', None)    
    
    if perturbation_params['SRP'][0]["value"]:
        for ecb in body_params['ecb']:
            #_, nu=es.eclipse(ecb, body_params, epoch_et, state)
            a_pert=srp.SRP_a(ecb, body_params, spacecraft_params[0], epoch_t, state, nu)
            a += a_pert
        
    print(f"State vector: {state}")
    return np.array([state[3], state[4], state[5], a[0], a[1], a[2]])

def calculate_nu(t, y, ecb_list, body_params, epoch_et):  # Calculation of the shadow function 
    nu = None  # Inicializa nu
    for i, ecb in enumerate(ecb_list):
        _, nu = es.eclipse(ecb, body_params, t, y)  # Calcula nu para el tiempo y estado actual
    return nu
def two_body_ode_with_nu(t, y, perts, coef_pot, cb_radius, mu_cb, epoch_et, ecb_list, body_params): #ODE with the shadow funtion. 
    """
    Two-body ODE with the shadow function

    Input:
        t: time
        y: state vector
        perts: dictionary with the perturbations data
        coef_pot: potential coefficients
        cb_radius: radius of the central body
        mu_cb: mu of the central body
        epoch_et: epoch of the state vector
        ecb_list: list of eclipsing bodies
        body_params: dictionary with body parameters

    Output:
        result of two_body_ode with the appropiate nu value
    """
    nu = calculate_nu(t, y, ecb_list, body_params, t)
    return two_body_ode(t, y, perts, coef_pot, cb_radius, mu_cb, t, nu)
#Integration of the ODE
def solve_orbit(method, steps, ets, states, orbit_params, perturbation_params, body_params):
    """
   Integrates the orbit using the specified method.

   Input:
       method: integration method ('1' for RK4, '2' for Adams-Predictor-Corrector, '3' for LSODA, '4' for ZVODE, '5' for RK45)
       steps: number of steps
       ets: array of time steps
       states: initial state vector
       orbit_params: dictionary with orbit parameters
       perturbation_params: dictionary with perturbation parameters
       body_params: dictionary with body parameters

   Output:
       states: array of state vectors at each time step
       ets: array of time steps
       eclipse_statuses: array of eclipse statuses at each time step and for each eclipsing body
   """
    f_prev = [None] * 4  # To store previous function evaluations
    eclipse_statuses = []  # List to store eclipse status for each step
    dt=orbit_params["dt"]
    epoch=orbit_params["epoch"]
    pert=perturbation_params["perturbations"]
    coef_pot=perturbation_params["coef_pot"]
    cb_radius=body_params["radius"]
    mu_cb=body_params["mass"]*G
    ecb_list=body_params["ecb"]
    
    epoch_et = spice.str2et(epoch)# Tiempo absoluto del epoch inicial
    
    #Eclipsis status: 
        #0: No eclipse
        #1: Partial eclipse
        #2: Total eclipse
    eclipse_statuses = np.zeros((steps, len(ecb_list)), dtype=int) # Array to store eclipse status for each step and eclipsing body
    if method == "1":
        for step in range(steps - 1):
            epoch_t = epoch_et + ets[step]  # Calcula el tiempo absoluto en este paso
            for i, ecb in enumerate(ecb_list): #Iterate over the eclipsing bodies
                eclipse_statuses[step, i], nu = es.eclipse(ecb, body_params, epoch_t, states[step]) #Calculate the eclipse status
                
            states[step + 1] = nm.rk4_step(lambda t, y: two_body_ode(t, y, pert, coef_pot , cb_radius.value, mu_cb.value, epoch_et, nu), ets[step], states[step], dt )
            
    elif method == "2":
        
        
        for step in range(steps - 1):
            epoch_t = epoch_et + ets[step]  # Calcula el tiempo absoluto en este paso
            for i, ecb in enumerate(ecb_list): #Iterate over the eclipsing bodies
                eclipse_statuses[step, i], nu = es.eclipse(ecb, body_params, ets[step], states[step]) ##calculus of the eclipse status and the shadow function
            if step < 3:
                # Use RK4 for the first three steps to get initial values
                states[step + 1] = nm.rk4_step(lambda t, y: two_body_ode(t, y, pert, coef_pot, cb_radius.value, mu_cb.value, epoch_t, nu), ets[step], states[step], dt)
                f_prev[step] = two_body_ode(ets[step], states[step], pert, coef_pot, cb_radius.value, mu_cb.value, epoch_t, nu)
            else:
                f_prev[3] = f_prev[2]
                f_prev[2] = f_prev[1]
                f_prev[1] = f_prev[0]
                f_prev[0] = two_body_ode(ets[step], states[step], pert, coef_pot, cb_radius.value, mu_cb.value, epoch_t, nu)
                states[step + 1] = nm.adams_predictor_corrector(lambda t, y: two_body_ode(t, y, pert, coef_pot, cb_radius.value, mu_cb.value, epoch_t, nu), ets[step], states[step], dt, f_prev)
            
    elif method == "3":
        

        # ***Corrección importante: Pasar argumentos a calculate_nu y two_body_ode_with_nu***
        t_eval = [epoch_et + t for t in ets]
        states, _ = nm.lsoda_solver(lambda t, y: two_body_ode_with_nu(t, y, pert, coef_pot, cb_radius.value, mu_cb.value, epoch_et, ecb_list, body_params), states, t_eval[0], t_eval[-1], orbit_params["dt"]) #Se usa states[0] como valor inicial y se pasa dt a lsoda_solver
        
        for step in range(len(ets)):
            epoch_t = epoch_et + ets[step]
            for i, ecb in enumerate(ecb_list):
                eclipse_statuses[step, i], _ = es.eclipse(ecb, body_params, t_eval[step], states[step])
    elif method == "4":
        
        for i, ecb in enumerate(ecb_list):
            eclipse_statuses[step, i], nu = es.eclipse(ecb, body_params, ets[step], states[step]) #calculus of the eclipse status and the shadow function
            
        
        states, ets = nm.zvode_solver(lambda t, y: two_body_ode(t, y, pert, coef_pot, cb_radius.value, mu_cb.value, epoch_t), states, ets[0], ets[-1], dt)
    elif method == "5":
        
        for i, ecb in enumerate(ecb_list):
            eclipse_statuses[step, i], _ = es.eclipse(ecb, body_params, ets[step], states[step])
            
        states, ets = nm.RK45_solver(lambda t, y: two_body_ode(t, y, pert, coef_pot, cb_radius.value, mu_cb.value, epoch_t), states, ets[0], ets[-1], dt)
    else:
        print("Incorrect value")
    return states, ets, np.array(eclipse_statuses)


if __name__ == '__main__':
    
    # Read Input
    body_params, orbit_params, perturbation_params, spacecraft_params = rd.read_json("Input.json")
    #Read egm96 coeffs if needed
    Non_spherical_body_bool=perturbation_params["perturbations"]["Non_spherical_body"][0].get("value", False)
    egm96_model_bool = perturbation_params["perturbations"]["Non_spherical_body"][0].get("EGM96_model", False)
    
    if egm96_model_bool and Non_spherical_body_bool:
        coeffs = rd.read_EGM96_coeffs('data/egm96_to360.ascii').astype(float)
        max_order = int(input("Enter the maximum degree for the EGM96 coefficients. \n"))
    #mu:
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    mu = (G * body_params['mass']).value
    
    #init trajectories:
    trajectories = []
    states_kepl_all = []
    
    #e, i vectors:
    e_vec=[0, 0]
    i_vec=[0,0]
    i_vec_all=[]
    e_vec_all=[]    
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
                       4- Vode (Variable-order, Adams-Bashforth-Moulton)
                       5- Runge-Kutta 4/5th order\n""")
        start_time = time.time()  # Execution time start
        # Propagation of the orbit
        states, ets, eclipse_statuses = solve_orbit(method, steps, ets, states, orbit_params, perturbation_params, body_params)
        end_time = time.time()
        # Execution time:
        execution_time = end_time - start_time
        print(f"Execution finished in {execution_time:.3f}s. Generating state vectors files and plots.")
        # Save trajectories:
        trajectories.append(states[:, :3])
        
        # Conversion to Keplerian elements
        states_kepl = np.apply_along_axis(cc.cart_2_kep, 1, states, mu)
        states_kepl_all.append(states_kepl)
        
        #Inicialization of i_vec y e_vec para la orbita:
        i_vec_orbit=[]
        e_vec_orbit=[]
        for state_kepl in states_kepl: #Iteramos sobre las filas del array
            i_vec = cc.i_vector(np.deg2rad(state_kepl[2]), np.deg2rad(state_kepl[3])) #input: i, raan in RADIANS
            i_vec_orbit.append(i_vec) #Append to the all orbits list
            e_vec = cc.e_vector(state_kepl[1], np.deg2rad(state_kepl[3]), np.deg2rad(state_kepl[4]))#input: e, raan, aop in radians
            e_vec_orbit.append(e_vec)
        i_vec_all.append(i_vec_orbit)
        e_vec_all.append(e_vec_orbit)
        #states_kepl_all = np.array(states_kepl_all)
        #Definition of the i and e vectors.
        

        # Save files
        df_states = pd.DataFrame(states, columns=['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz'])
        df_states.insert(0, 'Time', ets)
        for i, ecb in enumerate(body_params['ecb']):
            df_states.insert(len(df_states.columns), f'Eclipse_status_{ecb}', eclipse_statuses[:, i]) #Adds a column for each eclipsing body.
        df_states.to_csv(f'output/State_vectors_orbit_{idx+1}.csv', index_label='Step')

    # Plotting 3D orbit

    fig_3d, ax_3d = po.setup_3d_plot()

    po.plot_3D(ax_3d, trajectories, body_params['radius'], orbit_params['colors'], labels, df_states, body_params['ecb'])

    # Plotting the COEs

    fig_coes, axs_coes = po.setup_coes_plots(len(orbit_params['initial_states']))
    po.plot_coes(axs_coes, [ets] * len(orbit_params['initial_states']), states_kepl_all, orbit_params['colors'], labels, ets[-1], df_states, body_params['ecb'])


    #Plotting e and i vector:
    fig, axes = po.setup_vector_plot()
    po.plot_i_and_e_vectors(fig, axes, i_vec_all, e_vec_all, orbit_params) 

