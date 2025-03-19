# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:11:29 2025

@author: ddiaz.beca
"""
#from scipy.integrate import ode
import numpy as np
import ODE
import eclipse_status as es
import maneuvers as deltaV


def adams_predictor_corrector(f, t, y, h, f_prev):
    """
    Implements the Adams-Bashforth-Moulton predictor-corrector method.

    Args:
        f: The derivative function (ODE).
        t: The current time.
        y: The current state vector.
        h: The time step.
        f_prev: A list of previous function evaluations.

    Returns:
        The corrected state vector.
    """
    yp = y + (h / 24) * (
        55 * f_prev[0] - 59 * f_prev[1] + 37 * f_prev[2] - 9 * f_prev[3]
    )

    # Corrector de Adams-Moulton de tercer orden
    yc = y + (h / 24) * (9 * f(t + h, yp) + 19 * f_prev[0] - 5 * f_prev[1] + f_prev[2])

    return yc


def rk4_step(f, t, y, h):
    """
    Calculates one step of the Runge-Kutta 4th order method.

    Args:
        f: The derivative function (ODE).
        t: The current time.
        y: The current state vector.
        h: The time step.

    Returns:
        The updated state vector after one RK4 step.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * k1 * h)
    k3 = f(t + 0.5 * h, y + 0.5 * k2 * h)
    k4 = f(t + h, y + k3 * h)

    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk56_step(f, t, y, h):
    """
    Perform a single step of the Runge-Kutta 5(6) method.

    Parameters:
    f : function
        The function that defines the differential equation dy/dt = f(t, y).
    y : numpy array
        The current value of the dependent variable.
    t : float
        The current value of the independent variable.
    h : float
        The step size.

    Returns:
    y_next : numpy array
        The estimated value of the dependent variable at t + h.
    """
    k1 = h * f(t, y)
    k2 = h * f(t + h / 4, y + k1 / 4)
    k3 = h * f(t + h / 4, y + k1 / 8 + k2 / 8)
    k4 = h * f(t + h / 2, y - k2 / 2 + k3)
    k5 = h * f(t + 3 * h / 4, y + 3 * k1 / 16 + 9 * k4 / 16)
    k6 = h * f(
        t + h, y - 3 * k1 / 7 + 2 * k2 / 7 + 12 * k3 / 7 - 12 * k4 / 7 + 8 * k5 / 7
    )

    y_next = y + (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) / 90
    return y_next


def lsoda_solver(f, states, t0, t_end, dt):
    """
    Solves an ODE using the LSODA (Livermore Solver for Ordinary Differential Equations) method.

    Args:
        f: The derivative function (ODE).
        states: The initial state vector.
        t0: The initial time.
        t_end: The end time.
        dt: The time step.

    Returns:
        A tuple containing the solution (array of state vectors) and the time values.
    """
    solver = ode(f)
    solver.set_integrator("lsoda")
    solver.set_initial_value(states[0], t0)

    t = [t0]
    sol = [states[0]]

    while solver.successful() and solver.t < t_end:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        sol.append(solver.y)

    return np.array(sol), np.array(t)







def call_rk4(
    epoch_et,
    ets,
    steps,
    states,
    pert,
    coef_pot,
    cb_radius,
    mu_cb,
    ecb_list,
    body_params,
    man_epoch_chem_list,
    man_epoch_elec_list,
    deltaV_electric_maneuvers,
    orbit_params,
    coeffs,
    max_order,
    perturbation_params,
    spacecraft_params,
    default_dt,
    eclipse_statuses,
):
    """
    Performs orbit propagation using the Runge-Kutta 4th order method.

    Args:
        epoch_et: Initial epoch in ephemeris time.
        ets: Array of time steps.
        steps: Number of steps.
        states: Array of state vectors.
        pert: Perturbation flags.
        coef_pot: Potential coefficients.
        cb_radius: Central body radius.
        mu_cb: Gravitational parameter of the central body.
        ecb_list: List of eclipsing bodies.
        body_params: Dictionary of body parameters.
        man_epoch_chem_list: List of chemical maneuver epochs.
        man_epoch_elec_list: List of electrical maneuver epochs.
        deltaV_electric_maneuvers: Array of electric maneuver delta-V values.
        orbit_params: Dictionary of orbit parameters.
        coeffs: Potential model coefficients.
        max_order: Maximum order of the potential model.
        perturbation_params: Dictionary of perturbation parameters.
        spacecraft_params: Dictionary of spacecraft parameters.
        dt: Time step.
        eclipse_statuses: Array to store eclipse statuses.

    Returns:
        A tuple containing the propagated states, time values, and eclipse statuses.
    """
    t_eval = [epoch_et + t for t in ets]
    for step in range(steps - 1):
        rk4_dt_to_use = default_dt  # Inicializar con dt por defecto

        for maneuver in man_epoch_chem_list:
            maneuver_epoch = maneuver[1]
            t_current = t_eval[step]
            t_next_default_dt = t_eval[step] + default_dt

            print(
                f"Step: {step}, t_current: {t_current} dt: {default_dt}"
            )  # Debug print

            if t_current < maneuver_epoch < t_next_default_dt:
                print("Maneuver epoch WITHIN this step!")
                dt_adjusted = maneuver_epoch - t_current

                if dt_adjusted < 1e-9:
                    dt_adjusted = default_dt / 100
                    print(f"dt_adjusted too small, using reduced dt: {dt_adjusted}")
                else:
                    print(f"Adjusting dt to: {dt_adjusted} to hit maneuver epoch.")

                rk4_dt_to_use = dt_adjusted

                # Calculates the next state until the maneuver epoch
                state_maneuver = rk4_step(  # state just before the maneuver
                    lambda t, y: ODE.two_body_ode_with_f(
                        t,
                        y,
                        pert,
                        coef_pot,
                        cb_radius.value,
                        mu_cb,
                        ecb_list,
                        body_params,
                        man_epoch_chem_list,
                        man_epoch_elec_list,
                        deltaV_electric_maneuvers,
                        orbit_params,
                        coeffs,
                        max_order,
                        perturbation_params,
                        spacecraft_params,
                    ),
                    (
                        t_eval[step] 
                    ),
                    states[step],
                    rk4_dt_to_use
                    
                )
        
                

                # Apply maneuver in the propagated state at epoch maneuver_epoch
                print(
                    "Applying chemical maneuver!"
                )  # 
                dV_VNB = maneuver[2]
                deltaV_J2000 = (
                    deltaV.vector_J2000(state_maneuver, dV_VNB) / 1000
                )  
                v_post_man_J2000 = state_maneuver[3:6] + deltaV_J2000
                state_maneuver = np.array(
                    [  # Update the states with the DeltaV
                        state_maneuver[0],
                        state_maneuver[1],
                        state_maneuver[2],
                        v_post_man_J2000[0],
                        v_post_man_J2000[1],
                        v_post_man_J2000[2],
                    ]
                )
                
                rk4_dt_2=default_dt-rk4_dt_to_use
                states[step + 1] = rk4_step(  # state just before the maneuver
                    lambda t, y: ODE.two_body_ode_with_f(
                        t,
                        y,
                        pert,
                        coef_pot,
                        cb_radius.value,
                        mu_cb,
                        ecb_list,
                        body_params,
                        man_epoch_chem_list,
                        man_epoch_elec_list,
                        deltaV_electric_maneuvers,
                        orbit_params,
                        coeffs,
                        max_order,
                        perturbation_params,
                        spacecraft_params,
                    ),
                    (
                        maneuver_epoch
                    ),
                    state_maneuver,
                    rk4_dt_2,
                )
                t_eval[step + 1] = t_eval[step] + default_dt

                break

            else:
                # calculates steps without maneuver
                states[step + 1] = rk4_step(
                    lambda t, y: ODE.two_body_ode_with_f(
                        t,
                        y,
                        pert,
                        coef_pot,
                        cb_radius.value,
                        mu_cb,
                        ecb_list,
                        body_params,
                        man_epoch_chem_list,
                        man_epoch_elec_list,
                        deltaV_electric_maneuvers,
                        orbit_params,
                        coeffs,
                        max_order,
                        perturbation_params,
                        spacecraft_params,
                    ),
                    (
                        t_eval[step] if rk4_dt_to_use == default_dt else maneuver_epoch
                    ),  
                    states[step],
                    rk4_dt_to_use,  # Usar dt ajustado o default_dt
                )
                t_eval[step + 1] = t_eval[step] + rk4_dt_to_use

        

        print(f"State vector: {states[step + 1]}")
        for i, ecb in enumerate(ecb_list):  # Iterate over the eclipsing bodies
            eclipse_statuses[step, i], _ = es.eclipse(
                ecb, body_params, t_eval[step], states[step]
            )  # Calculate the eclipse status
    return states, t_eval, eclipse_statuses


def call_rk56(
    epoch_et,
    ets,
    steps,
    states,
    pert,
    coef_pot,
    cb_radius,
    mu_cb,
    ecb_list,
    body_params,
    man_epoch_chem_list,
    man_epoch_elec_list,
    deltaV_electric_maneuvers,
    orbit_params,
    coeffs,
    max_order,
    perturbation_params,
    spacecraft_params,
    default_dt,
    eclipse_statuses,
):
    """
    Performs orbit propagation using the Runge-Kutta 5(6)th order method.

    Args:
        epoch_et: Initial epoch in ephemeris time.
        ets: Array of time steps.
        steps: Number of steps.
        states: Array of state vectors.
        pert: Perturbation flags.
        coef_pot: Potential coefficients.
        cb_radius: Central body radius.
        mu_cb: Gravitational parameter of the central body.
        ecb_list: List of eclipsing bodies.
        body_params: Dictionary of body parameters.
        man_epoch_chem_list: List of chemical maneuver epochs.
        man_epoch_elec_list: List of electrical maneuver epochs.
        deltaV_electric_maneuvers: Array of electric maneuver delta-V values.
        orbit_params: Dictionary of orbit parameters.
        coeffs: Potential model coefficients.
        max_order: Maximum order of the potential model.
        perturbation_params: Dictionary of perturbation parameters.
        spacecraft_params: Dictionary of spacecraft parameters.
        dt: Time step.
        eclipse_statuses: Array to store eclipse statuses.

    Returns:
        A tuple containing the propagated states, time values, and eclipse statuses.
    """
    t_eval = [epoch_et + t for t in ets]
    for step in range(steps - 1):
        dt_to_use = default_dt  # Inicializar con dt por defecto

        for maneuver in man_epoch_chem_list:
            maneuver_epoch = maneuver[1]
            t_current = t_eval[step]
            t_next_default_dt = t_eval[step] + default_dt

            print(
                f"Step: {step}, t_current: {t_current} dt: {default_dt}"
            )  # Debug print

            if t_current < maneuver_epoch < t_next_default_dt:
                print("Maneuver epoch WITHIN this step!")
                dt_adjusted = maneuver_epoch - t_current

                if dt_adjusted < 1e-9:
                    dt_adjusted = default_dt / 100
                    print(f"dt_adjusted too small, using reduced dt: {dt_adjusted}")
                else:
                    print(f"Adjusting dt to: {dt_adjusted} to hit maneuver epoch.")

                dt_to_use = dt_adjusted

                # Calcular el siguiente estado *hasta* el epoch de la maniobra usando el dt ajustado
                states[step + 1] = (
                    rk56_step(  # Calcula el estado *antes* de la maniobra
                        lambda t, y: ODE.two_body_ode_with_f(
                            t,
                            y,
                            pert,
                            coef_pot,
                            cb_radius.value,
                            mu_cb,
                            ecb_list,
                            body_params,
                            man_epoch_chem_list,
                            man_epoch_elec_list,
                            deltaV_electric_maneuvers,
                            orbit_params,
                            coeffs,
                            max_order,
                            perturbation_params,
                            spacecraft_params,
                        ),
                        (
                            t_eval[step]
                            if dt_to_use == default_dt
                            else t_eval[step]
                        ),  # Usa t_eval[step] siempre como tiempo de inicio
                        states[step],
                        dt_to_use,
                    )
                )
                t_eval[step + 1] = (
                    maneuver_epoch  # Establece el siguiente tiempo al epoch de la maniobra
                )
                # Calculates the next state until the maneuver epoch
                state_maneuver = rk56_step(  # state just before the maneuver
                    lambda t, y: ODE.two_body_ode_with_f(
                        t,
                        y,
                        pert,
                        coef_pot,
                        cb_radius.value,
                        mu_cb,
                        ecb_list,
                        body_params,
                        man_epoch_chem_list,
                        man_epoch_elec_list,
                        deltaV_electric_maneuvers,
                        orbit_params,
                        coeffs,
                        max_order,
                        perturbation_params,
                        spacecraft_params,
                    ),
                    (
                        t_eval[step] 
                    ),
                    states[step],
                    dt_to_use
                    
                )
                # Apply maneuver in the propagated state at epoch maneuver_epoch
                print(
                    "Applying chemical maneuver!"
                )  # 
                dV_VNB = maneuver[2]
                deltaV_J2000 = (
                    deltaV.vector_J2000(state_maneuver, dV_VNB) / 1000
                )  
                v_post_man_J2000 = state_maneuver[3:6] + deltaV_J2000
                state_maneuver = np.array(
                    [  # Update the states with the DeltaV
                        state_maneuver[0],
                        state_maneuver[1],
                        state_maneuver[2],
                        v_post_man_J2000[0],
                        v_post_man_J2000[1],
                        v_post_man_J2000[2],
                    ]
                )
                
                dt_2=default_dt-dt_to_use
                states[step + 1] = rk56_step(  # state just before the maneuver
                    lambda t, y: ODE.two_body_ode_with_f(
                        t,
                        y,
                        pert,
                        coef_pot,
                        cb_radius.value,
                        mu_cb,
                        ecb_list,
                        body_params,
                        man_epoch_chem_list,
                        man_epoch_elec_list,
                        deltaV_electric_maneuvers,
                        orbit_params,
                        coeffs,
                        max_order,
                        perturbation_params,
                        spacecraft_params,
                    ),
                    (
                        maneuver_epoch
                    ),
                    state_maneuver,
                    dt_2,
                )
                t_eval[step + 1] = t_eval[step] + default_dt

                break

            else:
                # calculates steps without maneuver
                states[step + 1] = rk56_step(
                    lambda t, y: ODE.two_body_ode_with_f(
                        t,
                        y,
                        pert,
                        coef_pot,
                        cb_radius.value,
                        mu_cb,
                        ecb_list,
                        body_params,
                        man_epoch_chem_list,
                        man_epoch_elec_list,
                        deltaV_electric_maneuvers,
                        orbit_params,
                        coeffs,
                        max_order,
                        perturbation_params,
                        spacecraft_params,
                    ),
                    (
                        t_eval[step] if dt_to_use == default_dt else maneuver_epoch
                    ),  
                    states[step],
                    dt_to_use,  # Usar dt ajustado o default_dt
                )
                t_eval[step + 1] = t_eval[step] + dt_to_use

        

        print(f"State vector: {states[step + 1]}")
        for i, ecb in enumerate(ecb_list):  # Iterate over the eclipsing bodies
            eclipse_statuses[step, i], _ = es.eclipse(
                ecb, body_params, t_eval[step], states[step]
            )  # Calculate the eclipse status
    return states, t_eval, eclipse_statuses



"""
def call_adamsBM(
    epoch_et,
    ets,
    steps,
    states,
    pert,
    coef_pot,
    cb_radius,
    mu_cb,
    ecb_list,
    body_params,
    man_epoch_chem_list,
    man_epoch_elec_list,
    deltaV_electric_maneuvers,
    orbit_params,
    coeffs,
    max_order,
    perturbation_params,
    spacecraft_params,
    dt,
    eclipse_statuses,
    f_prev,
):
    
    Performs orbit propagation using the Adams-Bashforth-Moulton predictor-corrector method.
    The first 3 steps are calculated with RK4, as Adams-BM needs initial values to start.

    Args:
        epoch_et (float): Initial epoch in ephemeris time.
        ets (list): Array of time steps.
        steps (int): Number of steps.
        states (numpy.ndarray): Array of state vectors.
        pert (list): Perturbation flags.
        coef_pot (numpy.ndarray): Potential coefficients.
        cb_radius (astropy.units.quantity.Quantity): Central body radius.
        mu_cb (float): Gravitational parameter of the central body.
        ecb_list (list): List of eclipsing bodies.
        body_params (dict): Dictionary of body parameters.
        man_epoch_chem_list (list): List of chemical maneuver epochs.
        man_epoch_elec_list (list): List of electrical maneuver epochs.
        deltaV_electric_maneuvers (numpy.ndarray): Array of electric maneuver delta-V values.
        orbit_params (dict): Dictionary of orbit parameters.
        coeffs (numpy.ndarray): Potential model coefficients.
        max_order (int): Maximum order of the potential model.
        perturbation_params (dict): Dictionary of perturbation parameters.
        spacecraft_params (dict): Dictionary of spacecraft parameters.
        dt (float): Time step.
        eclipse_statuses (numpy.ndarray): Array to store eclipse statuses.
        f_prev (list): A list of previous function evaluations.

    Returns:
        tuple: A tuple containing the propagated states, time values, and eclipse statuses.
    
    t_eval = [epoch_et + t for t in ets]
    for step in range(steps - 1):

        for i, ecb in enumerate(ecb_list):  # Iterate over the eclipsing bodies
            eclipse_statuses[step, i], nu = es.eclipse(
                ecb, body_params, t_eval[step], states[step]
            )  # calculus of the eclipse status and the shadow function

        if step < 3:
            # Use RK4 for the first three steps to get initial values
            states[step + 1] = rk4_step(
                lambda t, y: ODE.two_body_ode_with_f(
                    t_eval[step],
                    states[step],
                    pert,
                    coef_pot,
                    cb_radius.value,
                    mu_cb,
                    ecb_list,
                    body_params,
                    man_epoch_chem_list,
                    man_epoch_elec_list,
                ),
                t_eval[step],
                states[step],
                dt,
            )
            f_prev[step] = ODE.two_body_ode_with_nu(
                t_eval[step],
                states[step],
                pert,
                coef_pot,
                cb_radius.value,
                mu_cb,
                ecb_list,
                body_params,
                man_epoch_chem_list,
                man_epoch_elec_list,
            )
        else:
            f_prev[3] = f_prev[2]
            f_prev[2] = f_prev[1]
            f_prev[1] = f_prev[0]
            f_prev[0] = ODE.two_body_ode_with_nu(
                t_eval[step],
                states[step],
                pert,
                coef_pot,
                cb_radius.value,
                mu_cb,
                ecb_list,
                body_params,
                man_epoch_chem_list,
                man_epoch_elec_list,
            )
            states[step + 1] = adams_predictor_corrector(
                lambda t, y: ODE.two_body_ode_with_nu(
                    t,
                    y,
                    pert,
                    coef_pot,
                    cb_radius.value,
                    mu_cb,
                    ecb_list,
                    body_params,
                    man_epoch_chem_list,
                    man_epoch_elec_list,
                ),
                (
                    t_eval[step]
                    if dt_to_use == default_dt
                    else t_eval[step]
                ),  # Usa t_eval[step] siempre como tiempo de inicio
                states[step],
                rk4_dt_to_use,
            )
        t_eval[step + 1] = (
            maneuver_epoch  # Establece el siguiente tiempo al epoch de la maniobra
        )

        # Aplicar la maniobra química *DESPUÉS* de que rk4_step haya propagado hasta (o cerca de) maneuver_epoch
        print(
            "Applying chemical maneuver!"
        )  # Aplica la maniobra al estado *propagado*
        dV_VNB = maneuver[2]
        deltaV_J2000 = (
            deltaV.vector_J2000(states[step + 1], dV_VNB) / 1000
        )  # Aplica ahora a states[step+1]
        print(deltaV_J2000)
        v_post_man_J2000 = states[step + 1][3:6] + deltaV_J2000
        print(v_post_man_J2000)
        states[step + 1] = np.array(
            [  # Actualiza states[step+1] con el estado maniobrado
                states[step + 1][0],
                states[step + 1][1],
                states[step + 1][2],
                v_post_man_J2000[0],
                v_post_man_J2000[1],
                v_post_man_J2000[2],
            ]
        )
        print(states[step + 1])
        break

    else:
        # Calcular el siguiente estado *después* de (posible) ajuste de dt y maniobra
        states[step + 1] = rk56_step(
            lambda t, y: ODE.two_body_ode_with_f(
                t,
                y,
                pert,
                coef_pot,
                cb_radius.value,
                mu_cb,
                ecb_list,
                body_params,
                man_epoch_chem_list,
                man_epoch_elec_list,
                deltaV_electric_maneuvers,
                orbit_params,
                coeffs,
                max_order,
                perturbation_params,
                spacecraft_params,
            ),
            (
                t_eval[step] if rk4_dt_to_use == default_dt else maneuver_epoch
            ),  # Usar t_eval[step] si dt no ajustado, epoch maniobra si ajustado
            states[step],
            rk4_dt_to_use,  # Usar dt ajustado o default_dt
        )
        t_eval[step + 1] = t_eval[step] + rk4_dt_to_use

# Electrical maneuvers

print(f"State vector: {states[step + 1]}")
for i, ecb in enumerate(ecb_list):  # Iterate over the eclipsing bodies
    eclipse_statuses[step, i], _ = es.eclipse(
        ecb, body_params, t_eval[step], states[step]
    )  # Calculate the eclipse status
                
    return states, t_eval, eclipse_statuses
"""