# -*- coding: utf-8 -*-
"""
@author: ddiaz.beca
This script defines the ordinary differential equations (ODEs) used for orbit propagation,
including two-body motion and various perturbation forces.
"""

# 3rd party libraries
import numpy as np
import spiceypy as spice


# Own libraries
import coord_conversion as cc
import plot_orbit as po
import geopotential_1 as geop1
import n_methods as nm
import geopotential_model as gm
import n_body_per as nb
import SRP as srp
import eclipse_status as es
import planetary_data as planets
import read as rd
import maneuvers as man

"""
Load SPICE kernels
"""

# Leap second kernel
spice.furnsh("data/naif0012.tls.pc")

# solar system ephemeris kernel
spice.furnsh("data/de432s.bsp")


def two_body_ode_with_f(
    epoch,
    state,
    perts,
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
):  # ODE with the shadow funtion.
    """
    Two-body ODE with the shadow function (eclipse status).

    Input:
        epoch (float): Time in ephemeris time.
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        perts (dict): Dictionary with perturbation parameters.
        coef_pot (dict): Dictionary with potential coefficients.
        cb_radius (float): Radius of the central body.
        mu_cb (float): Gravitational parameter of the central body.
        ecb_list (list): List of eclipsing bodies.
        body_params (dict): Dictionary with body parameters.
        man_epoch_chem_list (list): List of chemical maneuver epochs.
        man_epoch_elec_list (list): List of electrical maneuver epochs.
        deltaV_electric_maneuvers (numpy.ndarray): Array to store delta-V for electrical maneuvers.
        orbit_params (dict): Dictionary with orbit parameters.
        coeffs (numpy.ndarray): Potential model coefficients.
        max_order (int): Maximum order of the potential model.
        perturbation_params (dict): Dictionary with perturbation parameters.
        spacecraft_params (dict): Dictionary with spacecraft parameters.

    Output:
        numpy.ndarray: Result of two_body_ode with the appropriate shadow function value.
    """

    # Shadow function:
    f = es.calculate_nu(state, ecb_list, body_params, epoch)

    return two_body_ode(
        epoch,
        state,
        perts,
        coef_pot,
        cb_radius,
        mu_cb,
        f,
        man_epoch_chem_list,
        man_epoch_elec_list,
        deltaV_electric_maneuvers,
        orbit_params,
        coeffs,
        max_order,
        perturbation_params,
        body_params,
        spacecraft_params,
    )


def two_body_ode(
    epoch_et,
    state,
    perts,
    coef_dict,
    cb_radius,
    mu_cb,
    f,
    man_epoch_chem_list,
    man_epoch_elec_list,
    deltaV_electric_maneuvers,
    orbit_params,
    coeffs,
    max_order,
    perturbation_params,
    body_params,
    spacecraft_params,
):
    """

    Newton's Universal Law of Gravitation with perturbation forces and maneuvers

    Input:
        epoch_et (float): Time in ephemeris time.
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        perts (dict): Dictionary with perturbation parameters.
        coef_dict (dict): Dictionary with potential coefficients.
        cb_radius (float): Radius of the central body.
        mu_cb (float): Gravitational parameter of the central body.
        f (float): Value of the shadow function (eclipse status).
        man_epoch_chem_list (list): List of chemical maneuver epochs.
        man_epoch_elec_list (list): List of electrical maneuver epochs.
        deltaV_electric_maneuvers (numpy.ndarray): Array to store delta-V for electrical maneuvers.
        orbit_params (dict): Dictionary with orbit parameters.
        coeffs (numpy.ndarray): Potential model coefficients.
        max_order (int): Maximum order of the potential model.
        perturbation_params (dict): Dictionary with perturbation parameters.
        body_params (dict): Dictionary with body parameters.
        spacecraft_params (dict): Dictionary with spacecraft parameters.

    Output:
        numpy.ndarray: Derivative of the state vector [vx, vy, vz, ax, ay, az].
    """

    r = state[:3]
    a = -(mu_cb / np.linalg.norm(r) ** 3.0) * r

    # Potential perturbations
    if (
        perturbation_params["perturbations"]["Non_spherical_body"][0]["value"] == True
        and perturbation_params["EGM96_model"] == False
    ):
        a_j2 = 0
        a_j3 = 0
        a_C22 = 0
        a_S22 = 0

        if "J2" in coef_dict:
            a_j2 = geop1.j2(r, coef_dict["J2"], mu_cb, cb_radius)
        if "J3" in coef_dict:
            a_j3 = geop1.j3(r, coef_dict["J3"], mu_cb, cb_radius)
        if "C22" in coef_dict:
            a_C22 = geop1.C22(r, coef_dict["C22"], mu_cb, cb_radius)
        if "S22" in coef_dict:
            a_S22 = geop1.S22(r, coef_dict["S22"], mu_cb, cb_radius)
        a += a_j2 + a_j3 + a_C22 + a_S22
    # Instead, with the harmonics from EGM96
    elif perturbation_params["EGM96_model"] == True:
        a_pert = gm.perturbation_potential_2(
            state, coeffs, mu_cb, cb_radius, max_order, epoch_et
        )
        a += a_pert
    # N-body perturbations
    if perturbation_params["N-body"][0]["value"]:

        a_pert = nb.n_body_a(perturbation_params, body_params, epoch_et, r)
        a += a_pert
    # Solar radiation pressure (SRP)
    if perturbation_params["SRP"][0]["value"] == True:
        if perturbation_params["SRP_model"]== "Cannonball":
        
            a_pert = srp.SRP_cannonball(body_params, spacecraft_params[0], epoch_et, state, f)
            a += a_pert
        if perturbation_params["SRP_model"]== "Realistic":
            a_pert = srp.SRP_realistic(body_params, spacecraft_params[0], epoch_et, state, f)
            a += a_pert
    # Chemical maneuvers
    for idx, maneuver in enumerate(man_epoch_elec_list):
        m = spacecraft_params[0]["mass"]

        if maneuver[2] == "TT":  # Input: Time and Thrust
            end_time = maneuver[1] + maneuver[4]
            thrust = maneuver[3]  # Thrust in VNB (Newton)
            if epoch_et >= maneuver[1] and epoch_et < end_time:

                thrust_J2000 = man.vector_J2000(state, thrust)

                a_prop = (
                    man.electric_maneouver_time(state, thrust_J2000, m) / 1000
                )  # km/s^2
                a += a_prop

            else:
                pass
        elif maneuver[2] == "TD":  # Input: Thrust and DeltaV
            thrust = maneuver[3]  # Thrust in VNB (Newton)
            needed_DeltaV = maneuver[5]
            needed_DeltaV = np.array(needed_DeltaV)
            thrust_J2000 = man.vector_J2000(state, thrust)

            print(needed_DeltaV, thrust_J2000)
            if (
                abs(deltaV_electric_maneuvers[idx, :]) <= abs(needed_DeltaV)
            ).all() and epoch_et >= maneuver[1]:
                delta_DeltaV = (
                    man.electric_maneouver_time(state, thrust, m) * orbit_params["dt"]
                )
                deltaV_electric_maneuvers[idx, :] += delta_DeltaV
                print(deltaV_electric_maneuvers[idx, :])
                # perturbation in J2000
                a_prop = (
                    man.electric_maneouver_time(state, thrust_J2000, m) / 1000
                )  # en km/s^2
                a += a_prop
            else:
                pass
        elif maneuver[2] == "DT":  # Input: DeltaV and Time
            end_time = maneuver[1] + maneuver[4]
            DeltaV = maneuver[5]
            DeltaV = np.array(DeltaV)  # We need to convert it to array
            Duration = maneuver[4]
            a_prop = man.electric_maneouver_DT(DeltaV, Duration) / 1000  # km/s^2
            if epoch_et >= maneuver[1] and epoch_et < end_time:
                # a in VNB:

                a_prop_J2000 = man.vector_J2000(state, a_prop)
                a += a_prop_J2000
    # print(f"State vector: {state}")
    dstate = np.array([state[3], state[4], state[5], a[0], a[1], a[2]])
    return dstate
