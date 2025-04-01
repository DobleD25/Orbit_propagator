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
import spiceypy as spice
import coord_conversion as cc
import planetary_data as pd


def read_json(input_file):
    """
    Reads simulation parameters from a JSON input file.

    Args:
        input_file (str): Path to the JSON input file.

    Returns:
        tuple: A tuple containing body parameters, orbit parameters, perturbation parameters, and spacecraft parameters.
    """
    np.set_printoptions(precision=3, suppress=True)

    # Read input file:
    with open(input_file, "r") as file:
        input = json.load(file)

    # Read parameters of each mission
    orbits = input["Missions"]
    initial_states = []
    colors = []
    maneuver_params_lists = []
    #  Central body parameters
    central_body = input["Central_body"][
        0
    ]  
    body_params = {
        "name": central_body["name"],
        "radius": central_body["radius"] * u.km,
        "mass": central_body["mass"] * u.kg,
        "ecb": central_body.get("eclipsing_bodies", []),
    }
    G = const.G.to(u.km**3 / (u.kg * u.s**2))
    mu = getattr(pd, central_body["name"].lower())["mu"] 
    mu_value = mu
    

    # Print central body parameters
    print(
        f"""
          Central Body Parameters:
              ------------------------
              Name            : {body_params['name']}
              Radius          : {body_params['radius']:.4e}
              Mass            : {body_params['mass']:.4e}
              Mu              : {mu:.4e} 
              eclipsing bodies: {body_params['ecb']}
              """
    )

    # Spacecraft parameters
    spacecraft_data = input.get("Spacecraft", [])  # Handle missing 'Spacecraft' key
    spacecraft_params = []
    for spacecraft in spacecraft_data:
        spacecraft_params.append(
            {
                "name": spacecraft["name"],
                "mass": spacecraft["mass"] * u.kg,  # Add units
                "area": spacecraft["area"]
                * u.m**2,  # Add units and convert area to m^2
                "Cr": spacecraft["Cr"],
            }
        )

        # Print spacecraft parameters - similar to central body print
        print(
            f"""
           Spacecraft Parameters:
              -----------------------
              Name   : {spacecraft['name']}
              Mass   : {spacecraft['mass']:.4e}
              Area   : {spacecraft['area']:.4e}
              Cr     : {spacecraft['Cr']}
              """
        )
    for orbit in orbits:
        coord_sys_orbit = orbit["system"].lower()
        print(f"Coordinate system: {coord_sys_orbit}")
        color = orbit["color"]
        init_epoch = orbit["init_epoch"]

        colors.append(color)
        if coord_sys_orbit == "cartesians":
            coords_cart = orbit["Cartesian_coordinates"][0]
            x, y, z = coords_cart["x"], coords_cart["y"], coords_cart["z"]
            vx, vy, vz = coords_cart["vx"], coords_cart["vy"], coords_cart["vz"]
            initial_state_cart = np.array([x, y, z, vx, vy, vz])
            initial_states.append(initial_state_cart)
            initial_state_kepl = cc.cart_2_kep(initial_state_cart, mu_value)
            print(
                f"Initial state vector (cartesians): {np.round(initial_state_cart, 3)}"
            )
            print(
                f"Initial state vector (keplerians): {np.round(initial_state_kepl, 3)}"
            )

        elif coord_sys_orbit == "keplerians":
            coords_kepl = orbit["Keplerian_coordinates"][0]
            a, e, i, Omega_AN, omega_PER, nu = coords_kepl.values()
            initial_state_kepl = np.array([a, e, i, Omega_AN, omega_PER, nu])
            initial_state_cart = cc.kep_2_cart(initial_state_kepl, mu_value)
            initial_states.append(initial_state_cart)
            print(
                f"Initial state vector (cartesians): {np.round(initial_state_cart, 3)}"
            )
            print(
                f"Initial state vector (keplerians): {np.round(initial_state_kepl, 3)}"
            )
        else:
            raise ValueError("Invalid coordinate system specified")
            # Leer maniobras para cada órbita:
        maneuver_data = orbit.get("Maneouvers", [])
        maneuver_params = []
        for maneuver_group in maneuver_data:
            chemical_maneuvers = maneuver_group.get("Chemical", [])
            electrical_maneuvers = maneuver_group.get("Electrical", [])

            for chemical_maneuver in chemical_maneuvers:
                maneuver_params.append(
                    {
                        "type": "Chemical",
                        "epoch": chemical_maneuver.get("Epoch", None),
                        "delta_v": chemical_maneuver.get("DeltaV (VNB)", None),
                    }
                )

            for electrical_maneuver in electrical_maneuvers:
                maneuver_params.append(
                    {
                        "type": "Electrical",
                        "epoch": electrical_maneuver.get("Epoch", None),
                        "input_sel": electrical_maneuver.get(
                            "Trust and Time (TT) or Thrust and DeltaV (TD) or DeltaV and Time (DT)",
                            None,
                        ),
                        "thrust": electrical_maneuver.get("Thrust(VNB)", None),
                        "duration": electrical_maneuver.get("Time", None),
                        "delta_v": electrical_maneuver.get("DeltaV (VNB)", None),
                    }
                )

        maneuver_params_lists.append(
            maneuver_params
        )  # Añadir maniobras de esta órbita a la lista de listas

    dt = orbits[0]["step_size"]
    tspan = orbits[0]["time_span"]

    # Read perturbations
    perturbations = input.get("Perturbations", [{}])[0]
    Geopotential_bool = perturbations.get("Non_spherical_body", [{}])[0].get(
        "value", False
    )
    N_body_bool = perturbations.get("N-body", [{}])[0].get("value", False)
    SRP_bool = perturbations.get("SRP", [{}])[0].get(
        "value", False
    )  

    coefficients = perturbations["Non_spherical_body"][0].get("coefficients", [{}])[0]
    if Geopotential_bool:

        EGM96_model = perturbations["Non_spherical_body"][0].get("EGM96_model")

        coef_pot = {k: coefficients.get(k, 0) for k in ["J2", "J3", "C22", "S22"]}
    if N_body_bool:
        n_body_list = perturbations.get("N-body", [{}])[0].get("list", [])

    if SRP_bool:
        ecb_list = perturbations.get("Central_body", [{}])[0].get(
            "eclipsing_bodies", []
        )
        SRP_model=perturbations.get("SRP", [{}])[0].get("Model (Cannonball/Realistic)", [])
    # dictionary of parameters:
    orbit_params = {
        "coord_sys_orbit": coord_sys_orbit,
        "initial_states": initial_states,
        "dt": dt,
        "tspan": tspan,
        "colors": colors,
        "epoch": init_epoch,
        "maneuver_params_list": maneuver_params_lists,  
    }
    perturbation_params = {"perturbations": perturbations}
    if Geopotential_bool:
        perturbation_params.update({"coef_pot": coef_pot, "EGM96_model": EGM96_model})
    else:
        perturbation_params.update({"coef_pot": "0", "EGM96_model": "0"})
    if "N-body" in perturbations:
        perturbation_params["N-body"] = perturbations["N-body"]
    if N_body_bool:
        perturbation_params.update({"body_list": n_body_list})
    if SRP_bool:
        perturbation_params["SRP"] = perturbations["SRP"]
        perturbation_params.update({"eclipsing_bodies": ecb_list})
        perturbation_params.update({"SRP_model": SRP_model})
    if "SRP" not in perturbation_params:
        perturbation_params["SRP"] = [{"value": False}]

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


def read_maneuvers(orbit_params):
    """
    Reads and processes maneuver parameters from orbit parameters.

    Args:
    orbit_params (dict): Dictionary containing orbit parameters, including maneuver parameters.

    Returns:
    dict: Updated orbit parameters with processed maneuver epochs.
    """
    # Init maneuver lists
    man_epoch_chem_lists = []
    man_epoch_elec_lists = []

    for maneuver_list in orbit_params[
        "maneuver_params_list"
    ]:  # Iterate about the lists for each mission
        man_epoch_chem_list = []
        man_epoch_elec_list = []
        for maneuver in maneuver_list:  # Iterate about each maneuvers list
            if maneuver["type"] == "Chemical":
                man_epoch_et = spice.str2et(maneuver["epoch"])
                delta_v = maneuver["delta_v"]
                man_epoch_chem_list.append((maneuver["epoch"], man_epoch_et, delta_v))
            elif maneuver["type"] == "Electrical":
                man_epoch_et = spice.str2et(maneuver["epoch"])
                input_sel = maneuver["input_sel"]
                thrust = maneuver["thrust"]
                duration = maneuver["duration"]
                delta_v_target = maneuver["delta_v"]
                man_epoch_elec_list.append(
                    (
                        maneuver["epoch"],
                        man_epoch_et,
                        input_sel,
                        thrust,
                        duration,
                        delta_v_target,
                    )
                )
        man_epoch_chem_lists.append(man_epoch_chem_list)
        man_epoch_elec_lists.append(man_epoch_elec_list)
    orbit_params["man_epoch_chem_lists"] = man_epoch_chem_lists
    orbit_params["man_epoch_elec_lists"] = man_epoch_elec_lists

    # Print chemical maneuvers
    for i, maneuver in enumerate(man_epoch_chem_list):
        print(
            f"Chemical maneuver {i+1}: epoch UT: {maneuver[0]}; Delta V: {maneuver[2]}"
        )

    # Print electric maneuvers
    for i, maneuver in enumerate(man_epoch_elec_list):
        print(f"Electrical maneuver {i+1}: epoch UT: {maneuver[0]};")
    return orbit_params
