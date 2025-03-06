# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:31:23 2025

@author: ddiaz.beca
"""
import numpy as np


def difference(ref_state, state_prop):
    """
    Calcula el vector diferencia entre dos vectores de estado en J2000.

    Args:
      ref_state: Vector de estado de referencia en J2000 [posición_x, posición_y, posición_z, velocidad_x, velocidad_y, velocidad_z].
      state_prop: Vector de estado del propagador en J2000 [posición_x, posición_y, posición_z, velocidad_x, velocidad_y, velocidad_z].

    Returns:
      Vector diferencia en J2000 [delta_posición_x, delta_posición_y, delta_posición_z, delta_velocidad_x, delta_velocidad_y, delta_velocidad_z].
    """
    ref_state = np.array(ref_state)
    state_prop = np.array(state_prop)

    difference_vector = ref_state - state_prop
    return difference_vector


def M_J2000toNTW_calculation(ref_state):
    """
    Calculates the rotation matrix to transform vectors from J2000 to NTW.

    Args:
        ref_state (numpy.ndarray): Reference state vector [x, y, z, vx, vy, vz].

    Returns:
        numpy.ndarray: Rotation matrix from J2000 to NTW.
    """
    r_ref_J2000 = ref_state[:3]  # Reference position vector
    v_ref_J2000 = ref_state[3:]  # Reference velocity vector

    # 1. Tangential Unit Vector (T̂) - In the direction of the orbital velocity
    tangential_axis_T = v_ref_J2000 / np.linalg.norm(v_ref_J2000)

    # 2. Cross-track Unit Vector (Ŵ) - Orthogonal to the orbital plane (direction of angular momentum)
    W_axis = np.cross(r_ref_J2000, v_ref_J2000)
    W_axis_unit = W_axis / np.linalg.norm(W_axis)

    # 3. Normal Unit Vector (N̂) - Completes the right-handed system (T̂ x Ŵ)
    normal_axis_N = np.cross(tangential_axis_T, W_axis_unit)
    # Build the rotation matrix. Each column is an NTW unit vector expressed in J2000 in T, N, W order.
    M_J2000toNTW = np.array([tangential_axis_T, normal_axis_N, W_axis_unit]).T

    return M_J2000toNTW


def proyect_vectortoTNW(difference_vector, M_J2000toTNW):
    delta_r_J2000 = difference_vector[:3]
    delta_v_J2000 = difference_vector[3:]

    delta_r_TNW = M_J2000toTNW @ delta_r_J2000
    delta_v_TNW = M_J2000toTNW @ delta_v_J2000

    return delta_r_TNW, delta_v_TNW


def difference_TNW(ref_state, state_prop):
    """


    Parameters
    ----------
    ref_state : reference vector
    state_prop : vector to test

    Returns
    -------
    delta_r_TNW, delta_v_TNW : position and speed vector of the state vector to test with respect to the reference in the TNW frame

    """

    difference_vector = difference(ref_state, state_prop)  # difference vector
    M_J2000toTNW = M_J2000toNTW_calculation(
        ref_state
    )  # Calculation of the Rotation matrix J2000 to TNW
    # Position and speed in TNW
    delta_r_TNW, delta_v_TNW = proyect_vectortoTNW(difference_vector, M_J2000toTNW)

    return delta_r_TNW, delta_v_TNW
