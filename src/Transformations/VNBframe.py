# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:10:14 2025

@author: ddiaz.beca
"""
import numpy as np


def M_J2000toVNB_calculation(ref_state):
    """
    Calculates the rotation matrix to transform vectors from J2000 to VNB.

    Args:
        ref_state (numpy.ndarray): Reference state vector [x, y, z, vx, vy, vz].

    Returns:
        numpy.ndarray: Rotation matrix from J2000 to VNB.
    """
    r_ref_J2000 = ref_state[:3]  # Reference position vector
    v_ref_J2000 = ref_state[3:]  # Reference velocity vector

    # 1. Tangential Unit Vector (V) - In the direction of the orbital velocity
    tangential_axis_V = v_ref_J2000 / np.linalg.norm(v_ref_J2000)
    V_axis_unit = tangential_axis_V / np.linalg.norm(tangential_axis_V)
    # 2. Normal Unit vector (N) - Orthogonal to the orbital plane (direction of angular momentum)
    N_axis = np.cross(r_ref_J2000, v_ref_J2000)
    N_axis_unit = N_axis / np.linalg.norm(N_axis)
    # 3. Binormal unit vector (B) - Completes the right-handed system (VxN)
    #normal_axis_B = np.cross(tangential_axis_V, N_axis_unit)
    
    #B_axis_unit = normal_axis_B / np.linalg.norm(normal_axis_B)
    B_axis_unit=r_ref_J2000/np.linalg.norm(r_ref_J2000)
    # Build the rotation matrix. Each column is an VNB unit vector expressed in J2000 in V, N, B order.
    M_J2000toVNB = np.array([V_axis_unit, N_axis_unit, B_axis_unit])

    return M_J2000toVNB


def proyect_vectortoVNB(state_vector, M_J2000toVNB):
    """
    Projects a state vector from J2000 to VNB.

    Args:
        state_vector (numpy.ndarray): State vector in J2000 [x, y, z, vx, vy, vz].
        M_J2000toVNB (numpy.ndarray): Rotation matrix from J2000 to VNB.

    Returns:
        tuple: Position and velocity vectors in VNB.
    """
    r_J2000 = state_vector[:3]
    v_J2000 = state_vector[3:]

    r_VNB = M_J2000toVNB @ r_J2000
    v_VNB = M_J2000toVNB @ v_J2000

    return r_VNB, v_VNB


def project_VNBtoInertial(state_vector, M_VNBtoJ2000):
    """
    Projects a state vector from VNB to J2000.

    Args:
        state_vector (numpy.ndarray): State vector in VNB [x, y, z, vx, vy, vz].
        M_VNBtoJ2000 (numpy.ndarray): Rotation matrix from VNB to J2000.

    Returns:
        tuple: Position and velocity vectors in J2000.
    """
    r_VNB = state_vector[:3]
    v_VNB = state_vector[3:]

    r_J2000 = M_VNBtoJ2000 @ r_VNB
    v_J2000 = M_VNBtoJ2000 @ v_VNB

    return r_J2000, v_J2000
