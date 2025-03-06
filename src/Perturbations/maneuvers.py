# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:08:21 2025

@author: ddiaz.beca
"""

import VNBframe as VNB


def electric_maneouver_time(state, Thrust, m):
    """
    Calculates the acceleration due to a given thrust in a given time.

    Args:
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        Thrust (numpy.ndarray): Thrust force. in VNB spacecraft frame
        m (float): Mass of the spacecraft.

    Returns:
        float: Acceleration vector in the VNB spacecraft frame
    """
    a_prop = Thrust / m

    return a_prop.value


def vector_J2000(state, vector_VNB):
    """
    Transforms a vector from the VNB frame to the J2000 inertial frame.
    The primary axis lies in the orbital plane, normal to the velocity vector. The T axis is tangential to the orbit, and the W axis is normal to the orbital plane.
    Args:
        state (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        vector_VNB (numpy.ndarray): Vector in the VNB frame.

    Returns:
        numpy.ndarray: Vector in the J2000 frame.
    """
    M_J2000toVNB = VNB.M_J2000toVNB_calculation(state)
    M_VNBtoJ2000 = M_J2000toVNB.T

    vector_J2000 = M_VNBtoJ2000 @ vector_VNB

    return vector_J2000


def electric_maneouver_DT(deltaV, time):

    a_prop = deltaV / time

    return a_prop
