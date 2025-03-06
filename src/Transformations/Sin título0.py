# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:43:13 2025

@author: ddiaz.beca
"""
import coord_conversion as cc
import numpy as np


def proyect_lat_lon(state_J2000, epoch):
    # position vector in J2000 cartesian
    r_J2000 = state_J2000[:3]
    # position vector in BodyFixed cartesian

    r_body_fixed_cart = cc.J2000_to_bodyfixed(r_J2000, epoch)

    # spherical coordinates in Body-Fixed

    r, lon, lat = cc.cartesian_to_spherical(r_body_fixed_cart)

    return np.array([r, lon, lat])
