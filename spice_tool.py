# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:44:01 2025

@author: ddiaz.beca
"""

import spiceypy as spice


def n_body(central_body, third_body, time):

    
    et=spice.str2et(time)
    state, lt = spice.spkezr(third_body, et, "J2000", "NONE", central_body)
    
    return state


