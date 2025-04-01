# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:01:04 2025

@author: ddiaz.beca
"""
from astropy.time import Time
import numpy as np
import spiceypy as spice


def spiceET2GHA(epoch_et_value):
    """
    Calculates the Greenwich Hour Angle (GHA) from a time value in SPICE Ephemeris Time (ET).

    Assumes that epoch_et_value is in seconds from J2000 TDB (SPICE ET format).

    Parameters:
    - epoch_et_value: Time value in SPICE Ephemeris Time (ET) (float).

    Returns:
    - gha_radians: Greenwich Hour Angle (GST) in radians.
    - gha_degrees: Greenwich Hour Angle (GST) in degrees.
    """

    # J2000.0 epoch in Julian Date (TDB) is 2451545.0
    j2000_jd_tdb = 2451545.0

    # Convert SPICE ET (seconds from J2000 TDB) to Julian Date (TDB)

    if isinstance(epoch_et_value, list) or isinstance(epoch_et_value, np.ndarray):
        # Convert SPICE ET (seconds from J2000 TDB) to Julian Date (TDB)
        jd_tdb = j2000_jd_tdb + np.array(epoch_et_value) / 86400.0
        # Create an astropy Time object with Julian Date in TDB scale
        time_tdb = Time(jd_tdb, format="jd", scale="tdb")
        # Calculate Greenwich Sidereal Time (GST) using UTC time
        #gst = time_tdb.sidereal_time("apparent", "greenwich")
        gst=Time.earth_rotation_angle(longitude="greenwich")
        # Get GST in radians and degrees
        gha_radians = gst.radian
        gha_degrees = gst.degree
    else:
        # Convert SPICE ET (seconds from J2000 TDB) to Julian Date (TDB)
        jd_tdb = j2000_jd_tdb + epoch_et_value / 86400.0
        # Create an astropy Time object with Julian Date in TDB scale
        time_tdb = Time(jd_tdb, format="jd", scale="tdb")
        # Calculate Greenwich Sidereal Time (GST) using UTC time
        #gst = time_tdb.sidereal_time("apparent", "greenwich")
        gst=time_tdb.earth_rotation_angle(longitude=0)
        # Get GST in radians and degrees
        gha_radians = gst.radian
        gha_degrees = gst.degree

    

    return gha_radians, gha_degrees
