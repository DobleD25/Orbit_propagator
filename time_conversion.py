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
    Calcula el Greenwich Hour Angle (GHA) a partir de un valor de tiempo en SPICE Ephemeris Time (ET).

    Asume que epoch_et_value está en segundos desde J2000 TDB (formato SPICE ET).

    Parámetros:
    - epoch_et_value: Valor de tiempo en SPICE Ephemeris Time (ET) (float).

    Retorna:
    - gha_radians: Greenwich Hour Angle (GST) en radianes.
    - gha_degrees: Greenwich Hour Angle (GST) en grados.
    """
    # J2000.0 epoch en Julian Date (TDB) es 2451545.0
    j2000_jd_tdb = 2451545.0

    # Convertir SPICE ET (segundos desde J2000 TDB) a Julian Date (TDB)
    jd_tdb = j2000_jd_tdb + epoch_et_value / 86400.0  # 86400 segundos en un día

    # Crear un objeto Time de astropy con Julian Date en escala TDB
    tiempo_tdb = Time(jd_tdb, format='jd', scale='tdb')

    # Convertir el tiempo TDB a UTC para calcular el tiempo sideral de Greenwich.
    # Aunque estrictamente el Tiempo Sideral se define mejor con UT1, para muchas aplicaciones
    # la diferencia entre UTC y UT1 es pequeña, y usar UTC para obtener GST es común.
    # Si se requiere máxima precisión, se debería usar UT1 y considerar la diferencia TDB-UT1.
    tiempo_utc = tiempo_tdb.utc  # Convertir a UTC para cálculo de GST (simplificación común)


    # Calcular Greenwich Sidereal Time (GST) usando el tiempo UTC (o TDB directamente, ver nota abajo)
    gst = tiempo_tdb.sidereal_time('apparent', 'greenwich')


    # Obtener GST en radianes y grados
    gha_radians = gst.radian
    gha_degrees = gst.degree

    return gha_radians, gha_degrees