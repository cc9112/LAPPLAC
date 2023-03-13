#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for calculating theoretical values from laserParameters

Created on Wed Mar  3 23:57:44 2021

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt

from .constants import *

def get_zr(laserParameters):
    w0 = laserParameters['w0']
    lambda0 = laserParameters['lambda0']
    return np.pi*w0**2/(lambda0)


def get_nCr(omega0):
    return m_e*epsilon_0*omega0**2/e**2		# Critical density


def make_inputBeam(X,Y,laserParameters):
    #Setup intensity profile
    x = X[0].copy()
    y = Y[:,0].copy()
    
    if laserParameters['shape']=='gaussian':
        r_sq = (X-laserParameters['xOffset'])**2 + (Y-laserParameters['yOffset'])**2
        r = r_sq**(0.5)
        
        if 'n' not in laserParameters:
            n = 2
        else:
            n = laserParameters['n']
        
        intensity = np.exp(-2 * (r/laserParameters['w_z0'])**n)
            
    elif laserParameters['shape']=='uniform':
        intensity = np.ones_like(X)

    	# Fix the intensity by scaling it appropriately 
    flux = np.trapz(np.trapz(intensity, x, axis=1), y) #previously dim=2, basically 2d integral
    inputBeam = np.sqrt(laserParameters['E0']/(flux*laserParameters['tau']) * intensity) # The field amplitude in sqrt(W/cm^2)

    return inputBeam

def get_w0(wavelength, f_number):
    """Assuming perfect flat-top profile so perfect Airy focus
    """
    return 1.22 * f_number * wavelength

def get_zR(w0, wavelength, f_number):
    """Assuming perfect flat-top profile so perfect Airy focus
    """
    return np.pi * w0**2 /  wavelength
    