#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for analysing outputs from simulations

Created on Wed July 28th 2021

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt

from .constants import *



# below copied over from 
# /Users/Cary/Box Sync/My Documents/Experiments/2018 I Linear BW/analysis/Morton Simulations/2020 Morton Simulations/20210726_Morton_timing_viewer_v3.py

def gauss(x, mu, sigma, A):
    sigma = np.abs(sigma)
    A = np.abs(A)
    index = (x-mu)**2/(2.0*sigma**2)
    return A * np.exp(-index)


from scipy.special import erf

def cum_dist(z):
    return 0.5 * (1.0 + erf(z / 2**0.5) )

def gauss_skew(x, mu, sigma, A, alpha):
    G = gauss(x, mu, sigma, A)
    C = cum_dist(alpha * x)
    return 2.0 * G * C

def find_FWHM(x,y,frac=0.5, offset=0.1):
    """Brute force FWHM calculation.
    Frac allows you to easily change, so to e-2 value etc.
    
    """
    fm = y.copy() - y.min()
    fm = fm.max()
    hm = fm * frac

    hm += y.min()
    fm = fm + y.min()
    max_idx = np.argmax(y)
    
    first_half = np.arange( int((1.0 - offset) * max_idx) )
    second_half = np.arange( int((1.0 + offset) * max_idx), x.size )
    
    hm_x_left = np.interp(hm, y[first_half], x[first_half])
    hm_x_right = np.interp(hm, y[second_half][::-1], x[second_half][::-1])
    
    return hm_x_right - hm_x_left