#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:35:39 2021

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import unwrap_phase

def phase(Field_slice):
    """Unwrapped phase of the beam from arg(E)
    
    Expects Field_slice to be a 2D array    
    
    Also needs to unwrap in z?
    """
    phase = np.angle(Field_slice, deg=False)
    return unwrap_phase(phase)


def shadow(Field_slice):
    """Intensity profile of the beam from |E|^2
    
    Expects Field_slice to be a 2D array    
    """
    return np.abs(Field_slice)**2