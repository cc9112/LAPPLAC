#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:01:10 2021

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants
um		= 1e-6		# microns
mm 		= 1e-3		# millimeters
fs 		= 1e-15	# femtoseconds
re 		= 2.81794e-15 		# classical electron radius

from scipy.constants import epsilon_0, m_e, e, c