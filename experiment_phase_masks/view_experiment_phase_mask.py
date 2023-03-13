#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:55:49 2021 by Cary Colgan. 
Email: cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt

filename = './2018_BW/Gemini 80x200um.CSV'

x = np.loadtxt(filename, delimiter=',', dtype=int)

full_width =  177.8 # [mm] Full-width / full-height of phase-mask
half_width = hw = full_width/2.0 # [mm] Half-width


plt.figure()
plt.imshow(x, origin='lower', extent=[-hw, +hw, -hw, +hw])

gemini_beam_radius = r = 150.0/2.0 # [mm]
theta = np.linspace(0.0, 2.0*np.pi, 100)
x,y = r*np.cos(theta), r*np.sin(theta)
plt.plot(x, y, color='w', linewidth=3)

plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.show()