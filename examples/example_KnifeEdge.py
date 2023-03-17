#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate propagtion of a typical Gaussian beam alignment laser that has been
perfectly obstructed down its centre by a knife edge.

The inital z and zR values require extra care here, to ensure their values 
match the manufacturer's information.

Created on Wed Jul 9th 20:43:59 2020

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import sys
if '../' not in sys.path: sys.path.append('../')   
from beamPropagator_source import *


#%%
# Setup Laser Parameters
# ----------------------

# Do it based on this Class 2 alignment laser
# Thorlabs CPS532-C2
# https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=1487

# 0.5 mrad divervence (full-angle)
# beam width 3.5 mm (full)
# Guassian profile

lambda0 = 532e-9 # [m]

# Guassian beam waist (radius at 1/e**2) size 
# w = w0 * sqrt(1 + (z/zR)**2 )
# in limit of large z, so beam diverging along constant angle
# w ~ w0 * z / zR

# Therefore, theta_half = (w/z) = w0/zR
theta_half = 0.25e-3 # [rad]

# single solution now for w0 from lambda0 and theta_half
w0 = lambda0 / (np.pi * theta_half) # [m]
# w0 = 0.677 mm

laserParameters = {
            'lambda0' : lambda0,
            'tau'    	: 1.0,	 	    # pulse width (s)
            'E0'     	: 1.0,		  	# pulse energy (J)
            'xOffset'	: 0.0,			# Spatial offset from center in x (m) - use at your own risk for the moment
            'yOffset'	: 0.0,			# Spatial offset from center in y (m) - use at your own risk for the moment
            'shape'  	: 'gaussian',
            'n'          : 2,                         # n=2 is gaussian, n>2 is supergaussian
            'w0'         : w0          #final waist size
            }

zR = get_zr(laserParameters) 
# zR = 2.71 mm

# solve for what z positon where waist is (3.5 / 2) = 1.75 mm
w_initial = 1.75e-3
z_initial = ((w_initial / w0)**2 - 1)**(0.5) * zR
# occurs at z_initial = 6.45 mm from focus


# Define the beam profile and generate the beam 
# taken from generateInputBeam.m
# returns [x,y,z,inputBeam] 

# want this to be enough into Far-Field to see pattern
zMax   = 2 * zR #(-3*zR to 3*zR)

z0 = z_initial  #relative to focal plane
Rz = z0 * (1.0 + (zR/z0)**2) #radius of curvature

w0 = laserParameters['w0']
w_z0 = w0 * np.sqrt(1.0 + (z0/zR)**2)
laserParameters['w_z0'] = w_z0
#%%
# Setup computational parameters
# ------------------------------
n_pts  = 512   		# number of pixels (must be even)
res    = 50*um   		# resolution in image plane (m)
nz     = 512    		# number of points in z (must be even)

mid = int(n_pts/2)
#%%
# Create calculation grid
# -----------------------
x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts ) #centering occurs here
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
X, Y = np.meshgrid(x,y)
z = np.linspace(0,zMax,nz)

inputBeam = make_inputBeam(X,Y,laserParameters)

# Knife-Edge Effect
# Cut beam in half. Bottom half blocked
inputBeam[mid:, :] = 0.0

# Setup phase profile
# Focusing beam
w_z0 = laserParameters['w_z0']
lambda0 = laserParameters['lambda0']

k = 2.0*np.pi/lambda0
R_sq = X**2 + Y**2
guoy = np.arctan(z0/zR)
inputPhase = k * R_sq/(2.0*Rz) - guoy

inputBeam = inputBeam * np.exp(-1j * inputPhase) 

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('inputBeam')

extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6

ax1.set_title('|E|$^{2}$ Waist')
t = shadow(inputBeam)**2
#t[t<t.max()*np.exp(-2)] = 0.0
ax1.imshow(t, extent=extent)

# should be waist size
theta = np.linspace(0.0, 2.0*np.pi, 100)
r = w_z0 * 1e6
x_circ,y_circ = r*np.cos(theta), r*np.sin(theta)
ax1.plot(x_circ,y_circ,'r')


ax2.set_title('Phase')
ax2.imshow(phase(inputBeam), extent=extent)


#%%
# Specify plasma
# None - propagate in vacuum
ne = np.zeros((n_pts, n_pts, nz))

#%%
#Setup first step and tools for propagation
# Propagate the beam through the plasma structure
# data.Field = beamPropagator(data); 

# Absorbing boundaries
# Reduces the effect of the edges of the domain
absorbingBoundaries = 1

eta0 = 1.0

eta = eta0 * np.ones((n_pts,n_pts,z.size))  #has to be for free-space
k0 = 2.0*np.pi/laserParameters['lambda0']

AS, xFreq, yFreq, G, mask, X, Y, z, eta = setup_propagate(inputBeam, X, Y, z, eta, k0, eta0, absorbingBoundaries)

#%%
#% MAIN ITERATION LOOP IN dz
Field = propagate_in_z(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k0)

#%%
# Interactive viewer of whole propagation of beam

buttons = explorer(Field, ne, z)

#%%
# COMPARE WITH GUASSIAN BEAM PROPAGATION THEORY
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

fig.suptitle('Free Space Propagation')

asp = (z.max() - z.min()) / (y.max()  - y.min())
ax1.imshow(shadow(Field[:, mid, :]), aspect=asp, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])

im = phase(Field[:, mid, :])
ax2.imshow(im, aspect=asp, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3],
           cmap=matplotlib.cm.RdBu_r, 
           norm=MidpointNormalize(midpoint=0.0,vmin=-np.pi/2.0, vmax=np.pi/2.0))


# not what we usually measure
# we think of a plane in z and ask whats the phase at this plane?
# then you get a parabola if focussing
Zd,Yd = np.meshgrid(z*1e3, y*1e3)
levels = np.linspace(-np.pi/2.0, np.pi/2.0, 9)
ax2.contour(Zd, Yd, im, levels=levels)

ax1.set_xlabel('z (mm)')
ax1.set_ylabel('y (mm)')
ax2.set_xlabel('z (mm)')

ax1.set_title('$|E|$')
ax2.set_title('Phase')

#%%
# Check simulation end

idx = -1
focal_slice = Field[:,:,idx].copy()

# Get beam intensity at this position
I = shadow(focal_slice)
e2_val = I.max()*np.exp(-2)

extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6

plt.figure()
plt.title('Intensity')
plt.imshow(I, cmap='viridis', 
           extent=extent,
           origin='lower')
ax = plt.gca()
CS = ax.contour(1e6*X, 1e6*Y, I, levels=[e2_val], colors=['y'])
ax.clabel(CS, CS.levels, inline=True, fmt='$1/e^{2}$', fontsize=10)

plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')

#%%
# lineout plot

plt.figure()
plt.plot(y*1e3, t[:, mid] / t[:, mid].max(), 'r', label='Input')
plt.plot(y*1e3, I[:, mid] / I[:, mid].max(), 'k', label='Output')
plt.xlabel('y [mm]')
plt.ylabel('Intensity [arb.]')
plt.xlim((-10, +10))
plt.grid()
plt.show()
