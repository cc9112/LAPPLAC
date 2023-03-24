#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a python script that replicates the 
example_TransverseProbing.m

A 400 nm beam passes side on to a plasma-channel of 50 um radius. 
A supergaussian input laser profile is used.

Results and plots agree with output from Matlab code, although the way this 
script is written is not one-to-one (this was done with another script).

The workflow is a copy of Rob's LAPPLAC code, from matlab to python.
A combination of many scripts enough to replicate the example.

Created on Fri Jul 10th 11:31:00 2020

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

w0 = 200.0*um

laserParameters = {
            'lambda0' : 0.400*um,
            'tau'    	: 100*fs,	 	    # pulse width (s)
            'E0'     	: 0.001,		  	# pulse energy (J)
            'xOffset'	: 0.0,			# Spatial offset from center in x (m) - use at your own risk for the moment
            'yOffset'	: 0.0,			# Spatial offset from center in y (m) - use at your own risk for the moment
            'shape'  	: 'gaussian',
            'n'          : 10,                         # n=2 is gaussian, n>2 is supergaussian
            'w0'         : w0          #final waist size
            }


#%%
# Setup computational parameters
# ------------------------------

n_pts  = 512   		# number of pixels (must be even)
res    = 1.0*um   		# resolution in image plane (m)

nz     = 512    		# number of points in z (must be even)
zMax   = 2.0*mm

mid = int(n_pts/2)

#%%
# Create calculation grid
# -----------------------

x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts ) #centering occurs here
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
X, Y = np.meshgrid(x,y)

z = np.linspace(0,zMax,nz)

# attempt to make perfectly non-focusing beam
laserParameters['w_z0'] = w0
inputBeam = make_inputBeam(X,Y,laserParameters)
inputBeam = inputBeam * np.exp(-1j*0.0) # get it into complex format - no phase profile



fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('inputBeam')

extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6

ax1.set_title('|E|$^{2}$ Waist')
t = shadow(inputBeam)**2
#t[t<t.max()*np.exp(-2)] = 0.0
ax1.imshow(t, extent=extent)


ax2.set_title('Phase')
ax2.imshow(phase(inputBeam), extent=extent)


#%%
# Setup Plasma Channel Parameters
# -------------------------------

# Rob's code had a function
# Generate the transverse density profile 
#taken from generateTransversePlasmaChannel.m
#[data.nCr,data.ne] = generateTransversePlasmaChannel(data);
# I haven't copied this yet.

# Set up cylinder of plasma
# uniform density (ne0) inside channel of radius r0
# vacuum outside

eta0 		 	= 1.0			# Refractive index of surrounding media - vacuum
z0               = 50 * zMax/nz    # Centre point for channel in z
r0               = 50 * um           # radius of plasma channel;
y0               = 0.0


# Using these input parameters generate the input beam and plasma channel 
#[data.x, data.y, data.z, data.beam] = generateInputBeam(data);

omega0 = 2.0 * np.pi * c / laserParameters['lambda0']	# Angular frequency of laser
n_crit = m_e * epsilon_0 * omega0**2 / e**2		# Critical density

ne0 = 0.001 * n_crit # 1 % of n_crit inside channel



# Make plasma channel 
ne = np.zeros((n_pts, n_pts, nz))

X_pl, Y_pl, Z_pl = np.meshgrid(0.0*x, y-y0, z-z0) 
# get distance from channel centre
dst = np.sqrt(X_pl**2 + Y_pl**2 + Z_pl**2)
ne[dst <= r0] = ne0

eta 		=  np.sqrt(1.0 - ne/n_crit) 	# Refractive Index of the Plasma
eta0     = 1.0 # vacuum

#%%
# Setup first step and tools for propagation 
# ------------------------------------------

# Absorbing boundaries
# Reduces the effect of the edges of the domain
absorbingBoundaries = 1


k0 = 2.0 * np.pi / laserParameters['lambda0']
AS, xFreq, yFreq, G, mask, X, Y, z, eta = setup_propagate(inputBeam, X, Y, z, eta, k0, eta0, absorbingBoundaries)

#%%
#% MAIN ITERATION LOOP IN dz
# --------------------------
# Propagate the beam through the plasma structure

Field = propagate_in_z(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k0)

#%%
# Interactive viewer of whole propagation of beam
# -----------------------------------------------

buttons = explorer(Field, ne, z)


#%%
#COMPARE WITH ROB'S PLOTS



# Plot intensity profile
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.abs(inputBeam)**2, extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='Greys_r')
ax2.imshow(np.abs(Field[:, :, -1])**2, extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='Greys_r')
ax1.set_title('Intensity at z=0.0 mm')
ax2.set_title('Intensity at z=2.0 mm')
ax1.set_xlabel('x (mm)')
ax2.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')



# Plot phase of beam
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.angle(inputBeam), extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='jet')
ax2.imshow(np.angle(Field[:, :, -1]), 
           extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], 
           cmap='jet',
           vmin = 0.0,
           vmax = 1.1)
ax1.set_title('Phase at z=0.0 mm')
ax2.set_title('Phase at z=2.0 mm')


# Rob's used a custom colormap here I can't quite copy
# Plot propagation intensity vs Z
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
asp = (z.max() - z.min()) / (y.max()  - y.min())
ax1.imshow(np.abs(Field[:, 256, :])**2, aspect=1.0, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])
ax2.imshow(ne[:, 256, :], extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])

ax1.set_title('Propagation Intensity')
ax2.set_title('Plasma $n_e$')
ax1.set_xlabel('z (mm)')
ax2.set_xlabel('z (mm)')
ax1.set_ylabel('y (mm)')
ax2.set_ylabel('y (mm)')
