#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicate propagation of ideal Guassian beam to its focus, but much faster.

This uses the quick_free_propagate method, which removes most of the 
intermediate steps in propagating a beam a large distance in vacuum.

This method works for any beam profile, as long as its in vacuum.

Here it is just tested with a simple Gaussian beam case to directly compare to 
example_FreeSpacePropagation_FocusingGaussian.py

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
laserParameters = {
'lambda0' 	: 0.800*um,  	    # wavelength (m)
'tau'    	: 100.0*fs,	 	    # pulse width (s)
'E0'     	: 1.0,		  	# pulse energy (J)
'xOffset'	: 0.0*um,			# Spatial offset from center in x (m) - use at your own risk for the moment
'yOffset'	: 0.0*um,			# Spatial offset from center in y (m) - use at your own risk for the moment
'shape'  	: 'gaussian',
'n'          : 2,                         # n=2 is gaussian, n>2 is supergaussian
'w0'         : 30.0*um          #final waist size
}

zR = get_zr(laserParameters)

zf = 3.0*zR #z point of focus

z0 = -zf  #relative to focal plane
Rz = z0 * (1.0 + (zR/z0)**2) #radius of curvature

w0 = laserParameters['w0']
w_z0 = w0 * np.sqrt(1.0 + (z0/zR)**2)

laserParameters['w_z0'] = w_z0

k0 = 2.0*np.pi / laserParameters['lambda0']

#%%
# Setup computational parameters
# -----------------------------

# As computation is much faster, this can be set to a much higher resolution
# than when calculating all intermediate steps.

n_pts  = 1024   		# number of pixels (must be even)
res    = 0.8*um   		# resolution in image plane (m)


mid = int(n_pts/2)

#%%

#%%
# Create calculation grid
# ------------------------
x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts ) #centering occurs here
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
X, Y = np.meshgrid(x,y)


#get E field amplitude 
inputBeam = make_inputBeam(X,Y,laserParameters)

# Setup phase profile

# focusing beam
w_z0 = laserParameters['w_z0']
lambda0 = laserParameters['lambda0']

zR = np.pi*w0**2/(lambda0) #raylrigh range for guassian beam


k = 2.0*np.pi/lambda0
R_sq = X**2 + Y**2
guoy = np.arctan(z0/zR) #seems to have little, to no, effect?
inputPhase = k * R_sq/(2.0*Rz) - guoy

inputBeam = inputBeam * np.exp(-1j * inputPhase) #get it into complex format


# CHECK INPUT BEAM
# WAIST SIZE IS SPOT ON!

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
#Setup Plasma Channel Parameters
eta0 		 	= 1.0			# Refractive index of surrounding media

#%%
#JUMP TO POINT IN Z OF INTEREST
    
final_z = zf #jump to focal plane

#Absorbing boundaries
absorbingBoundaries = 1
z = np.array([0.0, final_z]) # Quick hack here for setup_propagate function
eta = eta0 * np.ones((n_pts,n_pts,z.size))  # Has to be for free-space
AS, xFreq, yFreq, G, mask, X, Y, z, eta = setup_propagate(inputBeam, X, Y, z, eta, k0, eta0, absorbingBoundaries)

field_final = quick_free_propagate(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k0, eta0=1)

#%%
# COMPARE FOCAL SPOT TO GUASSIAN BEAM PROPAGATION THEORY
# AT FOCAL PLANE, W0 = 1/e^2 RADIUS OF THE BEAM'S INTENSITY

# Get beam intensity at this position
I = shadow(field_final)
e2_val = I.max()*np.exp(-2)

extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6

plt.figure()
plt.title('Intensity at Focus')
plt.imshow(I, cmap='viridis', 
           extent=extent)
ax = plt.gca()
CS = ax.contour(1e6*X, 1e6*Y, I, levels=[e2_val], colors=['y'])
ax.clabel(CS, CS.levels, inline=True, fmt='$1/e^{2}$', fontsize=10)

plt.xlabel('x [$\mu$m]')
plt.ylabel('y [$\mu$m]')

# Compare to theoretial focal spot size
# Good agreement
theta = np.linspace(0.0, 2.0*np.pi, 100)
r = w0 * 1e6
x_circ,y_circ = r*np.cos(theta), r*np.sin(theta)
plt.plot(x_circ,y_circ,'r', label='Theoretical focal spot')
plt.legend()