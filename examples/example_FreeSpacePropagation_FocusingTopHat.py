#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a python script that replicates the 
example_FreeSpacePropagation.m

It propagates a top-hat beam from z=-3zR to z=+3zR, and compares the focal spot
to analytical theory.

The grid spacing is important:
    Too small and edge effects alter the beam (seen as repeating 2D patterns)

    Too large and the focal spot cannot be resolved at z=0

    A 512 x 512 grid of 3 um pixels is a good balance between these two.

Beam propagation is consistent (within the simulation) to theory, i.e.:
    Discretised grid causes slight correction to beam size in simulation. 
    The initial e**(-2) size differs by 3 % to the user-specified.
    
    The focal spot intensity profile matches well to an input beam with a 3 %
    larger initial size.
    
    RMS of focal spot profile differs from analytical by 0.1 %

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
#Setup Laser Parameters
laserParameters = {
'lambda0' 	: 0.800*um,          # wavelength [m]
'tau'    	: 100.0*fs,	 	   # pulse width [s]
'E0'     	: 1.0,		  	   # pulse energy [J]
'xOffset'	: 0.0*um,		   # Spatial offset from center in x [m] - use at your own risk for the moment
'yOffset'	: 0.0*um,		   # Spatial offset from center in y [m] - use at your own risk for the moment
'shape'  	: 'gaussian',
'n'          : 20,                # n=2 is gaussian, n>2 is supergaussian
'w0'         : 30.0*um            # final waist size
}

# Define the beam profile and generate the beam 
# ---------------------------------------------

zR = get_zr(laserParameters) # calculated from given w0 and lambda0
zMax   = 2.0 * 3.0 * zR # want z to go from -3*zR to +3*zR

zf = 3.0*zR # z position of focus relative to start of simulation
z0 = -zf  # invert - z position of start relative to focus (which is now at z=0)


Rz = z0 * (1.0 + (zR/z0)**2) # radius of curvature
w0 = laserParameters['w0']
w_z0 = w0 * np.sqrt(1.0 + (z0/zR)**2) # beam waist size at focus
laserParameters['w_z0'] = w_z0

#%%
# Setup computational parameters
# ------------------------------

n_pts  = 512   		# number of pixels (must be even)
res    = 3*um   		# resolution in image plane [m]
nz     = 512    		# number of points in z (must be even)

mid = int(n_pts/2)
#%%
# Create simulation grid / domain
# -------------------------------
x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
X, Y = np.meshgrid(x,y)
z = np.linspace(0, zMax, nz)


#%%
# Get E field amplitude 

inputBeam = make_inputBeam(X,Y,laserParameters)

#%%
# Setup phase profile
# -------------------

# Focusing beam
w_z0 = laserParameters['w_z0']
lambda0 = laserParameters['lambda0']
zR = np.pi * w0**2 / lambda0 # Rayleigh range for guassian beam


k = 2.0*np.pi/lambda0
R_sq = X**2 + Y**2
guoy = np.arctan(z0/zR) # Seems to have little, to no, effect?
inputPhase = k * R_sq/(2.0*Rz) - guoy
inputBeam = inputBeam * np.exp(-1j * inputPhase) # Get into complex format


# Check input beam
# ----------------
intensity = shadow(inputBeam)**2
p = phase(inputBeam)

# get w_z0 value in simulation domain
w_z0_actual = (1/2) * find_FWHM(x, intensity[n_pts//2,:], frac=np.exp(-2))

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('inputBeam')
extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6
ax1.set_title('|E|$^{2}$ Waist')
ax1.imshow(intensity, extent=extent)

# Compare user-specified and domain beam sizes
theta = np.linspace(0.0, 2.0*np.pi, 100)
r = w_z0 * 1e6
x_circ,y_circ = r*np.cos(theta), r*np.sin(theta)
ax1.plot(x_circ,y_circ,'r', label='User Specified')
r = w_z0_actual * 1e6
x_circ,y_circ = r*np.cos(theta), r*np.sin(theta)
ax1.plot(x_circ,y_circ,'w--', label='Simulated')
ax1.legend()

# Check phase - at this stage just looks mental
ax2.set_title('Phase')
ax2.imshow(p, extent=extent)

#%%
# Setup tools for propagation
# --------------------------

# Absorbing boundaries - reduces the affect of the edges of the domain
absorbingBoundaries = 1

# Define refractive index and ne for vacuum
ne = np.zeros((n_pts, n_pts, nz))
eta0 = 1.0
eta = eta0 * np.ones((n_pts, n_pts, nz))
k0 = 2.0*np.pi/laserParameters['lambda0']

# First step
AS, xFreq, yFreq, G, mask, X, Y, z, eta = setup_propagate(inputBeam, X, Y, z, eta, k0, eta0, absorbingBoundaries)

#%%
#% MAIN ITERATION LOOP IN dz
# --------------------------

Field = propagate_in_z(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k0)

#%%
# Interactive plot to explore propagation
# --------------------------------------

buttons = explorer(Field, ne, z)

#%%
# View Propagation
# ----------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

fig.suptitle('Propagation')

asp = (z.max() - z.min()) / (y.max()  - y.min())
ax1.imshow(shadow(Field[:, mid, :]), aspect=asp, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])

im = phase(Field[:, mid, :])
ax2.imshow(im, aspect=asp, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3],
           cmap=matplotlib.cm.RdBu_r, 
           norm=MidpointNormalize(midpoint=0.0,vmin=-np.pi/2.0, vmax=np.pi/2.0))
Zd,Yd = np.meshgrid(z*1e3, y*1e3)
levels = np.linspace(-np.pi/2.0, np.pi/2.0, 9)
ax2.contour(Zd, Yd, im, levels=levels)

ax1.set_xlabel('z (mm)')
ax1.set_ylabel('y (mm)')
ax2.set_xlabel('z (mm)')
ax3.set_xlabel('z (mm)')

# Look at how much of beam remains > e-2 during propagation
thresh = np.zeros_like(Field[:, mid, :], dtype=np.float64)
for zi in range((Field.shape[-1])):
    f = Field[:,mid,zi].copy()
    f = np.abs(f)**2
    
    top = np.max(f)
    f[f <= top*np.e**(-2)] = 0.0
    f[f > top*np.e**(-2)] = 1.0
    
    thresh[:,zi] = f
      
ax3.imshow(thresh, aspect=asp, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])

ax1.set_title('$|E|$')
ax2.set_title('Phase')
ax3.set_title('$|E|^2 \geq e^{-2}$')

#%%
# Focal Spot Analysis - Compare to similar Guassian
# -------------------------------------------------

# Find closest z slice to focal plane
idx = np.argmin((z-zf)**2)
focal_slice = Field[:,:,idx].copy()

# Get intensity of focal plane
I = shadow(focal_slice)

# Plot Intensity Lineouts
plt.figure()
plt.title('Focal Spot Intensity')
I_x_lineout = I[n_pts//2,:]
I_y_lineout = I[:,n_pts//2]
plt.plot(x*1e6, I_x_lineout, '-', label='x lineout')
plt.plot(y*1e6, I_y_lineout, '--', label='y lineout')
plt.xlabel('r [$\mu$m]')
plt.ylabel('I [arb.]')
plt.grid()

# Similar Gaussian spot
effective_f_number = zf / (2 * w_z0_actual)
FWHM = effective_f_number * lambda0 * 1e6
sigma = FWHM / (2 * (2*np.log(2))**(0.5))
I0 = I_x_lineout.max()
mu = 0.0
I_Gauss = I0 * np.exp(-((x*1e6-mu)**2 / (2 * sigma**2)))

plt.plot(x*1e6, I_Gauss, color='k', label='Gaussian FWHM=$\lambda_0\cdot$f/#')

plt.xlim((-100, +100))
plt.legend(loc=1)

#%%
plt.show()