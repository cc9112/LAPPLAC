#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to see focus changes from Zernike polynomials
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
'lambda0' 	: 0.800*um,  	    # wavelength (m)
'tau'    	: 100.0*fs,	 	    # pulse width (s)
'E0'     	: 1.0,		  	# pulse energy (J)
'xOffset'	: 0.0*um,			# Spatial offset from center in x (m) - use at your own risk for the moment
'yOffset'	: 0.0*um,			# Spatial offset from center in y (m) - use at your own risk for the moment
'shape'  	: 'gaussian',
'n'          : 10,                         # n=2 is gaussian, n>2 is supergaussian
'w0'         : 50.0*um          #final waist size
}

zR = get_zr(laserParameters)

zf = 2.0*zR #z point of focus

z0 = -zf  #relative to focal plane
Rz = z0 * (1.0 + (zR/z0)**2) #radius of curvature

w0 = laserParameters['w0']
w_z0 = w0 * np.sqrt(1.0 + (z0/zR)**2)

laserParameters['w_z0'] = w_z0

k0 = 2.0*np.pi / laserParameters['lambda0']

#%%
#Setup computational parameters
n_pts  = 2048   		# number of pixels (must be even)
res    = 0.8*um   		# resolution in image plane (m)

mid = int(n_pts/2)

#%%

# Create calculation grid
x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts ) #centering occurs here
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
X, Y = np.meshgrid(x,y)


#get E field amplitude 
inputBeam = make_inputBeam(X,Y,laserParameters)
#Setup phase profile
#non-focusing beam - infinite rayleigh range
#zR = np.inf


I = shadow(inputBeam)
extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6
plt.figure()
plt.imshow(I, cmap='viridis', 
           extent=extent)
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.title('inputBeam')

ls = inputBeam[mid,:mid]
l_axis = x[:mid]
hm = ls.max()/2.0
l_point = np.interp(hm, ls, l_axis)

rs = inputBeam[mid,mid:]
r_axis = x[mid:]
hm = rs.max()/2.0
r_point = np.interp(hm, rs[::-1], r_axis[::-1])
fwhm = r_point - l_point

fwhm *= 1.0

theta = np.linspace(0.0, 2.0*np.pi, 100)
r = w_z0/2.0 *1e6
r = fwhm/2.0 *1e6
plt.plot(r*np.cos(theta), r*np.sin(theta), 'r--')


#add phase terms in defocus is n=2 m=1
Rho = (X**2 + Y**2)**(0.5) / (fwhm/2.0)

Theta = np.arctan2(Y,X)
Z_defocus = 2*Rho**2 - 1.0
Z_coma = (3*Rho**3 - 2*Rho) * np.sin(Theta)

A_defocus = 0.0
A_coma = 0.2

terms = [A_defocus * Z_defocus, A_coma*Z_coma]
inputPhase = np.sum(terms, axis=0)

inputPhase[Rho>1] = 0.0

inputBeam = inputBeam * np.exp(-1j * 2.0*np.pi * inputPhase) #get it into complex format


plt.figure()
plt.imshow(inputPhase, cmap='bwr', 
           extent=extent)
plt.plot(r*np.cos(theta), r*np.sin(theta), 'r--')
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.title('inputPhase')

#%%
#Setup Plasma Channel Parameters
eta0 		 	= 1.0			# Refractive index of surrounding media

#%%
#JUMP TO POINT IN Z OF INTEREST
    
final_z = zf #jump to focal plane

#Absorbing boundaries
absorbingBoundaries = 1
z = np.array([0.0, final_z]) #quick hack
eta = eta0 * np.ones((n_pts,n_pts,z.size))  #has to be for free-space

AS, xFreq, yFreq, G, mask, X, Y, z, eta = setup_propagate(inputBeam, X, Y, z, eta, k0, eta0, absorbingBoundaries)

field_final = quick_free_propagate(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k0, eta0=1)

#%%
# FOCAL SPOT ANALYSIS

# AT FOCAL PLANE, W0 = 1/e^2 RADIUS OF THE BEAM'S INTENSITY

focal_slice = field_final.copy()

I = shadow(focal_slice)

extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e6

plt.figure()
plt.imshow(I, cmap='viridis', 
           extent=extent)
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.title('Focal Spot')