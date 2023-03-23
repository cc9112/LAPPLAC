#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input beam at OAP with phase mask applied, then propagate to its focal spot.

Use quick_free_propagate method to perform quicker.

Need large simulation domain to have good resolution in focal plane AND 
see full input beam. Instead, use large f_number.

Currently set to f/400 (10x larger). Expect focal spot to be 10**2 larger than
designed.

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
lambda0 = 0.8 * um # [m]
beamDiameter = 150e-3
f_number = 400
zf = f_number * beamDiameter # distance to focus
z0 = -zf  # inital z value (relative to focal plane)

w_z0 = beamDiameter / 2.0 # waist size at start (z = 0)

theta_half = 0.5 * (1 / f_number) # divergence half-angle

# single solution now for w0 from lambda0 and theta_half
w0 = lambda0 / (np.pi * theta_half) # focal spot radius size (waist) if perfect Gaussian beam


laserParameters = {
'lambda0' 	: lambda0,  	    # wavelength (m)
'tau'    	: 50.0*fs,	 	    # pulse width (s)
'E0'     	: 0.7 * 15.0,		  	# pulse energy (J)
'xOffset'	: 0.0*um,			# Spatial offset from center in x (m) - use at your own risk for the moment
'yOffset'	: 0.0*um,			# Spatial offset from center in y (m) - use at your own risk for the moment
'shape'  	: 'gaussian',
'n'          : 10,                # n=2 is gaussian, n>2 is supergaussian
'w0'         : w0,                # final waist size
'w_z0'       : w_z0               # initial waist size
}



zR = get_zr(laserParameters) # get Rayleigh range for ideal Gaussian beam

Rz = z0 * (1.0 + (zR/z0)**2) # Radius of curvature of focusing Gaussian beam

#%%
# Setup computational parameters to visualise input field

total_simulation_size = 200 * 1e-3 # [m] - big enough to fit inputBeam

n_pts  = 2**10 * 2**2	# number of pixels (must be even)

res = total_simulation_size / n_pts  # smallest pixel size, only 1 in the focal spot.


mid = int(n_pts/2)

#%%

# Create calculation grid
x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts ) #centering occurs here
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
X, Y = np.meshgrid(x,y)

print('Creating fields.')

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


# add in outline of waist
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

# add in phase for focusing beam
k = 2.0*np.pi/lambda0
R_sq = X**2 + Y**2
guoy = np.arctan(z0/zR) #seems to have little, to no, effect?
inputPhase = k * R_sq/(2.0*Rz) - guoy

# add in phaseMask to phase (distributed phase plate)
# load in data, then sample on grid on simulation
# then add onto inputBeam phase
filename = '../experiment_phase_masks/2018_BW/Gemini 80x200um.CSV'
dpp = np.loadtxt(filename, delimiter=',', dtype=int) # binary mask of pi phase shifts
dpp = dpp * np.pi
dpp_fw =  177.8 * 1e-3 #mm
dpp_hw = dpp_fw/2.0
nrows,ncols = dpp.shape

x_dpp = np.linspace(-dpp_hw, dpp_hw, ncols)
y_dpp = np.linspace(-dpp_hw, dpp_hw, nrows)
from scipy.interpolate import RegularGridInterpolator
dpp_fcn = RegularGridInterpolator((x_dpp,y_dpp),dpp,method='nearest', bounds_error=False, fill_value=0)
phaseMask = dpp_fcn((X,Y))
inputPhase = inputPhase + phaseMask



inputBeam = inputBeam * np.exp(-1j * inputPhase) #get it into complex format


plt.figure()
plt.imshow(inputPhase, cmap='bwr', 
           extent=extent)
plt.plot(r*np.cos(theta), r*np.sin(theta), 'r--')
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.title('inputPhase')


print('Finished creating fields.')


#%%
#Setup Plasma Channel Parameters
eta0 		 	= 1.0			# Refractive index of surrounding media
k0 = 2.0*np.pi / laserParameters['lambda0']

#%%
#JUMP TO POINT IN Z OF INTEREST
print('Setting up calculation.')
final_z = zf #jump to focal plane

#Absorbing boundaries
absorbingBoundaries = 1
z = np.array([0.0, final_z]) #quick hack
eta = eta0 * np.ones((n_pts,n_pts,z.size))  #has to be for free-space

AS, xFreq, yFreq, G, mask, X, Y, z, eta = setup_propagate(inputBeam, X, Y, z, eta, k0, eta0, absorbingBoundaries)
print('Finished setting up calculation.')

print('Start calculation.')
field_final = quick_free_propagate(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k0, eta0=1)
print('Calculation Finished.')


#%%
# FOCAL SPOT ANALYSIS

# AT FOCAL PLANE, W0 = 1/e^2 RADIUS OF THE BEAM'S INTENSITY

focal_slice = field_final.copy()

I = shadow(focal_slice)

extent = np.array([x.min(), x.max(), y.min(), y.max()])*1e3

plt.figure()
plt.imshow(I, cmap='viridis', 
           extent=extent)
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.title(f'Focal Spot: F/{f_number}')


# phasemask originally designed for F40 beam - and designed to produce 80 x 200 um spot
# F_number 10x larger
# So spot size increases by 10**2
# So focal spot should be 8 mm x 20 mm

plt.xlim((-30, +30))
plt.ylim((-30, +30))



#%%
plt.show()
