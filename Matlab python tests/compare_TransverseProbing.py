#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a python script that replicates the 
example_TransverseProbing.m

A 400 nm beam passes side on to a plasa-channel of 50 um radius. 
A supergaussian input laser profile is used, which even without plasma diffracts
a little.

Results and plots agree with Rob's.

The workflow is a copy of Rob's LAPPLAC code, from matlab to python.
A combination of many scripts enough to replicate the example

Created on Fri Jul 10th 11:31:00 2020

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import unwrap_phase

def phase(Field_slice):
    """Quick way to get phase out from laser field to compare with interferometry.
    
    Expects Field_slice to be a 2D array    
    
    Also needs to unwrap in z! 
    """
    phase = np.angle(Field_slice, deg=False)
    return unwrap_phase(phase)


def shadow(Field_slice):
    """
    """
    return np.abs(Field_slice)**2


def fourierTransform(signal,X,Y,padx,pady, debug):
    """padding necessary if spatially x is longer than y etc.
    see https://uk.mathworks.com/help/matlab/ref/fft2.html
    """
    #FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal), pady, padx))
    FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal)))
    #Not worrying about scalings in FT domain as we're only interested in the space domain

    
    # Calculate the sampling frequency in each direction
    xSampleFreq = len(X)/(X[-1]-X[0])
    ySampleFreq = len(Y)/(Y[-1] - Y[0])
    
    # Calculate Frequency Axis
    #xFreq = 2.0*np.pi*[-xSampleFreq/2 : xSampleFreq/padx : (xSampleFreq/2 - xSampleFreq/padx) ]
    #yFreq = 2.0*np.pi*[-ySampleFreq/2 : ySampleFreq/pady :  (ySampleFreq/2 - ySampleFreq/pady) ]
    #in matlab, [from this: steps of this: to this]
    xFreq = 2.0*np.pi*np.arange(-xSampleFreq/2, (xSampleFreq/2), xSampleFreq/padx) #- xSampleFreq/padx from middle doesn't seem to be needed.
    yFreq = 2.0*np.pi*np.arange(-ySampleFreq/2, (ySampleFreq/2), ySampleFreq/pady)#- ySampleFreq/pady
    

    if debug == 1:
        rows, cols = np.shape(FT)
        plt.figure()
        plt.imshow(np.log(np.abs(FT))) #xFreq,yFreq,
        plt.title('Output from fourierTransform.m')
    
    return xFreq,yFreq,FT

def inverseFourierTransform(FT,xFreq,yFreq,padx,pady,debug):
    """Don't exactly know why I need to do the two inverses...but it works...
    IFT = fftshift(ifft2(ifftshift(FT'),padx,pady))';
    ROB HAS CHANGED THE FREQUENCY AXIS BY A FACTOR OF 2PI
    THIS WAS DONE SUCH THAT WHEN YOU INPUT Y = A*SIN(K X) YOU GET BACK PEAKS AT K, NOT K/2PI
    """
    IFT = np.transpose(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.transpose(FT)))))

    xSampleFreq = len(xFreq)*np.abs(xFreq[2]-xFreq[1])
    ySampleFreq = len(yFreq)*np.abs(yFreq[2]-yFreq[1])

    dX = 2*np.pi/xSampleFreq
    dY = 2*np.pi/ySampleFreq
    
    #Y = [-dY*length(yFreq)/2 : dY : dY*length(yFreq)/2 - dY  ];  
    X = np.arange(-dX*len(xFreq)/2, dX*len(xFreq)/2 - dX, dX)
    Y = np.arange(-dY*len(yFreq)/2, dY*len(yFreq)/2 - dY, dY)
  
    
    if debug == 1:
        rows, cols = np.shape(FT)
        plt.figure()
        plt.title('Output of inverseFourierTransform.m')
        plt.imshow(np.abs(IFT))

    return X, Y, IFT

# Constants
um		= 1e-6		# microns
mm 		= 1e-3		# millimeters
fs 		= 1e-15	# femtoseconds
re 		= 2.81794e-15 		# classical electron radius
epsilon = 8.85e-12              # permittivity of free space
me 		= 9.1e-31			# electron mass
e 		= 1.6e-19			# electron charge
c 		= 3e8				# speed of light

#%%
#Setup computational parameters
n_pts  = 512   		# number of pixels (must be even)
res    = 1.0*um   		# resolution in image plane (m)
nz     = 256    		# number of points in z (must be even)
zMax   = 2.0*mm		


# Create calculation grid
x = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts ) #centering occurs here
y = np.linspace(-res*n_pts/2 , res*n_pts/2, n_pts )
z = np.linspace(0,zMax,nz)
X, Y = np.meshgrid(x,y)

#%%
#Setup Laser Parameters
lamda 	= 0.400*um  	    # wavelength (m)
tau    	= 100*fs	 	    # pulse width (s)
E0     	= 0.001		  	# pulse energy (J)
w0     	= 400.0*um     	# 1/e^2 intensity laser radius (m) (*assuming a gaussian pulse*)
xOffset	= 0.0*um			# Spatial offset from center in x (m) - use at your own risk for the moment
yOffset	= 0.0*um			# Spatial offset from center in y (m) - use at your own risk for the moment
shape 		= 'supergaussian'	#Rob actually used a supergaussian here instead
power_factor = 10.0

# Define the beam profile and generate the beam 
#taken from generateInputBeam.m
#returns [x,y,z,inputBeam] 


#Setup intensity profile
if shape=='gaussian':
    r_sq = (X-xOffset)**2 + (Y-yOffset)**2
    intensity = np.exp(-2*(r_sq)/(w0**2))
    
	# Fix the intensity by scaling it appropriately 
    flux = np.trapz(np.trapz(intensity, x, axis=1), y) #previously dim=2, basically 2d integral
    inputBeam = np.sqrt(E0/(flux*tau) * intensity) # The field amplitude in sqrt(W/cm^2)

elif shape=='supergaussian':
    r_sq = (X-xOffset)**2 + (Y-yOffset)**2
    index = (2.0*(r_sq)/(w0**2))**(power_factor)
    intensity = np.exp(-index)
    
	# Fix the intensity by scaling it appropriately 
    flux = np.trapz(np.trapz(intensity, x, axis=1), y) #previously dim=2, basically 2d integral
    inputBeam = np.sqrt(E0/(flux*tau) * intensity) # The field amplitude in sqrt(W/cm^2) 
    
elif shape=='uniform':
    intensity = np.ones_like(X)
    flux = np.trapz(np.trapz(intensity, x, axis=1), y)
    inputBeam = np.sqrt(E0/(flux*tau) * intensity)    


#Setup phase profile
#non-focusing beam
inputBeam = inputBeam * np.exp(-1j*0.0) #get it into complex format



#%%
#Setup Plasma Channel Parameters
eta0 		 	= 1.0			# Refractive index of surrounding media
channelType 	= 'cylinder'	# Options: 'parabolic', 'realistic','user', 'none'
ne0			= 5.0e18 * 1e6			# [parabolic or realistic channels] On-Axis Density (m^-3)
z0             = 50 * zMax/nz  # Starting point for channel in z
r0             = 50*um         # radius of plasma channel;
y0 = 0.0

# Using these input parameters generate the input beam and plasma channel 
#[data.x, data.y, data.z, data.beam] = generateInputBeam(data);

# Generate the transverse density profile 
#taken from generateTransversePlasmaChannel.m

#[data.nCr,data.ne] = generateTransversePlasmaChannel(data);

# Some useful conversions
omega = 2.0*np.pi*c/lamda	# Angular frequency of laser
nCr = me*epsilon*omega**2/e**2		# Critical density
ne = np.zeros((n_pts, n_pts, nz))

if channelType=='cylinder':    
    Z_proj, Y_proj = np.meshgrid(z-z0, y-y0) #first gives ncols (x), second gives nrows (y) - made to match iteration through ne below.
    
    neSlice = np.zeros((n_pts, nz)) #to match above
    dst = np.sqrt(Y_proj**2 + Z_proj**2)
    neSlice[dst<r0] = ne0 #removed transpose, not needed?

ne = np.zeros((n_pts, n_pts, nz))

for i in range(len(x)): #iterating through x, is iterating through columns
    #column by column
	ne[:, i, :] = neSlice
#returns nCr, ne


#%%
#Setup first step and tools for propagation
# Propagate the beam through the plasma structure
#data.Field = beamPropagator(data); 

#Absorbing boundaries
absorbingBoundaries = 1

# Plasma Channel Parameters
eta 		=  np.sqrt(1.0 - ne/nCr) 	# Refractive Index of the Plasma

# Some useful conversions
k 						= 2.0*np.pi/lamda		# Laser wave vector
eta = np.dstack((eta, eta[:,:,-1])) #Extend eta array by one to allow for stepped refraction calc.

# Find Angular Spectrum of the pulse
xFreq, yFreq, AS = fourierTransform(inputBeam,x,y,n_pts,n_pts,debug=0)
XF, YF = np.meshgrid(xFreq, yFreq)
dz = z[1]-z[0]

# Calculate the free space propagator G for half steps in z - this is constant over whole sim
p = np.sqrt((k**2)*eta0**2 - XF**2 - YF**2)
G = np.exp(-1j*(p-k*eta0)*dz/2.0)

# Blank array for scalar field values
Field = np.zeros((n_pts,n_pts,nz), dtype=np.complex64)

if absorbingBoundaries==1:
    scaleFactor = 5/8 #WHAT IS THIS FOR?
    mask = np.exp(-(2*((X**2 + Y**2)/((scaleFactor*n_pts*res)**2)))**10.0)
    
    #TYPO FIXED HERE RELATIVE TO ROB'S CODE - AS OF 30-07-20
    #(power of 10 added)


#%%
#% MAIN ITERATION LOOP IN dz
for i in range(len(z)):
    print('Step %i of %i' % (i,len(z)))
    # Propagate Angular spectrum downstream by half a z step
    AS = G*AS

    # Fourier transform to determine field 
    _, _, tmp = inverseFourierTransform(AS,xFreq,yFreq,n_pts,n_pts,debug=0)
    
    # Now Add refraction effects
    B = np.exp( - 1j*k*dz/2*(eta[:,:,i]-eta0) )
    Bprime = np.exp( - 1j*k*dz/2*(eta[:,:, i+1]-eta0) )
    T = tmp*B*Bprime 
    
    # Fourier Transform back to propagate the final half step
    xFreq, yFreq, AS = fourierTransform(T,x,y,n_pts,n_pts,0)
    AS = G*AS
    if absorbingBoundaries==1:
        AS = AS*mask
    # Add this field component to our array 
    _, _, Field[:,:,i] = inverseFourierTransform(AS,xFreq,yFreq,n_pts,n_pts,0)

#%%
#EXPLORER - PLOTS 
from matplotlib.widgets import Button
import matplotlib.gridspec
plt.ion()

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2,4)

#initial plots
ax_out = fig.add_subplot(gs[0,3])
im2 = ax_out.imshow(shadow(Field[:,:,-1]), origin='lower') 
ax_out.set_xlabel('Shadowgraphy')
ax_out.set_title('z = %.2f $\mu$m' % float(z[-1]*1e6))

ax_in = fig.add_subplot(gs[0,0])
vmin, vmax = im2.get_clim()
im = ax_in.imshow(shadow(Field[:,:,0]), origin='lower', vmin=vmin, vmax=vmax) 
ax_in.set_ylabel('Y (pixels)')
ax_in.set_xlabel('X (pixels)')
ax_in.set_title('Input beam')


ax_out2 = fig.add_subplot(gs[1,3])
im3 = ax_out2.imshow(phase(Field[:,:,-1]), origin='lower') 
ax_out2.set_xlabel('Phase')
ax_out2.set_title('z = %.2f $\mu$m' % float(z[-1]*1e6))



nemin, nemax = np.min(ne), np.max(ne)
ax_endon = fig.add_subplot(gs[0,1])
#fig2, ((ax_endon, ax_topdown), (_, ax_transverse)) = plt.subplots(2,2)
ax_endon.imshow(np.transpose(ne[:, int(n_pts/2), :]), origin='lower', vmin=nemin, vmax=nemax) #constant x - end on
ax_endon.set_title('end-on')
ax_endon.set_xlabel('Y')
ax_endon.set_ylabel('Z')

ax_topdown = fig.add_subplot(gs[0,2])
ax_topdown.imshow(np.transpose(ne[int(n_pts/2), :, :]), origin='lower', vmin=nemin, vmax=nemax) #constant y - top down
ax_topdown.set_title('topdown')
ax_topdown.set_xlabel('X')
ax_topdown.set_ylabel('Z')

z_start = nz
lineout, = ax_topdown.plot([0.0, n_pts], [z_start, z_start], 'w')

ax_transverse = fig.add_subplot(gs[1,2])
im_transverse = ax_transverse.imshow(ne[:, :, -1], origin='lower', vmin=nemin, vmax=nemax) #constant z - transverse
ax_transverse.set_title('transverse')
ax_transverse.set_xlabel('X')
ax_transverse.set_ylabel('Y')




class Index(object):
    ind = -1

    def next(self, event):
        self.ind += 1
        i = (self.ind * 10) % nz
        im2.set_data(shadow(Field[:,:,i]))  
        im_transverse.set_data(ne[:,:,i])
        im3.set_data(phase(Field[:,:,i]))
        ax_out.set_title('z = %.2f $\mu$m' % float(z[i]*1e6))
        lineout.set_ydata([i, i])
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = (self.ind * 10) % nz
        im2.set_data(shadow(Field[:,:,i]))
        im_transverse.set_data(ne[:,:,i])
        im3.set_data(phase(Field[:,:,i]))
        ax_out.set_title('z = %.2f $\mu$m' % float(z[i]*1e6))
        lineout.set_ydata([i, i])
        plt.draw()
        

callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
plt.grid()

#%%
#COMPARE WITH ROB'S PLOTS

#Rob's used a custom colormap here I can't quite copy
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.abs(inputBeam)**2, extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='Greys_r')
ax2.imshow(np.abs(Field[:, :, -1])**2, extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='Greys_r')
ax1.set_title('Intensity at z=0.0 mm')
ax2.set_title('Intensity at z=2.0 mm')
ax1.set_xlabel('x (mm)')
ax2.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')



fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.angle(inputBeam), extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='jet')
ax2.imshow(np.angle(Field[:, :, -1]), extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3], cmap='jet')
ax1.set_title('Phase at z=0.0 mm')
ax2.set_title('Phase at z=2.0 mm')


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


#%%
#RAY DIAGNOSTICS TO COMPARE - INTERFEROMETRY
const = (lamda * e**2)/(4.0*np.pi * me * epsilon * c**2)
phase_change = const * dz * np.cumsum(ne, axis=-1)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
fig.suptitle('Test Ray Interferometry')

ax1.set_title('$n_e$')
ax1.imshow(ne[:, 256, :], extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])

ax2.set_title('Accumulated $n_e$ - no deflection')
ax2.imshow(phase_change[:, 256, :], aspect=1.0, extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])


ax3.imshow(np.angle(Field[:, 256, :]), extent=[z.min()*1e3, z.max()*1e3, y.min()*1e3, y.max()*1e3])
ax3.set_title('Angle of propagated field')
ax3.set_xlabel('z (mm)')

ax1.set_ylabel('y (mm)')
ax2.set_ylabel('y (mm)')
ax3.set_ylabel('y (mm)')

