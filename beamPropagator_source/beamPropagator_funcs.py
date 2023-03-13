#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to completely copy of Rob's LAPPLAC code, from matlab to python.

A combination of many scripts enough to replicate the example transverse probing. 

Created on Wed Nov 20 23:49:59 2019

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

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


def setup_propagate(inputBeam, X, Y, z, eta, k, eta0=1.0, absorbingBoundaries=1):
    """get mathematical bits for propagating field
    
    expects eta to be shape (n_pts, n_pts, nz)
    """
    absorbingBoundaries = bool(absorbingBoundaries)
    x = X[0].copy()
    y = Y[:,0].copy()
    
    dz = np.mean(np.gradient(z))
    
    n_pts = x.size
    res = np.mean(np.gradient(x))

    eta = np.dstack((eta, eta[:,:,-1])) #Extend eta array by one to allow for stepped refraction calc.

    # Find Angular Spectrum of the pulse
    xFreq, yFreq, AS = fourierTransform(inputBeam,x,y,n_pts,n_pts,debug=0)
    XF, YF = np.meshgrid(xFreq, yFreq)
    
    # Calculate the free space propagator G for half steps in z - this is constant over whole sim
    p = np.sqrt((k**2)*eta0**2 - XF**2 - YF**2)
    G = np.exp(-1j*(p-k*eta0)*dz/2)
    
    mask = np.ones_like(X)
    if absorbingBoundaries==1:
        scaleFactor = 5/8 #WHAT IS THIS?
        power_factor = 10. # WHY SO LARGE?
        X, Y = np.meshgrid(x,y)
        mask = np.exp(-(2*((X**2 + Y**2)/((scaleFactor*n_pts*res)**2)))**power_factor)
        
    return AS, xFreq, yFreq, G, mask, X, Y, z, eta


def propagate_in_z(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k, eta0=1, verbose=True):
    """evaluate electric field at each point in z
    """
    
    XF, YF = np.meshgrid(xFreq, yFreq)
    n_pts = xFreq.size
    nz = z.size
    dz = np.mean(np.gradient(z))
    
    x = X[0].copy()
    y = Y[:,0].copy()
    
    Field = np.zeros((n_pts,n_pts,nz), dtype=np.complex64)
    
    for i in range(len(z)):
        if verbose:
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
        
        #assume there is always a mask, but if set up right it can do nothign
        AS = AS*mask
        # Add this field component to our array
        _, _, Field[:,:,i] = inverseFourierTransform(AS,xFreq,yFreq,n_pts,n_pts,0)

    return Field


def quick_free_propagate(AS, xFreq, yFreq, G, mask, X, Y, z, eta, k, eta0=1, verbose=True):
    """Jumps straight to final position - i.e. like a focus
    
    Only works for free-space propagation.
    """
    XF, YF = np.meshgrid(xFreq, yFreq)
    n_pts = xFreq.size
    
    # 30-7-21 changed line below from *2 to **2 as its the arg in the exp that is x2
    # now seems to be working
    GG = G.copy() ** 2.0
    
    #calculate how many masks would have been applied - not sure how much of a difference this makes
    n = 255
    GG *= mask**(n)
    
    AS_final = GG * AS
    _, _, Field_final = inverseFourierTransform(AS_final,xFreq,yFreq,n_pts,n_pts,0)
    
    return Field_final
