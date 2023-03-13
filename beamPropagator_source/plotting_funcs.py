#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:03:05 2021

Created by Cary Colgan, cary.colgan13@imperial.ac.uk
"""
from .Field_funcs import *

from matplotlib.widgets import Button
import matplotlib.gridspec

import numpy as np

import matplotlib 
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
                    
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
        
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y)
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def explorer(Field, ne, z):
    """
    """
    plt.ion()
    n_pts = Field.shape[0]
    nz = Field.shape[-1]
    
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
    im3 = ax_out2.imshow(phase(Field[:,:,-1]), origin='lower',
                         cmap=matplotlib.cm.RdBu_r, 
                         norm=MidpointNormalize(midpoint=0.0,
                                                vmin=phase(Field[:,:,-1]).min(), vmax=phase(Field[:,:,-1]).max())) 
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
    
    buttons = [bnext, bprev]
    
    return buttons