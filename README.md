# LAPPLAC
Diffractive laser beam propagator.

The simulation takes an input initial electric field (amplitude and phase) and solves its evolution through a given medium of refractive index, η.
The simulation is written in cartesian coordinates (x, y, z), and only considers monochromatic laser beams. It has no notion of time. 

For example, the code can be used to simulate:
* Propagation of arbitrary laser beam profiles.
* Effects of optical aberrations and distributed phase plates on laser focal spots.
* Diffraction of beams around sharp edges and obstacles.
* Imaging-based diagnostics, such as shadowgraphy and interferometry.
* Optical waveguides.

### FFT-BPM
The complex electric field is propagated using a split-step fast Fourier transform based beam propagation method (FFT-BPM).\
For more information on the underlying equations, see [1] or [2].\
Alternatively, for an in-depth description as to why the exponential of a Laplacian operator is mathematically equivalent to taking the Fourier transform, and why using a split-step scheme in δz provides higher order accuracy (~ δz<sup>3</sup>), see [3].

**The main workflow here is a python-translated version of the LAPPLAC code. This was originally written by Rob Shalloo (University of Oxford) in Matlab.**


### Loading in the source code
All functions for using the code can be imported from the beamPropagator_source directory.\
This can either be done by running the script from the root directory of this repo and importing the module, or by relative path reference using the following

~~~~
import sys
if '../' not in sys.path: sys.path.append('../') # navigate to root directory from script
from beamPropagator_source import *
~~~~

### Requirements
The code was tested on 13/03/23 on macOs 10.15.1, and on Windows 10 22H2, using the following versions (Windows versions in parantheses):

* python 3.6.7 (3.9.13)
* numpy 1.17.3 (1.21.5)
* matplotlib 3.0.1 (3.5.2)
* scipy 1.3.2 (1.9.1)
* skimage 0.15.0 (0.19.2)

### References
[[1] Shalloo, Hydrodynamic optical-field-ionized plasma waveguides for laser plasma accelerators, (2018), pp. 34 - 36](https://ora.ox.ac.uk/objects/uuid:aa7a03d0-2d64-423f-be42-40e01479d312)\
[[2] Colgan, Laser-plasma interactions as tools for studying processes in quantum electrodynamics, (2022), pp. 32 - 34](https://spiral.imperial.ac.uk/handle/10044/1/100927)\
[3] Kawana and Kitoh, Introduction to Optical Waveguide Analysis: Solving Maxwell's Equation and the Schrodinger Equation, John Wiley & Sons, (2004)
