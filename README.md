# LAPPLAC
Diffractive laser beam propagator.

**n.b. this is a python-translated version of the LAPPLAC code, originally written by Rob Shalloo (Uni of Oxford) in Matlab.**

The code takes an input initial electric field (amplitude and phase) and solves for its evolution through a given medium or refractive index, η.
The code is written in cartesian coordinates (x, y, z), and only considers monochromatic laser beams.


### Loading in the source code
All functions for using the code can be imported from the beamPropagator_source directory.\
Scripts in this repo do this by relative path reference, using the following\

~~~~
import sys
if '../' not in sys.path: sys.path.append('../')   
from beamPropagator_source import *
~~~~

## Requirements
The code was tested on 13/03/23, using the following versions:

* python 3.6.7
* numpy 1.17.3
* matplotlib 3.0.1
* scipy 1.3.2
* skimage 0.15.0
