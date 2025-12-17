"""
Rotor Balancing Package
=======================

A Python package for static and dynamic rotor balancing calculations based on
the influence coefficient method described in "Static and Dynamic Balancing 
using portable measuring equipment" by John Vaughan, Brüel & Kjær.

Classes
-------
VibrationVector
    Represents vibration measurements as vectors with amplitude and phase.
    
StaticBalancing
    Performs single-plane (static) balancing calculations.
    
DynamicBalancing
    Performs two-plane (dynamic) balancing calculations.

Examples
--------
Static balancing:

>>> from pyPRB import VibrationVector, StaticBalancing
>>> V_0 = VibrationVector(3.4, 116)  # initial vibration vector
>>> V_1 = VibrationVector(1.8, 42)  # vibration vector after trial mass
>>> balancer = StaticBalancing(V_0, V_1, mass=2.0)
>>> compensation_mass, position = balancer.compute_compensation()

Dynamic balancing:

>>> from pyPRB import VibrationVector, DynamicBalancing
>>> V_1_0 = VibrationVector(7.2, 238)
>>> V_2_0 = VibrationVector(13.5, 296)
>>> V_1_1 = VibrationVector(4.9, 114)
>>> V_2_1 = VibrationVector(9.2, 347)
>>> V_1_2 = VibrationVector(4.0, 79)
>>> V_2_2 = VibrationVector(12.0, 292)
>>> balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
...                             V_1_2, V_2_2, mass_1_1=2.5, mass_2_2=2.5)
>>> results = balancer.compute_compensation()
"""

from .core import VibrationVector, StaticBalancing, DynamicBalancing
from .__version__ import __version__, __author__, __email__

# Define what gets imported with "from pyPRB import *"
__all__ = [
    'VibrationVector',
    'StaticBalancing', 
    'DynamicBalancing',
    '__version__',
]