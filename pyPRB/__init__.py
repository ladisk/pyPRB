"""
Rotor Balancing Package
=======================

A Python package for static and dynamic rotor balancing calculations based on
the influence coefficient method described in "Static and Dynamic Balancing
using portable measuring equipment" by John Vaughan, Bruel & Kjaer.

Classes
-------
VibrationVector
    Represents vibration measurements as vectors with amplitude and phase.

MassVector
    Represents a mass placed at an angular position on a rotor.

StaticBalancing
    Performs single-plane (static) balancing calculations.

DynamicBalancing
    Performs two-plane (dynamic) balancing calculations.

MultiPlaneBalancing
    Performs balancing in an arbitrary number of correction planes.

IllConditionedWarning
    Issued when the correction planes are hard to tell apart.

InfluenceCoefficients
    Stores influence coefficients for corrections without trial runs.

Examples
--------
Static balancing:

>>> from pyPRB import VibrationVector, MassVector, StaticBalancing
>>> V_0 = VibrationVector(3.4, 116)  # initial vibration vector
>>> V_1 = VibrationVector(1.8, 42)  # vibration vector after trial mass
>>> trial_mass = MassVector(2.0, 0.0)
>>> balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)
>>> comp_mass = balancer.compute_compensation()

Dynamic balancing:

>>> from pyPRB import VibrationVector, MassVector, DynamicBalancing
>>> V_1_0 = VibrationVector(7.2, 238)
>>> V_2_0 = VibrationVector(13.5, 296)
>>> V_1_1 = VibrationVector(4.9, 114)
>>> V_2_1 = VibrationVector(9.2, 347)
>>> V_1_2 = VibrationVector(4.0, 79)
>>> V_2_2 = VibrationVector(12.0, 292)
>>> trial_mass_1 = MassVector(2.5, 0.0)
>>> trial_mass_2 = MassVector(2.5, 0.0)
>>> balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
...                             V_1_2, V_2_2,
...                             trial_mass_1=trial_mass_1,
...                             trial_mass_2=trial_mass_2)
>>> comp_mass_1, comp_mass_2 = balancer.compute_compensation()
"""

from .core import (VibrationVector, MassVector, StaticBalancing,
                   DynamicBalancing, MultiPlaneBalancing,
                   InfluenceCoefficients, IllConditionedWarning)
from .core import convert_angle, convert_convention
from .tolerance import (BALANCE_QUALITY_GRADES, ToleranceCheck,
                        allocate_tolerance, check_residual, permissible_mass,
                        permissible_specific_unbalance, permissible_unbalance)
from .weights import (RoundedMass, change_radius, combine_masses, mass_error,
                      opposite_position, round_mass, split_mass,
                      split_mass_on_positions)
from .processing import VibrationExtractor, IQDemodulation, LeastSquaresFit, SynchronousAveraging, FFTExtraction, detect_rotation_freq
from .__version__ import __version__, __author__, __email__

# Define what gets imported with "from pyPRB import *"
__all__ = [
    'VibrationVector',
    'MassVector',
    'StaticBalancing',
    'DynamicBalancing',
    'MultiPlaneBalancing',
    'InfluenceCoefficients',
    'IllConditionedWarning',
    'convert_angle',
    'convert_convention',
    'BALANCE_QUALITY_GRADES',
    'ToleranceCheck',
    'allocate_tolerance',
    'check_residual',
    'permissible_mass',
    'permissible_specific_unbalance',
    'permissible_unbalance',
    'RoundedMass',
    'change_radius',
    'combine_masses',
    'mass_error',
    'opposite_position',
    'round_mass',
    'split_mass',
    'split_mass_on_positions',
    'VibrationExtractor',
    'IQDemodulation',
    'LeastSquaresFit',
    'SynchronousAveraging',
    'FFTExtraction',
    'detect_rotation_freq',
    '__version__',
]
