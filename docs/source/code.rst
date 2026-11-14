Code documentation
==================

This page provides detailed documentation for all classes and methods in the
rotor-balancing package.

Angle convention
----------------

.. automodule:: pyPRB.core
   :no-members:

.. autofunction:: pyPRB.convert_angle

.. autofunction:: pyPRB.convert_convention


VibrationVector
---------------

.. autoclass:: pyPRB.VibrationVector
   :members:
   :special-members: __init__, __add__, __sub__, __mul__, __truediv__, __repr__, __str__
   :undoc-members:
   :show-inheritance:


MassVector
----------

.. autoclass:: pyPRB.MassVector
   :members:
   :special-members: __init__, __repr__
   :undoc-members:
   :show-inheritance:


StaticBalancing
---------------

.. autoclass:: pyPRB.StaticBalancing
   :members:
   :special-members: __init__, __repr__
   :undoc-members:
   :show-inheritance:


DynamicBalancing
----------------

.. autoclass:: pyPRB.DynamicBalancing
   :members:
   :special-members: __init__, __repr__
   :undoc-members:
   :show-inheritance:


MultiPlaneBalancing
-------------------

.. autoclass:: pyPRB.MultiPlaneBalancing
   :members:
   :special-members: __init__, __repr__
   :undoc-members:
   :show-inheritance:


InfluenceCoefficients
---------------------

.. autoclass:: pyPRB.InfluenceCoefficients
   :members:
   :special-members: __init__, __repr__
   :undoc-members:
   :show-inheritance:

IllConditionedWarning
---------------------

.. autoclass:: pyPRB.IllConditionedWarning
   :show-inheritance:


Balance tolerance
=================

.. automodule:: pyPRB.tolerance
   :members:
   :undoc-members:
   :member-order: bysource


Weight tools
============

.. automodule:: pyPRB.weights
   :members:
   :undoc-members:
   :member-order: bysource


Trial mass
==========

.. automodule:: pyPRB.trial
   :members:
   :undoc-members:
   :member-order: bysource


Signal Processing
=================

Pipeline for extracting :class:`~pyPRB.VibrationVector` objects from
time-domain signals (response + trigger).

Typical workflow:

1. Detect the rotation frequency with :func:`~pyPRB.detect_rotation_freq`.
2. Create a :class:`~pyPRB.VibrationExtractor` (optionally with a custom
   :class:`~pyPRB.processing.ExtractionMethod`).
3. Call :meth:`~pyPRB.VibrationExtractor.get_vibration_vector` for each
   measurement run.

To judge a reading, :meth:`~pyPRB.VibrationExtractor.get_vector_scatter`
gives the vector with its scatter over the record,
:func:`~pyPRB.rotation_freq_variation` the speed and its variation, and
:meth:`~pyPRB.VibrationExtractor.get_harmonic_content` the content at 1×,
2×, 3× against the overall level.

detect_rotation_freq
--------------------

.. autofunction:: pyPRB.detect_rotation_freq


rotation_freq_variation
-----------------------

.. autofunction:: pyPRB.rotation_freq_variation

.. autoclass:: pyPRB.SpeedVariation
   :members:


VibrationExtractor
------------------

.. autoclass:: pyPRB.VibrationExtractor
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:


.. autoclass:: pyPRB.VectorScatter
   :members:

.. autoclass:: pyPRB.HarmonicContent
   :members:


ExtractionMethod
----------------

.. autoclass:: pyPRB.processing.ExtractionMethod
   :members:
   :undoc-members:
   :show-inheritance:


IQDemodulation
--------------

.. autoclass:: pyPRB.IQDemodulation
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:


LeastSquaresFit
---------------

.. autoclass:: pyPRB.LeastSquaresFit
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:


SynchronousAveraging
--------------------

.. autoclass:: pyPRB.SynchronousAveraging
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:


FFTExtraction
-------------

.. autoclass:: pyPRB.FFTExtraction
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance: