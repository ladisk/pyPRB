Changelog
=========

1.1.0 (2026-11-14)
------------------

New features
~~~~~~~~~~~~

* ``MultiPlaneBalancing``: balancing in any number of correction planes with
  the influence coefficient method; a least-squares solution when there are
  more measurement points than planes, ``residual_vibration()`` and a
  warning (``IllConditionedWarning``) when the planes are hard to separate
  (#1, #2).
* Trial masses left on the rotor: ``trial_left_on=True`` and
  ``compute_compensation(keep_trial_masses=True)`` (#3, #9).
* ``InfluenceCoefficients``: influence coefficients as a reusable object,
  stored and loaded as plain data, corrections and trim corrections from one
  reading without trial runs, predicted vibration for the weights actually
  fitted, residual unbalance of a reading and a conditioning check (#3, #9).
* Angle convention: ``angle_convention="with_rotation"`` (default) or
  ``"against_rotation"`` for all balancing classes and the extractor, with
  ``convert_angle`` and ``convert_convention`` (#4, #9).
* Balance tolerance (``pyPRB.tolerance``): permissible residual unbalance
  from a balance quality grade, its allocation to two planes and a check of
  a residual against it (#5, #9).
* Weight tools (``pyPRB.weights``): split a correction onto fixed positions
  or two angles, combine weights, change the radius, opposite position,
  rounding to a weight step and the error that stays (#6, #9).
* Trial mass (``pyPRB.trial``): a suggested trial mass from the centrifugal
  force rule or from the tolerance, and an assessment of a trial run (#7, #9).
* Reading quality: ``VibrationExtractor.get_vector_scatter``,
  ``rotation_freq_variation`` and ``VibrationExtractor.get_harmonic_content``
  (#8, #9).
* ``edge`` option (``"rising"`` or ``"falling"``) for the extraction methods
  and ``detect_rotation_freq`` (#10, #12).

Changed behaviour
~~~~~~~~~~~~~~~~~

* All extraction methods now measure the phase from the trigger edge, located
  with sub-sample resolution and hysteresis. ``IQDemodulation``,
  ``LeastSquaresFit`` and ``FFTExtraction`` previously referred the phase to
  the fundamental of the trigger signal, which lies at the centre of the
  pulse, so their phases differ from version 1.0.0 by half the pulse width
  (90° for a square-wave trigger). All four methods now give the same phase,
  independent of the pulse width (#10, #11).
* ``detect_rotation_freq`` uses the trigger edges by default
  (``method="edges"``). The spectral method (``method="fft"``) now checks
  the sub-harmonics, so a short trigger pulse no longer gives twice the
  rotation frequency (#12, #13).

Other
~~~~~

* Python 3.9 is no longer supported; pyPRB requires Python 3.10 or newer.
* Showcase notebook ``examples/showcase.ipynb`` and example scripts for every
  feature.
* Continuous integration (tests, docstring examples, example scripts and the
  documentation build), a release workflow with trusted publishing to PyPI
  and a Read the Docs configuration.
* The license file is included in the source and wheel distributions.


1.0.0
-----

* First release: ``VibrationVector``, ``MassVector``, ``StaticBalancing``,
  ``DynamicBalancing`` and the extraction of vibration vectors from time
  signals (``VibrationExtractor`` with ``IQDemodulation``,
  ``LeastSquaresFit``, ``SynchronousAveraging`` and ``FFTExtraction``).
