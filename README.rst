Rotor Balancing
===============

A Python package for static and dynamic rotor balancing calculations based on the methodology described in "Static and Dynamic Balancing using portable measuring equipment" by John Vaughan (Brüel & Kjær Application Note).


Features
--------

* **VibrationVector**: Represents unbalance by means of a vector
* **MassVector**: Represents a mass placed at an angular position on a rotor
* **StaticBalancing**: Single-plane balancing for rotors
* **DynamicBalancing**: Two-plane balancing for rotors
* **MultiPlaneBalancing**: Balancing in any number of correction planes,
  with least-squares solution when there are more measurement points than planes
* **InfluenceCoefficients**: Influence coefficients as a reusable object:
  stored and loaded as plain data, corrections and trim corrections without
  trial runs, predicted vibration for the weights actually fitted, residual
  unbalance of a reading and a conditioning check
* **Angle convention**: angles counted with (default) or against the
  direction of rotation, with **convert_angle** and **convert_convention**
* **Balance tolerance**: permissible residual unbalance from a balance
  quality grade G, rotor mass and service speed, its allocation to two
  planes and a check of a residual against it
* **VibrationExtractor**: Extracts vibration vectors from time-domain signals
* **detect_rotation_freq**: Detects rotation frequency from a trigger signal
* Four pluggable extraction methods: **IQDemodulation** (default),
  **LeastSquaresFit**, **SynchronousAveraging**, **FFTExtraction**

Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install pyPRB

Or install from source:

.. code-block:: bash

    git clone https://github.com/ladisk/pyPRB.git
    cd pyPRB
    pip install .

For development installation:

.. code-block:: bash

    pip install -e .[dev]


Quick Start
-----------

Static Balancing Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyPRB import VibrationVector, MassVector, StaticBalancing

    # Initial measurements
    V_0 = VibrationVector(amplitude=3.4, phase=116)  # No trial mass
    V_1 = VibrationVector(amplitude=1.8, phase=42)   # With trial mass

    # Trial mass: 2.0 g mounted at 0° on the rotor
    trial_mass = MassVector(2.0, 0.0)

    # Create balancer
    balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)

    # Display measurement table
    print(balancer)

    # Compute compensation
    comp_mass = balancer.compute_compensation()
    # Output: Compensation mass: 2.01 g at position -30.8° on the rotor.


Dynamic Balancing Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyPRB import VibrationVector, MassVector, DynamicBalancing

    # Initial measurements (no trial masses)
    V_1_0 = VibrationVector(7.2, 238)   # Plane 1
    V_2_0 = VibrationVector(13.5, 296)  # Plane 2

    # With trial mass in Plane 1
    V_1_1 = VibrationVector(4.9, 114)   # Plane 1
    V_2_1 = VibrationVector(9.2, 347)   # Plane 2

    # With trial mass in Plane 2
    V_1_2 = VibrationVector(4.0, 79)    # Plane 1
    V_2_2 = VibrationVector(12.0, 292)  # Plane 2

    # Trial masses: 2.5 g each, mounted at 0° on their respective planes
    trial_mass_1 = MassVector(2.5, 0.0)
    trial_mass_2 = MassVector(2.5, 0.0)

    # Create balancer
    balancer = DynamicBalancing(
        V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2,
        trial_mass_1=trial_mass_1, trial_mass_2=trial_mass_2
    )

    # Display measurement table
    print(balancer)

    # Compute compensations for both planes
    comp_mass_1, comp_mass_2 = balancer.compute_compensation()
    # Output:
    # +---------+---------------------+------------+
    # |  Plane  |  Compensation Mass  |  Position  |
    # +=========+=====================+============+
    # |    1    |       2.95 g        |   50.2°    |
    # +---------+---------------------+------------+
    # |    2    |       2.84 g        |   -81.9°   |
    # +---------+---------------------+------------+


Multi-Plane Balancing Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The notebook ``examples/showcase.ipynb`` walks through all balancing methods:
static, dynamic and multi-plane (exact and least squares), from vibration
vectors and from time signals.

``MultiPlaneBalancing`` handles ``N`` correction planes and ``M >= N``
measurement points. Run ``j`` adds a trial mass to plane ``j``;
``V_trial[j][i]`` is the vibration at point ``i`` in that run.
For ``M > N`` the least-squares solution is returned.

.. code-block:: python

    from pyPRB import VibrationVector, MassVector, MultiPlaneBalancing

    # Initial run: one vector per measurement point
    V_0 = [VibrationVector(5.1, 40), VibrationVector(7.3, 115),
           VibrationVector(4.2, 260)]

    # One trial run per correction plane, each with all measurement points
    V_trial = [
        [VibrationVector(3.0, 75), VibrationVector(6.8, 128),
         VibrationVector(4.5, 250)],  # trial mass in plane 1
        [VibrationVector(5.6, 22), VibrationVector(4.1, 150),
         VibrationVector(3.9, 242)],  # trial mass in plane 2
        [VibrationVector(5.0, 48), VibrationVector(7.9, 104),
         VibrationVector(2.2, 300)],  # trial mass in plane 3
    ]
    trial_masses = [MassVector(2.0, 0.0), MassVector(2.0, 0.0),
                    MassVector(2.0, 0.0)]

    balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
    print(balancer)
    compensation = balancer.compute_compensation()  # list of MassVector
    residual = balancer.residual_vibration()        # predicted vibration


Trial Masses Left On and Stored Influence Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trial mass is often welded or clamped and stays on the rotor. With
``trial_left_on=True`` every trial run is compared with the run before it,
and ``keep_trial_masses=True`` gives the masses to add while the trial
masses stay. The influence coefficients can be kept for the next job on the
same machine (same speed, sensors and phase mark), which then needs only
one reading and no trial runs.

.. code-block:: python

    import json
    from pyPRB import (VibrationVector, MassVector, MultiPlaneBalancing,
                       InfluenceCoefficients)

    V_0 = [VibrationVector(7.2, 238), VibrationVector(13.5, 296)]
    V_trial = [
        [VibrationVector(4.9, 114), VibrationVector(9.2, 347)],   # mass 1 on
        [VibrationVector(6.1, 30), VibrationVector(8.0, 340)],    # masses 1 and 2 on
    ]
    trial_masses = [MassVector(2.5, 0.0), MassVector(2.5, 0.0)]

    balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                   trial_left_on=True)
    to_add = balancer.compute_compensation(keep_trial_masses=True)

    # Keep the coefficients for the next job
    coefficients = balancer.get_influence_coefficients()
    with open("machine.json", "w") as f:
        json.dump(coefficients.to_dict(), f)

    # Later: one check reading gives the trim correction, no trial runs
    with open("machine.json") as f:
        coefficients = InfluenceCoefficients.from_dict(json.load(f))
    check = [VibrationVector(0.9, 75), VibrationVector(1.4, 310)]
    trim = coefficients.compute_compensation(check)

    # Vibration to expect with the weights actually fitted
    fitted = [MassVector(0.3, 45), None]          # nothing in plane 2
    expected = coefficients.predict_vibration(check, fitted)

    # The reading as an equivalent unbalance in each plane
    heavy_spots = coefficients.residual_unbalance(check)

    coefficients.check_conditioning()   # warns if the planes are hard to separate


Angle Convention
~~~~~~~~~~~~~~~~

By default all angles are counted from the trigger/datum mark **in the
direction of rotation** (``angle_convention="with_rotation"``). Many
balancing instruments count **against** the rotation and show the phase as
a lag. Seen from the side on which the rotor turns counter-clockwise::

        "with_rotation"                 "against_rotation"

             90°                              270°
              |                                |
      180° ---+--- 0° (mark)          180° ---+--- 0° (mark)
              |                                |
             270°                             90°

        rotation: counter-clockwise in both pictures

A convention covers the phase readings and the weight positions together;
the result is in the convention of the input. The same physical case gives
the same place on the rotor either way:

.. code-block:: python

    from pyPRB import (VibrationVector, MassVector, StaticBalancing,
                       convert_angle, convert_convention)

    # Readings and trial position as an instrument shows them
    balancer = StaticBalancing(
        VibrationVector(3.4, 244), VibrationVector(1.8, 318),
        trial_mass=MassVector(2.0, 0.0),
        angle_convention="against_rotation")
    comp = balancer.compute_compensation()
    # Compensation mass: 2.01 g at position 30.8° on the rotor
    # (angles measured against the direction of rotation).

    convert_convention(comp, "against_rotation", "with_rotation")
    # MassVector(amplitude=2.01168, phase=-30.7888°, units='g')
    convert_angle(244, "against_rotation", "with_rotation")   # 116.0

``VibrationExtractor(..., angle_convention="against_rotation")`` returns
the phase lag, and ``InfluenceCoefficients.to_convention()`` converts
stored coefficients.


Balance Tolerance
~~~~~~~~~~~~~~~~~

How good is good enough? For a balance quality grade ``G`` (mm/s), rotor
mass ``M`` (kg) and service speed ``n`` (r/min) the permissible residual
unbalance is ``U_per = 9549 * G * M / n`` in g·mm.

.. code-block:: python

    from pyPRB import (permissible_unbalance, permissible_specific_unbalance,
                       permissible_mass, allocate_tolerance, check_residual,
                       MassVector)

    # 150 kg fan rotor, 1500 r/min, grade G 6.3
    U_per = permissible_unbalance(6.3, 150, 1500)        # 6016 g·mm
    e_per = permissible_specific_unbalance(6.3, 1500)    # 40.1 g·mm/kg (µm)
    m_per = permissible_mass(6.3, 150, 1500, radius=400) # 15.0 g at 400 mm

    # Share between two planes, 300 mm and 500 mm from the centre of mass
    U_1, U_2 = allocate_tolerance(U_per, distance_1=300, distance_2=500)
    # 3760 g·mm and 2256 g·mm (each share limited to 0.3 ... 0.7 of the total)

    # A residual of 4.2 g at a radius of 400 mm in plane 1
    check = check_residual(MassVector(4.2, 130), U_1, radius=400)
    check.within, check.ratio     # True, 0.45

The grade for a given machine and the rules for unusual rotors (for
example overhung ones) are to be taken from ISO 21940-11; the allocation
limits used by default are the commonly published ones and can be changed.


Extracting Vibration Vectors from Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyPRB import (
        VibrationExtractor, detect_rotation_freq,
        MassVector, StaticBalancing,
    )

    # Synthetic signals (B&K Example 2)
    sample_rate = 10000  # Hz
    f0_true = 25.0       # rotation frequency (Hz)
    duration = 5.0       # s
    t = np.arange(0, duration, 1 / sample_rate)
    rng = np.random.default_rng(0)

    trigger = np.sign(np.sin(2 * np.pi * f0_true * t))
    signal_run0 = (3.4 * np.cos(2 * np.pi * f0_true * t + np.deg2rad(116))
                   + 0.05 * rng.standard_normal(len(t)))
    signal_run1 = (1.8 * np.cos(2 * np.pi * f0_true * t + np.deg2rad(42))
                   + 0.05 * rng.standard_normal(len(t)))

    # 1. Detect rotation frequency from the trigger signal
    f0 = detect_rotation_freq(trigger, sample_rate=sample_rate, max_freq=50.0)

    # 2. Create extractor (default method: IQ demodulation)
    extractor = VibrationExtractor(sample_rate=sample_rate, rotation_freq=f0)

    # 3. Extract vibration vectors for each measurement run
    V_0 = extractor.get_vibration_vector(signal_run0, trigger)
    V_1 = extractor.get_vibration_vector(signal_run1, trigger)

    # 4. Use the vectors for balancing
    trial_mass = MassVector(2.0, 0.0)
    balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)
    comp_mass = balancer.compute_compensation()
    # Output: Compensation mass: 2.01 g at position -30.8° on the rotor.


Balancing procedure
-------------------

Static Balancing
~~~~~~~~~~~~~~~~

For a rotor with unbalance in a single plane:

1. Measure initial vibration vector **V₀**
2. Add known trial mass **m** and measure resulting vibration **V₁**
3. Compute compensation mass with `StaticBalancing`

Dynamic Balancing
~~~~~~~~~~~~~~~~~

For a rotor requiring two-plane balancing:

1. Measure initial vibration vectors in both planes: **V₁,₀** and **V₂,₀**
2. Add trial mass **m₁,₁** in plane 1, measure: **V₁,₁** and **V₂,₁**
3. Add trial mass **m₂,₂** in plane 2, measure: **V₁,₂** and **V₂,₂**
4. Compute compensation masses in both planes with `DynamicBalancing`