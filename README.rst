Rotor Balancing
===============

A Python package for static and dynamic rotor balancing calculations based on the methodology described in "Static and Dynamic Balancing using portable measuring equipment" by John Vaughan (Brüel & Kjær Application Note).


Features
--------

* **VibrationVector**: Represents unbalance by means of a vector
* **MassVector**: Represents a mass placed at an angular position on a rotor
* **StaticBalancing**: Single-plane balancing for rotors
* **DynamicBalancing**: Two-plane balancing for rotors
* **VibrationExtractor**: Extracts vibration vectors from time-domain signals
* **SynchronousAveraging**: Revolution-locked averaging for fundamental extraction
* **detect_rotation_freq**: Detects rotation frequency from a trigger signal

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


Extracting Vibration Vectors from Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyPRB import (
        VibrationExtractor, SynchronousAveraging,
        detect_rotation_freq, MassVector, StaticBalancing,
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

    # 2. Create extractor (default method: synchronous averaging)
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