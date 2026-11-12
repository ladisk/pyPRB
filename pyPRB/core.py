"""
Core classes for rotor balancing calculations.

Based on "Static and Dynamic Balancing using portable measuring equipment"
by John Vaughan, Brüel & Kjær.

Angle convention
----------------
All angles follow the rotor datum convention: 0° is at the trigger/datum
mark on the rotor, and positive angles are measured in the direction of
rotation.
"""

import warnings

import numpy as np
from tabulate import tabulate


def _as_complex(vector) -> complex:
    """Return a VibrationVector or MassVector as a complex number."""
    return vector.amplitude * np.exp(1j * np.deg2rad(vector.phase))


def _mass_from_complex(z: complex, units: str):
    """Return a complex mass as a MassVector, or ``None`` for a zero mass."""
    amplitude = float(np.abs(z))
    if amplitude == 0.0:
        return None
    return MassVector(amplitude, float(np.rad2deg(np.angle(z))), units)


class VibrationVector:
    """Represents a vibration vector in 2D space.
    
    A vibration vector is characterized by an amplitude and phase angle,
    which can be used to represent unbalance in rotating machinery.
    
    Parameters
    ----------
    amplitude : float
        The magnitude of the vibration vector.
    phase : float
        The phase angle in degrees.

    Notes
    -----
    Angles use the rotor datum convention: 0° is at the trigger/datum mark,
    positive angles are in the direction of rotation.

    Examples
    --------
    >>> v1 = VibrationVector(10.5, 45)
    >>> v2 = VibrationVector(5.2, 135)
    >>> v3 = v1 + v2
    >>> print(v3)
    VibrationVector(amplitude=11.7171, phase=71.3464°)
    """

    def __init__(self, amplitude: float, phase: float):
        """Initialize the vibration vector with amplitude and phase (in degrees)."""
        if amplitude < 0:
            raise ValueError("Amplitude must be non-negative.")
        
        self.amplitude = amplitude
        self.phase = phase % 360
        self.complex_num = amplitude * np.exp(1j * np.deg2rad(self.phase))

    @classmethod
    def from_complex(cls, z: complex) -> 'VibrationVector':
        """Create a VibrationVector from a complex number.
        
        Parameters
        ----------
        z : complex
            Complex number representation of the vector.
            
        Returns
        -------
        VibrationVector
            A new VibrationVector instance.
        """
        amplitude = np.abs(z)
        phase = np.rad2deg(np.angle(z))
        return cls(amplitude, phase)

    def __repr__(self) -> str:
        return f'VibrationVector(amplitude={self.amplitude:g}, phase={self.phase:g}°)'

    def __add__(self, other: 'VibrationVector') -> 'VibrationVector':
        """Add two vibration vectors."""
        z1 = self.complex_num
        z2 = other.complex_num
        z_sum = z1 + z2
        return VibrationVector.from_complex(z_sum)

    def __sub__(self, other: 'VibrationVector') -> 'VibrationVector':
        """Subtract two vibration vectors."""
        z1 = self.complex_num
        z2 = other.complex_num
        z_diff = z1 - z2
        return VibrationVector.from_complex(z_diff)

    def __mul__(self, other: 'float | VibrationVector') -> 'VibrationVector':
        """Multiply vibration vector by scalar or another vector."""
        if isinstance(other, (int, float)):
            if other >= 0:
                return VibrationVector(self.amplitude * other, self.phase)
            elif other < 0:
                return VibrationVector(self.amplitude * (-other),
                                       self.phase + 180)
        elif isinstance(other, VibrationVector):
            z1 = self.complex_num
            z2 = other.complex_num
            z_prod = z1 * z2
            return VibrationVector.from_complex(z_prod)
        else:
            return NotImplemented

    def __rmul__(self, other: float) -> 'VibrationVector':
        """Right multiplication (allows scalar * vector)."""
        return self.__mul__(other)

    def __truediv__(self, other: 'float | VibrationVector') -> 'VibrationVector':
        """Divide vibration vector by scalar or another vector."""
        if isinstance(other, (int, float)):
            if other > 0:
                return VibrationVector(self.amplitude / other, self.phase)
            elif other < 0:
                return VibrationVector(self.amplitude / (-other),
                                       self.phase + 180)
            else:
                raise ZeroDivisionError("division by zero")
        elif isinstance(other, VibrationVector):
            z1 = self.complex_num
            z2 = other.complex_num
            z_div = z1 / z2
            return VibrationVector.from_complex(z_div)
        else:
            return NotImplemented

    def __neg__(self) -> 'VibrationVector':
        """Negate the vibration vector (180° phase shift)."""
        return VibrationVector(self.amplitude, self.phase + 180)


class MassVector:
    """Represents a mass vector with amplitude, angular position, and units.

    A mass vector describes a mass placed at a specific angular position
    on a rotor. Used to represent both trial masses and compensation masses
    in balancing calculations.

    Parameters
    ----------
    amplitude : float
        The mass value (must be non-negative).
    phase : float
        The angular position on the rotor in degrees.
    units : str, optional
        The units of the mass value, by default "g" (grams).
        For display only — no unit conversion is performed.

    Notes
    -----
    Phase represents the angular position on the rotor where the mass is
    (or should be) mounted. Angles use the rotor datum convention: 0° is at
    the trigger/datum mark, positive angles are in the direction of rotation.

    Examples
    --------
    >>> trial = MassVector(2.5, 45)
    >>> print(trial)
    MassVector(amplitude=2.5, phase=45°, units='g')

    >>> comp = MassVector(3.0, -30, units='oz')
    >>> print(comp)
    MassVector(amplitude=3, phase=-30°, units='oz')
    """

    def __init__(self, amplitude: float, phase: float, units: str = "g"):
        """Initialize the mass vector with amplitude, phase, and units."""
        if amplitude <= 0:
            raise ValueError("Amplitude must be positive.")
        if not isinstance(units, str):
            raise TypeError("Units must be a string.")

        self.amplitude = amplitude
        self.phase = phase % 360
        if self.phase > 180:
            self.phase -= 360
        self.units = units

    def __repr__(self) -> str:
        return (f"MassVector(amplitude={self.amplitude:g}, "
                f"phase={self.phase:g}°, units='{self.units}')")


class StaticBalancing:
    """Performs static (single-plane) balancing calculations.
    
    Static balancing is used when the rotor can be treated as having all
    its mass in a single plane, such as narrow disks, flywheels, or grindstones.
    
    Parameters
    ----------
    V_0 : VibrationVector
        Initial vibration measurement (no trial mass).
    V_1 : VibrationVector
        Vibration measurement with trial mass attached.
    trial_mass : MassVector
        Trial mass used, with amplitude (mass value) and phase (angular
        position on the rotor where it was mounted).

    Examples
    --------
    >>> V_0 = VibrationVector(3.4, 116)
    >>> V_1 = VibrationVector(1.8, 42)
    >>> trial_mass = MassVector(2.0, 0.0)
    >>> balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)
    >>> comp_mass = balancer.compute_compensation()
    Compensation mass: 2.01 g at position -30.8° on the rotor.
    """

    def __init__(self, V_0: VibrationVector, V_1: VibrationVector,
                 trial_mass: MassVector):
        self.V_0 = V_0
        self.V_1 = V_1
        self.trial_mass = trial_mass

    def __repr__(self) -> str:
        """Return a formatted table representation of the balancing data.
        """
        # Create data rows
        tm = self.trial_mass
        data = [[
            "None", f"{self.V_0.amplitude:g}", f"{self.V_0.phase:g}°", "V_0"
        ],
                [
                    f"{tm.amplitude:.1f} {tm.units} @ {tm.phase:g}°",
                    f"{self.V_1.amplitude:g}",
                    f"{self.V_1.phase:g}°", "V_1"
                ]]

        # Column headers
        headers = ["Trial Mass", "Amplitude", "Phase", "Vector"]

        # Generate table similar to B&K document
        table = tabulate(data,
                         headers=headers,
                         tablefmt="grid",
                         stralign="center",
                         numalign="center")

        # Add title
        title = "Static Balancing - Measured Vibration Data"
        separator = "=" * 80

        return f"\n{separator}\n{title}\n{separator}\n{table}\n{separator}"

    def _compute_V_T(self) -> VibrationVector:
        """Compute the effect of the trial mass.
        
        Returns
        -------
        VibrationVector
            The vibration vector caused by the trial mass alone.
        """
        V_T = self.V_1 - self.V_0
        self.V_T = V_T
        return V_T

    def compute_compensation(self, repr: bool = True) -> MassVector:
        """Calculate the compensation mass and its angular position.

        Parameters
        ----------
        repr : bool, optional
            If ``True``, print the compensation details.  Default is ``True``.

        Returns
        -------
        MassVector
            Compensating mass with absolute angular position on the rotor.
        """
        V_T = self._compute_V_T()
        compensation_mass = (self.V_0.amplitude / V_T.amplitude
                             * self.trial_mass.amplitude)
        relative_position = -V_T.phase + self.V_0.phase + 180
        absolute_position = relative_position + self.trial_mass.phase
        result = MassVector(compensation_mass, absolute_position,
                            self.trial_mass.units)
        if repr:
            print(
                f"Compensation mass: {result.amplitude:.2f} {result.units} "
                f"at position {result.phase:.1f}° on the rotor.")

        return result


class DynamicBalancing:
    """Performs dynamic (two-plane) balancing calculations.
    
    Dynamic balancing is required for rotors with significant length, where
    unbalance exists in multiple planes. This class implements the two-plane
    influence coefficient method.
    
    Parameters
    ----------
    V_1_0 : VibrationVector
        Plane 1 vibration, initial run (no trial masses).
    V_2_0 : VibrationVector
        Plane 2 vibration, initial run (no trial masses).
    V_1_1 : VibrationVector
        Plane 1 vibration with trial mass in plane 1.
    V_2_1 : VibrationVector
        Plane 2 vibration with trial mass in plane 1.
    V_1_2 : VibrationVector
        Plane 1 vibration with trial mass in plane 2.
    V_2_2 : VibrationVector
        Plane 2 vibration with trial mass in plane 2.
    trial_mass_1 : MassVector
        Trial mass used in plane 1, with amplitude (mass value) and phase
        (angular position on the rotor where it was mounted).
    trial_mass_2 : MassVector
        Trial mass used in plane 2, with amplitude (mass value) and phase
        (angular position on the rotor where it was mounted).

    Notes
    -----
    The naming convention V_i_j represents:
    - i: measurement plane (1 or 2)
    - j: run number (0=initial, 1=trial mass in plane 1, 2=trial mass in plane 2)

    Trial masses in both planes must use the same units. No unit conversion
    is performed.

    Examples
    --------
    >>> # Initial measurements
    >>> V_1_0 = VibrationVector(7.2, 238)
    >>> V_2_0 = VibrationVector(13.5, 296)
    >>> # With trial mass in plane 1
    >>> V_1_1 = VibrationVector(4.9, 114)
    >>> V_2_1 = VibrationVector(9.2, 347)
    >>> # With trial mass in plane 2
    >>> V_1_2 = VibrationVector(4.0, 79)
    >>> V_2_2 = VibrationVector(12.0, 292)
    >>>
    >>> trial_mass_1 = MassVector(2.5, 0.0)
    >>> trial_mass_2 = MassVector(2.5, 0.0)
    >>> balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
    ...                             V_1_2, V_2_2,
    ...                             trial_mass_1=trial_mass_1,
    ...                             trial_mass_2=trial_mass_2)
    >>> comp_mass_1, comp_mass_2 = balancer.compute_compensation(repr=False)
    >>> print(comp_mass_1)
    MassVector(amplitude=2.95138, phase=50.1889°, units='g')
    >>> print(comp_mass_2)
    MassVector(amplitude=2.84414, phase=-81.8841°, units='g')
    """

    def __init__(self, V_1_0: VibrationVector, V_2_0: VibrationVector,
                 V_1_1: VibrationVector, V_2_1: VibrationVector,
                 V_1_2: VibrationVector, V_2_2: VibrationVector,
                 trial_mass_1: MassVector, trial_mass_2: MassVector):
        if trial_mass_1.units != trial_mass_2.units:
            raise ValueError(
                f"Trial mass units must match: got '{trial_mass_1.units}' "
                f"and '{trial_mass_2.units}'. No unit conversion is performed.")
        self.V_1_0 = V_1_0
        self.V_2_0 = V_2_0
        self.V_1_1 = V_1_1
        self.V_2_1 = V_2_1
        self.V_1_2 = V_1_2
        self.V_2_2 = V_2_2
        self.trial_mass_1 = trial_mass_1
        self.trial_mass_2 = trial_mass_2

    def __repr__(self) -> str:
        """Return a formatted table representation of the balancing data.
        """
        # Create data rows
        tm1 = self.trial_mass_1
        tm2 = self.trial_mass_2
        data = [
            [
                "None", f"{self.V_1_0.amplitude:g}", f"{self.V_1_0.phase:g}°",
                "V_1_0", f"{self.V_2_0.amplitude:g}", f"{self.V_2_0.phase:g}°",
                "V_2_0"
            ],
            [
                f"{tm1.amplitude:g} {tm1.units} @ {tm1.phase:g}° on Plane 1",
                f"{self.V_1_1.amplitude:g}",
                f"{self.V_1_1.phase:g}°", "V_1_1", f"{self.V_2_1.amplitude:g}",
                f"{self.V_2_1.phase:g}°", "V_2_1"
            ],
            [
                f"{tm2.amplitude:g} {tm2.units} @ {tm2.phase:g}° on Plane 2",
                f"{self.V_1_2.amplitude:g}",
                f"{self.V_1_2.phase:g}°", "V_1_2", f"{self.V_2_2.amplitude:g}",
                f"{self.V_2_2.phase:g}°", "V_2_2"
            ],
        ]

        # Column headers
        headers = [
            "Trial Mass\nSize and Location", "Amplitude\n(Plane 1)",
            "Phase\n(Plane 1)", "Vector", "Amplitude\n(Plane 2)",
            "Phase\n(Plane 2)", "Vector"
        ]

        # Generate table similar to B&K document
        table = tabulate(data,
                         headers=headers,
                         tablefmt="grid",
                         stralign="center",
                         numalign="center")

        # Add title
        title = "Dynamic Balancing - Measured Vibration Data"
        separator = "=" * 80

        return f"\n{separator}\n{title}\n{separator}\n{table}\n{separator}"

    def _compute_Q1(self) -> VibrationVector:
        """Calculate the influence coefficient Q1 for plane 1.
        
        Returns
        -------
        VibrationVector
            The Q1 influence coefficient vector.
        """
        A = self.V_1_1 - self.V_1_0
        B = self.V_1_2 - self.V_1_0
        C = self.V_2_1 - self.V_2_0
        D = self.V_2_2 - self.V_2_0

        Q1 = (B * self.V_2_0 - D * self.V_1_0) / (A * D - B * C)
        return Q1

    def _compute_Q2(self) -> VibrationVector:
        """Calculate the influence coefficient Q2 for plane 2.
        
        Returns
        -------
        VibrationVector
            The Q2 influence coefficient vector.
        """
        A = self.V_1_1 - self.V_1_0
        B = self.V_1_2 - self.V_1_0
        C = self.V_2_1 - self.V_2_0
        D = self.V_2_2 - self.V_2_0

        Q2 = (-A * self.V_2_0 + C * self.V_1_0) / (A * D - B * C)
        return Q2

    def compute_compensation(self,
                             repr: bool = True
                            ) -> tuple[MassVector, MassVector]:
        """Calculate compensating masses and positions for both planes.

        Parameters
        ----------
        repr : bool, optional
            If ``True``, print the compensation details.  Default is ``True``.

        Returns
        -------
        comp_1 : MassVector
            Compensating mass for plane 1 with absolute angular position.
        comp_2 : MassVector
            Compensating mass for plane 2 with absolute angular position.
        """
        Q1 = self._compute_Q1()
        Q2 = self._compute_Q2()

        compensation_mass_1 = Q1.amplitude * self.trial_mass_1.amplitude
        position_1 = Q1.phase + self.trial_mass_1.phase
        comp_1 = MassVector(compensation_mass_1, position_1,
                            self.trial_mass_1.units)

        compensation_mass_2 = Q2.amplitude * self.trial_mass_2.amplitude
        position_2 = Q2.phase + self.trial_mass_2.phase
        comp_2 = MassVector(compensation_mass_2, position_2,
                            self.trial_mass_2.units)

        if repr:
            data = [["1", f"{comp_1.amplitude:.2f} {comp_1.units}",
                     f"{comp_1.phase:.1f}°"],
                    ["2", f"{comp_2.amplitude:.2f} {comp_2.units}",
                     f"{comp_2.phase:.1f}°"]]
            headers = ["Plane", "Compensation Mass", "Position"]
            table = tabulate(data,
                             headers=headers,
                             tablefmt="grid",
                             stralign="center",
                             numalign="center")
            print(table)

        return comp_1, comp_2


class IllConditionedWarning(UserWarning):
    """Warns that the influence coefficient matrix is ill-conditioned.

    The correction planes are then hard to tell apart with the given
    measurement points: small errors in the readings give large errors in
    the computed masses.
    """


class MultiPlaneBalancing:
    """Performs multi-plane balancing with the influence coefficient method.

    Generalises :class:`StaticBalancing` (one plane) and
    :class:`DynamicBalancing` (two planes) to an arbitrary number of
    correction planes ``N`` and measurement points ``M``.  Each run adds a
    trial mass to one correction plane; vibration is measured at all ``M``
    measurement points in every run.

    Parameters
    ----------
    V_0 : sequence of VibrationVector
        Initial run (no trial masses), one vector per measurement point
        (length ``M``).
    V_trial : sequence of sequences of VibrationVector
        Trial runs.  ``V_trial[j][i]`` is the vibration at measurement
        point ``i`` with the trial mass mounted in correction plane ``j``.
        Shape ``N`` x ``M``.
    trial_masses : sequence of MassVector
        Trial mass used in each correction plane (length ``N``), with
        amplitude (mass value) and phase (angular position on the rotor
        where it was mounted).
    trial_left_on : bool, optional
        If ``False`` (default), each trial mass is removed before the next
        trial run, so every trial run is compared with the initial run.
        If ``True``, every trial mass stays on the rotor for the following
        runs (run ``j`` has the trial masses of planes ``1 .. j`` mounted),
        so every trial run is compared with the run before it.
    max_condition_number : float or None, optional
        If the condition number of the influence coefficient matrix
        exceeds this value, :class:`IllConditionedWarning` is issued when
        the compensation is computed, by default 100.  ``None`` disables
        the check.

    Notes
    -----
    The influence coefficient of correction plane ``j`` on measurement
    point ``i`` is

    .. math::

        \\alpha_{ij} = \\frac{V_{ij} - V_{i0}}{T_j},

    where :math:`T_j` is the trial mass of plane ``j`` written as a complex
    number.  The compensation masses :math:`W` follow from

    .. math::

        \\boldsymbol{\\alpha} \\, \\mathbf{W} = -\\mathbf{V}_0 .

    For ``M == N`` the system is solved exactly; for ``M > N`` (more
    measurement points than correction planes) the least-squares solution
    is returned, which minimises the sum of squared residual vibration
    amplitudes.  ``M < N`` is not allowed, because the compensation masses
    are then not uniquely determined.

    With ``N == 1`` and ``M == 1`` the result equals
    :class:`StaticBalancing`; with ``N == M == 2`` it equals
    :class:`DynamicBalancing`.

    With ``trial_left_on=True`` the numerator is the change against the
    previous run, :math:`V_{ij} - V_{i,j-1}`.  The compensation masses are
    still the masses for the rotor *without* trial masses; call
    ``compute_compensation(keep_trial_masses=True)`` for the masses to add
    while the trial masses stay where they are.

    Trial masses in all planes must use the same units. No unit conversion
    is performed.

    The condition number (ratio of the largest to the smallest singular
    value of the influence coefficient matrix) shows how well the
    correction planes can be told apart: a relative error in the readings
    can grow by up to this factor in the compensation masses.  It is high
    when two planes have nearly the same effect on all measurement points,
    for example planes close together or measurement points that do not
    see the difference between them.  The default limit of 100 is a
    pragmatic choice, not a published one; the condition number also
    depends on the scaling of the measurement points.

    Examples
    --------
    Two-plane example 4 from the B&K application note:

    >>> V_0 = [VibrationVector(7.2, 238), VibrationVector(13.5, 296)]
    >>> V_trial = [
    ...     [VibrationVector(4.9, 114), VibrationVector(9.2, 347)],  # plane 1
    ...     [VibrationVector(4.0, 79), VibrationVector(12.0, 292)],  # plane 2
    ... ]
    >>> trial_masses = [MassVector(2.5, 0.0), MassVector(2.5, 0.0)]
    >>> balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
    >>> comp = balancer.compute_compensation(repr=False)
    >>> print(comp[0])
    MassVector(amplitude=2.95138, phase=50.1889°, units='g')
    >>> print(comp[1])
    MassVector(amplitude=2.84414, phase=-81.8841°, units='g')
    """

    def __init__(self, V_0, V_trial, trial_masses, trial_left_on=False,
                 max_condition_number=100.0):
        V_0 = list(V_0)
        V_trial = [list(run) for run in V_trial]
        trial_masses = list(trial_masses)

        n_sensors = len(V_0)
        n_planes = len(trial_masses)
        if n_planes == 0 or n_sensors == 0:
            raise ValueError(
                "At least one correction plane and one measurement point "
                "are required.")
        if len(V_trial) != n_planes:
            raise ValueError(
                f"Number of trial runs ({len(V_trial)}) must match the number "
                f"of trial masses ({n_planes}).")
        for j, run in enumerate(V_trial):
            if len(run) != n_sensors:
                raise ValueError(
                    f"Trial run {j + 1} has {len(run)} measurements, expected "
                    f"{n_sensors} (one per measurement point in V_0).")
        if n_sensors < n_planes:
            raise ValueError(
                f"Number of measurement points ({n_sensors}) must be at "
                f"least the number of correction planes ({n_planes}).")
        units = {tm.units for tm in trial_masses}
        if len(units) > 1:
            raise ValueError(
                f"Trial mass units must match: got {sorted(units)}. "
                "No unit conversion is performed.")

        if max_condition_number is not None and not max_condition_number > 0:
            raise ValueError("max_condition_number must be positive or None.")

        self.V_0 = V_0
        self.V_trial = V_trial
        self.trial_masses = trial_masses
        self.max_condition_number = max_condition_number
        self.n_planes = n_planes
        self.n_sensors = n_sensors
        self.trial_left_on = bool(trial_left_on)

    def __repr__(self) -> str:
        """Return a formatted table representation of the balancing data."""
        def cell(v):
            return f"{v.amplitude:g} @ {v.phase:g}°"

        data = [["None"] + [cell(v) for v in self.V_0]]
        for j, (tm, run) in enumerate(zip(self.trial_masses, self.V_trial)):
            data.append(
                [f"{tm.amplitude:g} {tm.units} @ {tm.phase:g}° on Plane {j + 1}"]
                + [cell(v) for v in run])
        headers = (["Trial Mass\nSize and Location"]
                   + [f"Vibration\n(Point {i + 1})"
                      for i in range(self.n_sensors)])
        table = tabulate(data,
                         headers=headers,
                         tablefmt="grid",
                         stralign="center",
                         numalign="center")
        title = "Multi-Plane Balancing - Measured Vibration Data"
        if self.trial_left_on:
            title += "\n(each trial mass left on for the following runs)"
        separator = "=" * 80
        return f"\n{separator}\n{title}\n{separator}\n{table}\n{separator}"

    @property
    def influence_coefficients(self) -> np.ndarray:
        """Influence coefficient matrix (``M`` x ``N``, complex).

        Element ``[i, j]`` is the vibration at measurement point ``i`` caused
        by a unit mass mounted at 0° in correction plane ``j``.
        """
        V_0 = np.array([v.complex_num for v in self.V_0])
        alpha = np.empty((self.n_sensors, self.n_planes), dtype=complex)
        V_before = V_0
        for j, (run, tm) in enumerate(zip(self.V_trial, self.trial_masses)):
            V_j = np.array([v.complex_num for v in run])
            T_j = tm.amplitude * np.exp(1j * np.deg2rad(tm.phase))
            alpha[:, j] = (V_j - V_before) / T_j
            if self.trial_left_on:
                V_before = V_j
        return alpha

    @property
    def condition_number(self) -> float:
        """Condition number of the influence coefficient matrix.

        Ratio of the largest to the smallest singular value; ``inf`` if
        the matrix is rank deficient.
        """
        return float(np.linalg.cond(self.influence_coefficients))

    def get_influence_coefficients(self, max_condition_number=100.0):
        """Return the influence coefficients as a reusable object.

        The returned :class:`InfluenceCoefficients` can be stored (see
        :meth:`InfluenceCoefficients.to_dict`) and used later on the same
        machine without new trial runs.

        Parameters
        ----------
        max_condition_number : float or None, optional
            Passed to :class:`InfluenceCoefficients`.

        Returns
        -------
        InfluenceCoefficients
            Influence coefficients of this set of runs, in the units of
            the trial masses.
        """
        return InfluenceCoefficients(
            self.influence_coefficients,
            units=self.trial_masses[0].units,
            max_condition_number=max_condition_number)

    def _solve(self) -> np.ndarray:
        """Return compensation masses as complex numbers (length ``N``).

        Raises ``ValueError`` if the matrix is rank deficient and warns
        (:class:`IllConditionedWarning`) if it is ill-conditioned.  Must
        be called directly from a public method, so that the warning
        points to the user's call.
        """
        alpha = self.influence_coefficients
        V_0 = np.array([v.complex_num for v in self.V_0])
        W, _, rank, sv = np.linalg.lstsq(alpha, -V_0, rcond=None)
        if rank < self.n_planes:
            raise ValueError(
                "Influence coefficient matrix is rank deficient: the trial "
                "runs do not distinguish all correction planes.")
        cond = sv[0] / sv[-1]
        if (self.max_condition_number is not None
                and cond > self.max_condition_number):
            warnings.warn(
                f"Influence coefficient matrix is ill-conditioned (condition "
                f"number {cond:.3g} > {self.max_condition_number:g}): small "
                "errors in the readings give large errors in the masses.",
                IllConditionedWarning, stacklevel=3)
        return W

    def compute_compensation(self, repr: bool = True,
                             keep_trial_masses: bool = False) -> list:
        """Calculate compensating masses and positions for all planes.

        Parameters
        ----------
        repr : bool, optional
            If ``True``, print the compensation details.  Default is ``True``.
        keep_trial_masses : bool, optional
            If ``False`` (default), the result is the compensation for the
            rotor without trial masses (all trial masses are removed).
            If ``True``, the result is what has to be added while all trial
            masses stay on the rotor; this requires ``trial_left_on=True``,
            because only then the last run was measured with all trial
            masses mounted.

        Returns
        -------
        list of MassVector
            Compensating mass for each correction plane (length ``N``) with
            absolute angular position on the rotor.  With
            ``keep_trial_masses=True`` an entry is ``None`` if nothing has to
            be added in that plane.

        Raises
        ------
        ValueError
            If the influence coefficient matrix is rank deficient.

        Warns
        -----
        IllConditionedWarning
            If the condition number exceeds ``max_condition_number``.
        """
        W = self._solve()
        units = self.trial_masses[0].units
        if keep_trial_masses:
            if not self.trial_left_on:
                raise ValueError(
                    "keep_trial_masses=True requires trial_left_on=True: the "
                    "trial masses were removed between the runs.")
            T = np.array([_as_complex(tm) for tm in self.trial_masses])
            result = [_mass_from_complex(w, units) for w in W - T]
        else:
            result = [MassVector(np.abs(w), np.rad2deg(np.angle(w)), units)
                      for w in W]

        if repr:
            data = [[f"{j + 1}", f"{m.amplitude:.2f} {m.units}",
                     f"{m.phase:.1f}°"] if m is not None
                    else [f"{j + 1}", f"0.00 {units}", "-"]
                    for j, m in enumerate(result)]
            headers = ["Plane", "Compensation Mass", "Position"]
            print(tabulate(data,
                           headers=headers,
                           tablefmt="grid",
                           stralign="center",
                           numalign="center"))

        return result

    def residual_vibration(self) -> list:
        """Predict the vibration after mounting the compensation masses.

        For ``M == N`` the prediction is zero (up to round-off); for
        ``M > N`` it is the least-squares residual.

        Returns
        -------
        list of VibrationVector
            Predicted vibration at each measurement point (length ``M``).

        Warns
        -----
        IllConditionedWarning
            If the condition number exceeds ``max_condition_number``.
        """
        V_0 = np.array([v.complex_num for v in self.V_0])
        residual = V_0 + self.influence_coefficients @ self._solve()
        return [VibrationVector.from_complex(z) for z in residual]


class InfluenceCoefficients:
    """Influence coefficients of a machine as a reusable object.

    The influence coefficient :math:`\\alpha_{ij}` is the vibration at
    measurement point ``i`` caused by a unit mass at 0° in correction plane
    ``j``.  Once known from trial runs, the coefficients describe the
    machine (at the same speed, with the same sensors and the same phase
    reference), so that a later balancing job needs no trial runs: one
    reading is enough for a correction.

    Parameters
    ----------
    matrix : array_like of complex
        Influence coefficient matrix, shape ``M`` x ``N`` (``M``
        measurement points, ``N`` correction planes, ``M >= N``).  A scalar
        is taken as one plane and one point.
    units : str, optional
        Mass units the coefficients refer to, by default "g".  Masses
        given to and returned by the methods use these units; no unit
        conversion is performed.
    max_condition_number : float or None, optional
        Condition number of the matrix above which
        :class:`IllConditionedWarning` is issued, by default 100.  The
        default is a pragmatic choice, not a published limit; ``None``
        switches the warning off.  A rank-deficient matrix always raises
        ``ValueError``.

    Attributes
    ----------
    matrix : numpy.ndarray
        The ``M`` x ``N`` complex matrix.
    n_sensors, n_planes : int
        ``M`` and ``N``.

    Notes
    -----
    With the readings :math:`\\mathbf{V}` taken in some state of the rotor,
    the masses :math:`\\mathbf{W}` to add in that state follow from

    .. math::

        \\boldsymbol{\\alpha} \\, \\mathbf{W} = -\\mathbf{V}

    (exactly for ``M == N``, in the least-squares sense for ``M > N``).
    This is the same equation for the first correction and for every trim
    correction from a later check reading.

    The coefficients are valid only for the conditions under which they
    were measured: the same machine and speed, the same measurement points
    and the same phase reference mark.

    Examples
    --------
    Coefficients from the two-plane example 4 of the B&K application note,
    stored as plain data and used again without trial runs:

    >>> V_0 = [VibrationVector(7.2, 238), VibrationVector(13.5, 296)]
    >>> V_trial = [
    ...     [VibrationVector(4.9, 114), VibrationVector(9.2, 347)],
    ...     [VibrationVector(4.0, 79), VibrationVector(12.0, 292)],
    ... ]
    >>> trial_masses = [MassVector(2.5, 0.0), MassVector(2.5, 0.0)]
    >>> coefficients = InfluenceCoefficients.from_trial_runs(
    ...     V_0, V_trial, trial_masses)
    >>> stored = coefficients.to_dict()          # e.g. json.dump(stored, f)
    >>> coefficients = InfluenceCoefficients.from_dict(stored)
    >>> comp = coefficients.compute_compensation(V_0)
    >>> print(comp[0])
    MassVector(amplitude=2.95138, phase=50.1889°, units='g')
    >>> print(comp[1])
    MassVector(amplitude=2.84414, phase=-81.8841°, units='g')

    Predicted vibration if only rounded masses are fitted:

    >>> fitted = [MassVector(3.0, 50), MassVector(3.0, -80)]
    >>> for v in coefficients.predict_vibration(V_0, fitted):
    ...     print(v)
    VibrationVector(amplitude=0.788518, phase=31.4623°)
    VibrationVector(amplitude=0.333033, phase=105.618°)
    """

    def __init__(self, matrix, units: str = "g",
                 max_condition_number: 'float | None' = 100.0):
        matrix = np.array(matrix, dtype=complex)
        if matrix.ndim == 0:
            matrix = matrix.reshape(1, 1)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError(
                "The influence coefficient matrix must be two-dimensional "
                "(measurement points x correction planes) and not empty.")
        if matrix.shape[0] < matrix.shape[1]:
            raise ValueError(
                f"Number of measurement points ({matrix.shape[0]}) must be "
                f"at least the number of correction planes "
                f"({matrix.shape[1]}).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Influence coefficients must be finite.")
        if not isinstance(units, str):
            raise TypeError("Units must be a string.")
        if max_condition_number is not None and max_condition_number <= 0:
            raise ValueError("max_condition_number must be positive or None.")

        self.matrix = matrix
        self.units = units
        self.max_condition_number = max_condition_number
        self.n_sensors, self.n_planes = matrix.shape

    def __repr__(self) -> str:
        return (f"InfluenceCoefficients(n_sensors={self.n_sensors}, "
                f"n_planes={self.n_planes}, units='{self.units}')")

    # -- construction and storage ------------------------------------------

    @classmethod
    def from_trial_runs(cls, V_0, V_trial, trial_masses,
                        trial_left_on: bool = False,
                        max_condition_number: 'float | None' = 100.0
                        ) -> 'InfluenceCoefficients':
        """Compute the coefficients from an initial run and trial runs.

        The arguments are those of :class:`MultiPlaneBalancing`; for one
        plane and one measurement point single objects may be given
        instead of sequences.

        Parameters
        ----------
        V_0 : VibrationVector or sequence of VibrationVector
            Initial run, one vector per measurement point.
        V_trial : sequence of sequences of VibrationVector
            ``V_trial[j][i]`` is the vibration at point ``i`` in the trial
            run of plane ``j``.  For one plane and one point a single
            VibrationVector.
        trial_masses : MassVector or sequence of MassVector
            Trial mass of each correction plane.
        trial_left_on : bool, optional
            ``True`` if every trial mass stayed on the rotor for the
            following runs.  Default is ``False``.
        max_condition_number : float or None, optional
            See the class description.

        Returns
        -------
        InfluenceCoefficients

        Examples
        --------
        One plane, example 2 of the B&K application note:

        >>> coefficients = InfluenceCoefficients.from_trial_runs(
        ...     VibrationVector(3.4, 116), VibrationVector(1.8, 42),
        ...     MassVector(2.0, 0.0))
        >>> print(coefficients.compute_compensation(VibrationVector(3.4, 116))[0])
        MassVector(amplitude=2.01168, phase=-30.7888°, units='g')
        """
        if isinstance(V_0, VibrationVector):
            V_0 = [V_0]
        if isinstance(trial_masses, MassVector):
            trial_masses = [trial_masses]
        if isinstance(V_trial, VibrationVector):
            V_trial = [[V_trial]]
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                       trial_left_on=trial_left_on)
        return balancer.get_influence_coefficients(
            max_condition_number=max_condition_number)

    @classmethod
    def from_amplitude_phase(cls, amplitude, phase, units: str = "g",
                             max_condition_number: 'float | None' = 100.0
                             ) -> 'InfluenceCoefficients':
        """Create the coefficients from arrays of amplitude and phase.

        Parameters
        ----------
        amplitude : array_like of float
            Vibration amplitude per unit mass, shape ``M`` x ``N``.
        phase : array_like of float
            Phase in degrees, shape ``M`` x ``N``.
        units : str, optional
            Mass units, by default "g".
        max_condition_number : float or None, optional
            See the class description.

        Returns
        -------
        InfluenceCoefficients
        """
        amplitude = np.asarray(amplitude, dtype=float)
        phase = np.asarray(phase, dtype=float)
        if amplitude.shape != phase.shape:
            raise ValueError(
                f"amplitude and phase must have the same shape: got "
                f"{amplitude.shape} and {phase.shape}.")
        if np.any(amplitude < 0):
            raise ValueError("Amplitudes must be non-negative.")
        return cls(amplitude * np.exp(1j * np.deg2rad(phase)), units=units,
                   max_condition_number=max_condition_number)

    def to_amplitude_phase(self) -> 'tuple[np.ndarray, np.ndarray]':
        """Return the coefficients as arrays of amplitude and phase.

        Returns
        -------
        amplitude : numpy.ndarray
            Vibration amplitude per unit mass, shape ``M`` x ``N``.
        phase : numpy.ndarray
            Phase in degrees in the range [0, 360), shape ``M`` x ``N``.
        """
        return np.abs(self.matrix), np.rad2deg(np.angle(self.matrix)) % 360

    def to_dict(self) -> dict:
        """Return the coefficients as a plain dictionary.

        The dictionary holds only lists, floats and strings, so it can be
        written to JSON or any other plain format and read again with
        :meth:`from_dict`.

        Returns
        -------
        dict
            Keys ``"amplitude"`` and ``"phase"`` (nested lists, ``M`` x
            ``N``, phase in degrees) and ``"units"``.
        """
        amplitude, phase = self.to_amplitude_phase()
        return {
            "amplitude": amplitude.tolist(),
            "phase": phase.tolist(),
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: dict,
                  max_condition_number: 'float | None' = 100.0
                  ) -> 'InfluenceCoefficients':
        """Create the coefficients from a dictionary made by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary with the keys ``"amplitude"``, ``"phase"`` and,
            optionally, ``"units"``.
        max_condition_number : float or None, optional
            See the class description.

        Returns
        -------
        InfluenceCoefficients
        """
        return cls.from_amplitude_phase(
            data["amplitude"], data["phase"], units=data.get("units", "g"),
            max_condition_number=max_condition_number)

    # -- checks --------------------------------------------------------------

    @property
    def condition_number(self) -> float:
        """Condition number of the matrix (largest / smallest singular value).

        1 means that the correction planes are perfectly separated by the
        measurement points; a large value means that two planes have
        nearly the same effect on all points.  The value depends on the
        scaling of the rows and columns (different sensors or radii).
        """
        s = np.linalg.svd(self.matrix, compute_uv=False)
        if s[-1] == 0:
            return float("inf")
        return float(s[0] / s[-1])

    def check_conditioning(self) -> float:
        """Check that the correction planes can be told apart.

        Returns
        -------
        float
            The condition number.

        Raises
        ------
        ValueError
            If the matrix is rank deficient (the trial runs do not
            distinguish all correction planes).

        Warns
        -----
        IllConditionedWarning
            If the condition number exceeds ``max_condition_number``.
        """
        if np.linalg.matrix_rank(self.matrix) < self.n_planes:
            raise ValueError(
                "Influence coefficient matrix is rank deficient: the trial "
                "runs do not distinguish all correction planes.")
        cond = self.condition_number
        if (self.max_condition_number is not None
                and cond > self.max_condition_number):
            warnings.warn(
                f"Influence coefficient matrix is ill-conditioned (condition "
                f"number {cond:.3g} > {self.max_condition_number:g}): small "
                "errors in the readings give large errors in the masses.",
                IllConditionedWarning, stacklevel=3)
        return cond

    # -- use -----------------------------------------------------------------

    def _readings(self, V) -> np.ndarray:
        """Return readings as a complex array of length ``M``."""
        if isinstance(V, VibrationVector):
            V = [V]
        V = list(V)
        if len(V) != self.n_sensors:
            raise ValueError(
                f"Expected {self.n_sensors} readings (one per measurement "
                f"point), got {len(V)}.")
        return np.array([v.complex_num for v in V])

    def _masses(self, masses) -> np.ndarray:
        """Return fitted masses as a complex array of length ``N``."""
        if isinstance(masses, MassVector) or masses is None:
            masses = [masses]
        masses = list(masses)
        if len(masses) != self.n_planes:
            raise ValueError(
                f"Expected masses for {self.n_planes} correction planes, "
                f"got {len(masses)}.")
        W = np.zeros(self.n_planes, dtype=complex)
        for j, plane in enumerate(masses):
            if plane is None:
                continue
            if isinstance(plane, MassVector):
                plane = [plane]
            for m in plane:
                if m.units != self.units:
                    raise ValueError(
                        f"Mass units must match the coefficients: got "
                        f"'{m.units}', expected '{self.units}'. "
                        "No unit conversion is performed.")
                W[j] += _as_complex(m)
        return W

    def _solve(self, rhs: np.ndarray) -> np.ndarray:
        self.check_conditioning()
        W, _, _, _ = np.linalg.lstsq(self.matrix, rhs, rcond=None)
        return W

    def compute_compensation(self, V) -> list:
        """Calculate the masses to add from one reading.

        The reading may be the initial run of a new job or a check run
        after a correction (a trim correction): the result is what has to
        be added to the rotor *as it was when the reading was taken*.
        No trial runs are needed.

        Parameters
        ----------
        V : VibrationVector or sequence of VibrationVector
            Vibration at each measurement point (length ``M``).

        Returns
        -------
        list of MassVector
            Mass to add in each correction plane (length ``N``).  An entry
            is ``None`` if nothing has to be added in that plane.

        Examples
        --------
        A trim correction: the check run after the first correction still
        shows some vibration.

        >>> coefficients = InfluenceCoefficients([[1.2 + 0.5j]])
        >>> check = VibrationVector(0.6, 200)
        >>> print(coefficients.compute_compensation(check)[0])
        MassVector(amplitude=0.461538, phase=-2.61986°, units='g')
        """
        W = self._solve(-self._readings(V))
        return [_mass_from_complex(w, self.units) for w in W]

    def predict_vibration(self, V, masses) -> list:
        """Predict the vibration after fitting a given set of masses.

        Unlike the ideal compensation, the masses actually fitted are
        usually rounded, split onto available positions or limited by
        what is at hand.  This gives the vibration to expect with them.

        Parameters
        ----------
        V : VibrationVector or sequence of VibrationVector
            Vibration at each measurement point before fitting the masses
            (length ``M``).
        masses : sequence
            What is added in each correction plane (length ``N``): a
            MassVector, a sequence of MassVector (several weights in one
            plane) or ``None`` (nothing).  A removed mass is a mass added
            at the opposite position.

        Returns
        -------
        list of VibrationVector
            Predicted vibration at each measurement point (length ``M``).
        """
        predicted = self._readings(V) + self.matrix @ self._masses(masses)
        return [VibrationVector.from_complex(z) for z in predicted]

    def residual_unbalance(self, V) -> list:
        """Express a reading as an equivalent unbalance in each plane.

        The result is the set of masses (one per correction plane, at the
        radius of the trial masses) that would cause the reading: the
        heavy spots that are still on the rotor.  Multiplied by the
        radius, the amplitudes can be compared with a balance tolerance.

        Parameters
        ----------
        V : VibrationVector or sequence of VibrationVector
            Vibration at each measurement point (length ``M``).

        Returns
        -------
        list of MassVector
            Equivalent unbalance mass in each correction plane (length
            ``N``), opposite to the compensation.  An entry is ``None`` for
            a plane without unbalance.

        Examples
        --------
        >>> coefficients = InfluenceCoefficients([[1.2 + 0.5j]])
        >>> print(coefficients.residual_unbalance(VibrationVector(0.6, 200))[0])
        MassVector(amplitude=0.461538, phase=177.38°, units='g')
        """
        W = self._solve(self._readings(V))
        return [_mass_from_complex(w, self.units) for w in W]
