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

import numpy as np
from tabulate import tabulate


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
            If True, print the compensation details, by default True

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
            If True, print the compensation details, by default True

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
