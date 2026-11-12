"""
Tools for the weights that are actually fitted to a rotor.

The computed compensation is one mass at one angle.  On a real rotor the
weight often has to go to fixed positions (blades, bolt holes), to another
radius, be made of several pieces, be taken away instead of added, or be
rounded to the weights at hand.  The functions here do that arithmetic with
:class:`~pyPRB.MassVector` objects.

All functions work in either angle convention, as long as the positions
are counted the same way as the masses.  Where a result can be "no mass",
``None`` is returned in its place, because a :class:`~pyPRB.MassVector`
must have a positive mass.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .core import MassVector, _as_complex, _mass_from_complex

# Relative size below which a part of a split counts as zero.
_REL_TOL = 1e-12


def _as_list(masses) -> list:
    if isinstance(masses, MassVector):
        return [masses]
    return [m for m in masses if m is not None]


def _common_units(masses, default: str = "g") -> str:
    units = {m.units for m in masses}
    if len(units) > 1:
        raise ValueError(
            f"Mass units must match: got {sorted(units)}. "
            "No unit conversion is performed.")
    return units.pop() if units else default


def combine_masses(masses) -> 'MassVector | None':
    """Combine several masses in one plane into a single equivalent mass.

    Parameters
    ----------
    masses : sequence of MassVector
        Masses at the same radius in the same plane.  ``None`` entries are
        skipped.

    Returns
    -------
    MassVector or None
        The vector sum, or ``None`` if the masses cancel.

    Examples
    --------
    >>> from pyPRB import MassVector, combine_masses
    >>> print(combine_masses([MassVector(3.0, 0), MassVector(4.0, 90)]))
    MassVector(amplitude=5, phase=53.1301°, units='g')
    """
    masses = _as_list(masses)
    units = _common_units(masses)
    total = sum((_as_complex(m) for m in masses), 0j)
    scale = sum(m.amplitude for m in masses)
    if abs(total) <= _REL_TOL * scale:
        return None
    return _mass_from_complex(total, units)


def split_mass(mass: MassVector, angle_1: float, angle_2: float) -> tuple:
    """Split a mass onto two allowed angular positions.

    The two parts together are equivalent to the original mass:

    .. math::

        m_1 = m\\,\\frac{\\sin(\\theta_2 - \\theta)}{\\sin(\\theta_2 - \\theta_1)},
        \\qquad
        m_2 = m\\,\\frac{\\sin(\\theta - \\theta_1)}{\\sin(\\theta_2 - \\theta_1)} .

    Parameters
    ----------
    mass : MassVector
        The mass to replace (amplitude at angle :math:`\\theta`).
    angle_1, angle_2 : float
        The two allowed positions in degrees.  The position of ``mass``
        must lie between them, on the side where they are less than 180°
        apart.

    Returns
    -------
    part_1, part_2 : MassVector or None
        Masses at ``angle_1`` and ``angle_2``.  A part is ``None`` if the
        mass sits exactly on the other position.

    Raises
    ------
    ValueError
        If the two positions coincide or are opposite each other, or if
        the mass does not lie between them (a part would be negative).

    Examples
    --------
    >>> from pyPRB import MassVector, split_mass
    >>> part_1, part_2 = split_mass(MassVector(50.0, 35), 30, 60)
    >>> print(part_1)
    MassVector(amplitude=42.2618, phase=30°, units='g')
    >>> print(part_2)
    MassVector(amplitude=8.71557, phase=60°, units='g')
    """
    theta = np.deg2rad(mass.phase)
    a1, a2 = np.deg2rad(angle_1), np.deg2rad(angle_2)
    denominator = np.sin(a2 - a1)
    if abs(denominator) < 1e-9:
        raise ValueError(
            "The two positions must not coincide or be opposite each "
            f"other, got {angle_1:g}° and {angle_2:g}°.")
    m_1 = mass.amplitude * np.sin(a2 - theta) / denominator
    m_2 = mass.amplitude * np.sin(theta - a1) / denominator
    tol = 1e-9 * mass.amplitude
    if m_1 < -tol or m_2 < -tol:
        raise ValueError(
            f"The mass at {mass.phase:g}° does not lie between the "
            f"positions {angle_1:g}° and {angle_2:g}°: the split would "
            "need a negative mass.")
    part_1 = MassVector(m_1, angle_1, mass.units) if m_1 > tol else None
    part_2 = MassVector(m_2, angle_2, mass.units) if m_2 > tol else None
    return part_1, part_2


def split_mass_on_positions(mass: MassVector, n_positions: int,
                            first_position: float = 0.0) -> list:
    """Split a mass onto the two nearest of N equally spaced positions.

    Typical fixed positions are fan blades or the holes of a bolt circle.

    Parameters
    ----------
    mass : MassVector
        The mass to replace.
    n_positions : int
        Number of equally spaced positions on the rotor (at least 3).
    first_position : float, optional
        Angle of position 0 in degrees, by default 0.  The other positions
        follow in the direction of increasing angle.

    Returns
    -------
    list of (int, MassVector)
        Position number (0 to ``n_positions - 1``) and the mass to mount
        there.  Two entries in general; one if the mass sits exactly on a
        position.

    Examples
    --------
    A 12-blade fan with blade 0 at 0°, correction 50 g at 35°:

    >>> from pyPRB import MassVector, split_mass_on_positions
    >>> for blade, part in split_mass_on_positions(MassVector(50.0, 35), 12):
    ...     print(blade, part)
    1 MassVector(amplitude=42.2618, phase=30°, units='g')
    2 MassVector(amplitude=8.71557, phase=60°, units='g')
    """
    if not isinstance(n_positions, (int, np.integer)) or n_positions < 3:
        raise ValueError(
            f"n_positions must be an integer of at least 3, got "
            f"{n_positions!r}.")
    spacing = 360.0 / n_positions
    relative = (mass.phase - first_position) % 360.0
    lower = int(np.floor(relative / spacing + 1e-9)) % n_positions
    upper = (lower + 1) % n_positions
    parts = split_mass(
        MassVector(mass.amplitude, relative, mass.units),
        lower * spacing, lower * spacing + spacing)
    result = []
    for index, part in zip((lower, upper), parts):
        if part is not None:
            result.append((index, MassVector(
                part.amplitude, first_position + index * spacing,
                part.units)))
    return result


def change_radius(mass: MassVector, radius: float,
                  new_radius: float) -> MassVector:
    """Give the mass with the same effect at another radius.

    The unbalance ``mass * radius`` is kept: :math:`m_2 r_2 = m_1 r_1`.

    Parameters
    ----------
    mass : MassVector
        Mass at ``radius``.
    radius : float
        Radius the mass was computed for (e.g. that of the trial mass).
    new_radius : float
        Radius at which the mass will be mounted (same unit).

    Returns
    -------
    MassVector
        Mass at the same angle for ``new_radius``.

    Examples
    --------
    >>> from pyPRB import MassVector, change_radius
    >>> print(change_radius(MassVector(12.0, 40), radius=150, new_radius=200))
    MassVector(amplitude=9, phase=40°, units='g')
    """
    if not radius > 0 or not new_radius > 0:
        raise ValueError("Radii must be positive.")
    return MassVector(mass.amplitude * radius / new_radius, mass.phase,
                      mass.units)


def opposite_position(mass: MassVector) -> MassVector:
    """Turn a mass to add into a mass to remove, and back.

    Removing material has the effect of adding the same mass on the
    opposite side of the rotor.

    Parameters
    ----------
    mass : MassVector
        A mass to add (or to remove).

    Returns
    -------
    MassVector
        The same mass 180° away: where to remove it (or to add it).

    Examples
    --------
    >>> from pyPRB import MassVector, opposite_position
    >>> print(opposite_position(MassVector(2.0, -30)))     # drill here
    MassVector(amplitude=2, phase=150°, units='g')
    """
    return MassVector(mass.amplitude, mass.phase + 180.0, mass.units)


def mass_error(ideal: MassVector, fitted) -> 'MassVector | None':
    """Return the unbalance left by fitting other masses than the ideal one.

    Parameters
    ----------
    ideal : MassVector
        The computed compensation mass.
    fitted : MassVector or sequence of MassVector
        The mass or masses actually fitted in that plane.

    Returns
    -------
    MassVector or None
        The vector sum of the fitted masses minus the ideal mass: the
        equivalent mass that stays on the rotor.  ``None`` if the fitted
        masses equal the ideal one.

    Examples
    --------
    >>> from pyPRB import MassVector, mass_error
    >>> print(mass_error(MassVector(10.0, 0), MassVector(10.0, 15)))
    MassVector(amplitude=2.61052, phase=97.5°, units='g')
    """
    fitted = _as_list(fitted)
    units = _common_units(fitted + [ideal])
    total = sum((_as_complex(m) for m in fitted), 0j)
    error = total - _as_complex(ideal)
    scale = ideal.amplitude + sum(m.amplitude for m in fitted)
    if abs(error) <= _REL_TOL * scale:
        return None
    return _mass_from_complex(error, units)


class RoundedMass(NamedTuple):
    """Result of :func:`round_mass`."""

    #: The rounded mass at the same angle, or ``None`` if it rounds to zero.
    mass: 'MassVector | None'
    #: Rounded minus ideal mass as a vector (what stays on the rotor),
    #: or ``None`` if nothing was changed.
    error: 'MassVector | None'


def round_mass(mass: MassVector, step: float) -> RoundedMass:
    """Round a mass to the nearest multiple of a weight step.

    Parameters
    ----------
    mass : MassVector
        The ideal mass.
    step : float
        Smallest weight increment at hand, in the units of ``mass``.

    Returns
    -------
    RoundedMass
        Named tuple ``(mass, error)``: the rounded mass (``None`` if it
        rounds to zero) and the error vector, rounded minus ideal
        (``None`` if the mass is already a multiple of the step).

    Examples
    --------
    >>> from pyPRB import MassVector, round_mass
    >>> rounded = round_mass(MassVector(8.7, 60), step=2.0)
    >>> print(rounded.mass)
    MassVector(amplitude=8, phase=60°, units='g')
    >>> print(rounded.error)
    MassVector(amplitude=0.7, phase=-120°, units='g')
    """
    if not step > 0:
        raise ValueError(f"step must be positive, got {step!r}.")
    n_steps = int(np.floor(mass.amplitude / step + 0.5))
    amplitude = n_steps * step
    rounded = MassVector(amplitude, mass.phase, mass.units) if n_steps else None
    difference = amplitude - mass.amplitude
    if abs(difference) <= _REL_TOL * mass.amplitude:
        error = None
    elif difference > 0:
        error = MassVector(difference, mass.phase, mass.units)
    else:
        error = MassVector(-difference, mass.phase + 180.0, mass.units)
    return RoundedMass(mass=rounded, error=error)
