"""
Balance tolerance from a balance quality grade.

A balance quality grade ``G`` is the permissible product of the specific
unbalance and the angular speed, in mm/s.  The functions here turn a grade,
the rotor mass and the service speed into a permissible residual unbalance,
share it between two planes and compare a residual with it.

Units
-----
The functions use the units common in balancing practice and do not
convert:

==========================  =========================
grade ``G``                 mm/s
rotor mass                  kg
speed                       r/min (rpm)
unbalance                   g·mm
specific unbalance          g·mm/kg (equal to µm)
radius                      mm
correction mass             g
==========================  =========================

Notes
-----
The formula is the definition of the grade, :math:`G = e_{per}\\,\\Omega`,
written in practical units (IRD Balancing, Technical Paper 1, "Balance
Quality Requirements of Rigid Rotors"):

.. math::

    U_{per}\\,[\\mathrm{g\\,mm}] = \\frac{30000}{\\pi}\\,
        \\frac{G\\,[\\mathrm{mm/s}]\\; M\\,[\\mathrm{kg}]}{n\\,[\\mathrm{r/min}]}
        \\approx 9549\\,\\frac{G\\,M}{n} .

The grades and their use are specified in ISO 21940-11 (which replaced
ISO 1940-1).  This module does not reproduce the standard: which grade
applies to a machine, and the detailed rules for unusual rotors, must be
taken from the standard itself.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .core import MassVector

#: Standard series of balance quality grades in mm/s (steps of a factor 2.5).
BALANCE_QUALITY_GRADES = (0.4, 1.0, 2.5, 6.3, 16.0, 40.0, 100.0, 250.0,
                          630.0, 1600.0, 4000.0)

_FACTOR = 30000.0 / np.pi   # 9549.3: mm/s and r/min to g·mm/kg


def _positive(**values) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")


def permissible_specific_unbalance(grade: float, speed: float) -> float:
    """Permissible residual specific unbalance for a balance quality grade.

    Parameters
    ----------
    grade : float
        Balance quality grade ``G`` in mm/s (e.g. 6.3 for G 6.3).
    speed : float
        Service speed in r/min.  The highest service speed gives the
        tightest tolerance.

    Returns
    -------
    float
        Permissible specific unbalance :math:`e_{per}` in g·mm per kg of
        rotor mass.  The same number in µm is the permissible displacement
        of the centre of mass.

    Examples
    --------
    >>> from pyPRB import permissible_specific_unbalance
    >>> round(permissible_specific_unbalance(6.3, 1500), 2)
    40.11
    """
    _positive(grade=grade, speed=speed)
    return float(_FACTOR * grade / speed)


def permissible_unbalance(grade: float, rotor_mass: float,
                          speed: float) -> float:
    """Permissible residual unbalance of a rotor for a balance quality grade.

    Parameters
    ----------
    grade : float
        Balance quality grade ``G`` in mm/s (e.g. 6.3 for G 6.3).
    rotor_mass : float
        Mass of the rotor in kg.
    speed : float
        Service speed in r/min.

    Returns
    -------
    float
        Permissible residual unbalance :math:`U_{per}` of the whole rotor
        in g·mm (``9549 * grade * rotor_mass / speed``).

    Examples
    --------
    A 150 kg fan rotor at 1500 r/min balanced to G 6.3:

    >>> from pyPRB import permissible_unbalance
    >>> round(permissible_unbalance(6.3, 150, 1500))
    6016
    """
    _positive(rotor_mass=rotor_mass)
    return permissible_specific_unbalance(grade, speed) * rotor_mass


def permissible_mass(grade: float, rotor_mass: float, speed: float,
                     radius: float) -> float:
    """Permissible residual unbalance as a mass at a given radius.

    Parameters
    ----------
    grade : float
        Balance quality grade ``G`` in mm/s.
    rotor_mass : float
        Mass of the rotor in kg.
    speed : float
        Service speed in r/min.
    radius : float
        Radius at which correction masses are mounted, in mm.

    Returns
    -------
    float
        Mass in g that, at ``radius``, equals the permissible residual
        unbalance of the whole rotor.

    Examples
    --------
    >>> from pyPRB import permissible_mass
    >>> round(permissible_mass(6.3, 150, 1500, radius=400), 1)
    15.0
    """
    _positive(radius=radius)
    return permissible_unbalance(grade, rotor_mass, speed) / radius


def allocate_tolerance(permissible: float, distance_1: float,
                       distance_2: float,
                       limits: 'tuple[float, float] | None' = (0.3, 0.7)
                       ) -> 'tuple[float, float]':
    """Share a permissible residual unbalance between two planes.

    The plane closer to the centre of mass gets the larger share:

    .. math::

        U_1 = U_{per}\\,\\frac{L_2}{L_1 + L_2}, \\qquad
        U_2 = U_{per}\\,\\frac{L_1}{L_1 + L_2},

    where :math:`L_1` and :math:`L_2` are the distances of the planes from
    the centre of mass.  A symmetric rotor gets half in each plane.

    Parameters
    ----------
    permissible : float
        Permissible residual unbalance of the whole rotor in g·mm (see
        :func:`permissible_unbalance`); any other unit is passed through.
    distance_1 : float
        Distance from the centre of mass to plane 1 (any length unit).
    distance_2 : float
        Distance from the centre of mass to plane 2 (same unit).
    limits : tuple of float or None, optional
        Smallest and largest share of one plane, by default
        ``(0.3, 0.7)``; each share is limited to this range.  ``None``
        applies no limits.

    Returns
    -------
    U_1, U_2 : float
        Permissible residual unbalance in plane 1 and in plane 2.

    Raises
    ------
    ValueError
        If a distance is negative: the centre of mass must lie between the
        two planes.  Rotors with the centre of mass outside the planes
        (overhung rotors) follow other rules and are not covered.

    Notes
    -----
    The proportional rule and the limits of 0.3 and 0.7 for a rotor with
    the centre of mass between the planes are the ones commonly published
    in balancing literature for ISO 21940-11 (and ISO 1940-1 before it).
    They were taken from such secondary sources, not from the text of the
    standard; check the standard where it matters.  Other documents use
    other limits, e.g. a largest ratio of 2 to 1 between the planes in
    MIL-STD-167-1A, which is ``limits=(1/3, 2/3)``.

    Examples
    --------
    >>> from pyPRB import allocate_tolerance
    >>> U_1, U_2 = allocate_tolerance(100.0, distance_1=4, distance_2=6)
    >>> round(U_1, 6), round(U_2, 6)
    (60.0, 40.0)
    >>> U_1, U_2 = allocate_tolerance(100.0, distance_1=1, distance_2=9)
    >>> round(U_1, 6), round(U_2, 6)
    (70.0, 30.0)
    """
    _positive(permissible=permissible)
    if distance_1 < 0 or distance_2 < 0:
        raise ValueError(
            "Distances must be non-negative: the centre of mass must lie "
            "between the two planes. Overhung rotors are not covered.")
    total = distance_1 + distance_2
    if not total > 0:
        raise ValueError("The two planes must not coincide.")
    share_1 = distance_2 / total
    share_2 = distance_1 / total
    if limits is not None:
        low, high = limits
        if not 0 <= low <= high:
            raise ValueError(
                f"limits must satisfy 0 <= low <= high, got {limits!r}.")
        share_1 = min(max(share_1, low), high)
        share_2 = min(max(share_2, low), high)
    return float(permissible * share_1), float(permissible * share_2)


class ToleranceCheck(NamedTuple):
    """Result of :func:`check_residual`."""

    #: ``True`` if the residual does not exceed the permissible value.
    within: bool
    #: Residual divided by the permissible value (1.0 is the limit).
    ratio: float
    #: Permissible value minus the residual (negative when exceeded).
    margin: float
    #: The residual unbalance that was compared, in g·mm.
    residual: float


def check_residual(residual, permissible: float,
                   radius: 'float | None' = None) -> ToleranceCheck:
    """Compare a residual unbalance with the permissible value.

    The criterion is :math:`U_{res} \\le U_{per}`.  Only the amount
    counts, not the angle.  Measurement errors are not taken into account.

    Parameters
    ----------
    residual : float or MassVector
        Residual unbalance in g·mm, or a residual mass (for example from
        :meth:`pyPRB.InfluenceCoefficients.residual_unbalance`) together
        with ``radius``.  ``None`` counts as zero.
    permissible : float
        Permissible residual unbalance in g·mm for the same plane.
    radius : float, optional
        Radius of the residual mass in mm.  Required if ``residual`` is a
        MassVector, whose mass must be in g.

    Returns
    -------
    ToleranceCheck
        Named tuple ``(within, ratio, margin, residual)``.

    Examples
    --------
    >>> from pyPRB import MassVector, check_residual, permissible_unbalance
    >>> U_per = permissible_unbalance(6.3, 150, 1500)
    >>> check = check_residual(MassVector(9.0, 75), U_per, radius=400)
    >>> check.within, round(check.ratio, 2), round(check.margin)
    (True, 0.6, 2416)
    """
    _positive(permissible=permissible)
    if residual is None:
        residual = 0.0
    elif isinstance(residual, MassVector):
        if radius is None:
            raise ValueError(
                "radius is required when the residual is a MassVector.")
        _positive(radius=radius)
        residual = residual.amplitude * radius
    if residual < 0:
        raise ValueError("The residual unbalance must be non-negative.")
    residual = float(residual)
    return ToleranceCheck(within=bool(residual <= permissible),
                          ratio=residual / permissible,
                          margin=permissible - residual,
                          residual=residual)
