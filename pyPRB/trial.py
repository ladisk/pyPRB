"""
Choosing a trial mass and judging a trial run.

A trial mass must change the vibration enough to be measured, without
overloading the machine.  The suggestions here are **rules of thumb** from
balancing practice, not requirements of a standard; the person balancing
the machine decides.

Units: rotor mass in kg, speed in r/min, radius in mm, unbalance in g·mm,
trial mass in g, force in N.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .core import VibrationVector

_GRAVITY = 9.80665   # standard acceleration of gravity, m/s^2


def _positive(**values) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")


def centrifugal_force(unbalance: float, speed: float) -> float:
    """Centrifugal force of an unbalance at a given speed.

    Parameters
    ----------
    unbalance : float
        Unbalance (mass times radius) in g·mm.
    speed : float
        Rotor speed in r/min.

    Returns
    -------
    float
        Force :math:`F = U\\,\\omega^2` in N.

    Examples
    --------
    10 g at a radius of 200 mm, 3000 r/min:

    >>> from pyPRB import centrifugal_force
    >>> round(centrifugal_force(10 * 200, 3000), 1)
    197.4
    """
    _positive(speed=speed)
    if unbalance < 0:
        raise ValueError("unbalance must be non-negative.")
    omega = 2 * np.pi * speed / 60.0
    return float(unbalance * 1e-6 * omega ** 2)


def trial_mass_from_force(rotor_mass: float, speed: float, radius: float,
                          force_fraction: float = 0.1) -> float:
    """Suggest a trial mass from the centrifugal force it causes.

    Rule of thumb: the centrifugal force of the trial mass at the
    balancing speed should be a fraction (commonly 10 %) of the static
    rotor weight carried by a bearing,

    .. math::

        m_T\\, r\\, \\omega^2 = f\\, M\\, g .

    Parameters
    ----------
    rotor_mass : float
        Rotor mass in kg whose weight the force is compared with: the
        part of the rotor mass carried by the bearing next to the
        correction plane or, for a single plane, the whole rotor mass.
    speed : float
        Balancing speed in r/min.
    radius : float
        Radius at which the trial mass is mounted, in mm.
    force_fraction : float, optional
        Centrifugal force of the trial mass as a fraction of the rotor
        weight, by default 0.1.  A smaller fraction (e.g. 0.05) is often
        advised for high-speed machines.

    Returns
    -------
    float
        Suggested trial mass in g.

    Notes
    -----
    This is a rule of thumb, widely published in balancing instrument
    manuals and tutorials in the inch-pound form "trial weight [oz] =
    56 375 x rotor weight [lb] / (rpm² x radius [in])" for 10 %.  With
    :math:`f = 0.1` it reads :math:`U_T\\,[\\mathrm{g\\,mm}] \\approx
    8.94\\cdot10^{7}\\, M\\,[\\mathrm{kg}] / n^2\\,[\\mathrm{r/min}]`.
    At low speeds it gives very large masses; near a resonance even a
    small mass can cause a large response.  Start smaller when in doubt.

    Examples
    --------
    A 180 kg rotor at 600 r/min, trial mass at a radius of 450 mm:

    >>> from pyPRB import trial_mass_from_force
    >>> round(trial_mass_from_force(180, 600, 450), 1)
    99.4
    """
    _positive(rotor_mass=rotor_mass, speed=speed, radius=radius,
              force_fraction=force_fraction)
    omega = 2 * np.pi * speed / 60.0
    unbalance = force_fraction * rotor_mass * _GRAVITY / omega ** 2 * 1e6
    return float(unbalance / radius)


def trial_mass_from_tolerance(permissible: float, radius: float,
                              multiple: float = 5.0) -> float:
    """Suggest a trial mass as a multiple of the permissible residual mass.

    Rule of thumb from the Brüel & Kjær application note "Static and
    Dynamic Balancing using portable measuring equipment": a suitable
    trial mass is five to ten times the permissible residual mass.

    Parameters
    ----------
    permissible : float
        Permissible residual unbalance in g·mm for the plane (see
        :func:`pyPRB.permissible_unbalance`).
    radius : float
        Radius at which the trial mass is mounted, in mm.
    multiple : float, optional
        Trial unbalance as a multiple of the permissible one, by default
        5 (the lower end of the range of 5 to 10).

    Returns
    -------
    float
        Suggested trial mass in g.

    Examples
    --------
    >>> from pyPRB import trial_mass_from_tolerance
    >>> trial_mass_from_tolerance(permissible=6000.0, radius=400.0)
    75.0
    """
    _positive(permissible=permissible, radius=radius, multiple=multiple)
    return float(multiple * permissible / radius)


class TrialRunAssessment(NamedTuple):
    """Result of :func:`assess_trial_run`."""

    #: Change of the amplitude relative to the initial amplitude
    #: (0.25 is +25 %, -0.4 is -40 %); ``inf`` for a zero initial amplitude.
    amplitude_change: float
    #: Change of the phase in degrees, in the range (-180, 180].
    phase_change: float
    #: Effect of the trial mass, the vector ``V_1 - V_0``.
    effect: VibrationVector
    #: Amplitude of the effect relative to the initial amplitude.
    effect_ratio: float
    #: ``True`` if the amplitude changed by at least the given fraction.
    amplitude_ok: bool
    #: ``True`` if the phase changed by at least the given angle.
    phase_ok: bool
    #: ``True`` if the amplitude or the phase changed enough.
    adequate: bool


def assess_trial_run(V_0: VibrationVector, V_1: VibrationVector,
                     min_amplitude_change: float = 0.3,
                     min_phase_change: float = 30.0) -> TrialRunAssessment:
    """Assess whether a trial mass changed the vibration enough.

    If the trial mass changes the reading only a little, the influence
    coefficient is a small difference of two large numbers and the
    computed correction is unreliable.  The usual rule of thumb is the
    "30/30 rule": the amplitude should change by at least 30 % **or** the
    phase by at least 30°.  Other sources use 25 % and 25°; both limits
    are parameters.

    Parameters
    ----------
    V_0 : VibrationVector
        Reading before the trial mass was mounted.
    V_1 : VibrationVector
        Reading with the trial mass.
    min_amplitude_change : float, optional
        Smallest relative amplitude change that counts as enough, by
        default 0.3.
    min_phase_change : float, optional
        Smallest phase change in degrees that counts as enough, by
        default 30.

    Returns
    -------
    TrialRunAssessment
        Named tuple with the changes, the effect vector and the verdicts
        ``amplitude_ok``, ``phase_ok`` and ``adequate``.  No advice is
        given: if the run is not adequate, a larger trial mass or another
        position is the usual remedy.

    Examples
    --------
    Example 2 of the B&K application note (3.4 @ 116° to 1.8 @ 42°):

    >>> from pyPRB import VibrationVector, assess_trial_run
    >>> a = assess_trial_run(VibrationVector(3.4, 116), VibrationVector(1.8, 42))
    >>> round(a.amplitude_change, 3), round(a.phase_change, 1), a.adequate
    (-0.471, -74.0, True)
    >>> print(a.effect)
    VibrationVector(amplitude=3.38027, phase=326.789°)
    """
    if min_amplitude_change < 0 or min_phase_change < 0:
        raise ValueError("The limits must be non-negative.")
    effect = V_1 - V_0
    if V_0.amplitude > 0:
        amplitude_change = (V_1.amplitude - V_0.amplitude) / V_0.amplitude
        effect_ratio = effect.amplitude / V_0.amplitude
    else:
        amplitude_change = float("inf") if V_1.amplitude > 0 else 0.0
        effect_ratio = float("inf") if V_1.amplitude > 0 else 0.0
    if V_0.amplitude > 0 and V_1.amplitude > 0:
        phase_change = (V_1.phase - V_0.phase + 180.0) % 360.0 - 180.0
        if phase_change == -180.0:
            phase_change = 180.0
    else:
        phase_change = 0.0   # the phase of a zero vector has no meaning
    amplitude_ok = bool(abs(amplitude_change) >= min_amplitude_change)
    phase_ok = bool(abs(phase_change) >= min_phase_change)
    return TrialRunAssessment(
        amplitude_change=float(amplitude_change),
        phase_change=float(phase_change),
        effect=effect,
        effect_ratio=float(effect_ratio),
        amplitude_ok=amplitude_ok,
        phase_ok=phase_ok,
        adequate=amplitude_ok or phase_ok)
