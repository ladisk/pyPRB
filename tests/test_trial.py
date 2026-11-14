"""Tests for pyPRB.trial (trial mass suggestions, trial run assessment)."""

import numpy as np
import pytest

from pyPRB import (MassVector, StaticBalancing, TrialRunAssessment,
                   VibrationVector, assess_trial_run, centrifugal_force,
                   permissible_unbalance, trial_mass_from_force,
                   trial_mass_from_tolerance)

LB = 0.45359237      # kg
OZ = 28.349523125    # g
INCH = 25.4          # mm
LBF = 4.4482216153   # N


class TestCentrifugalForce:
    def test_definition(self):
        """F = m r omega^2 in SI units."""
        omega = 2 * np.pi * 3000 / 60
        assert np.isclose(centrifugal_force(10 * 200, 3000),
                          0.010 * 0.200 * omega ** 2)

    def test_published_inch_pound_form(self):
        """F [lbf] = 1.77 (rpm / 1000)^2 U [oz·in].

        The form in which the centrifugal force of an unbalance is given
        in the balancing literature (e.g. IRD Balancing technical papers).
        Reproduced to 0.5 % (the published constant is rounded).
        """
        for rpm in (900, 1800, 3600):
            force = centrifugal_force(1.0 * OZ * INCH, rpm) / LBF
            assert np.isclose(force, 1.77 * (rpm / 1000) ** 2, rtol=5e-3)

    def test_invalid(self):
        with pytest.raises(ValueError, match="speed must be positive"):
            centrifugal_force(1.0, 0)
        with pytest.raises(ValueError, match="non-negative"):
            centrifugal_force(-1.0, 100)


class TestTrialMassFromForce:
    def test_published_fan_example(self):
        """Fan rotor of 400 lb at 600 rpm, trial weight at 18 in: 3.48 oz.

        Worked example of a balancing-instrument maker's tutorial
        (Acoem, "Balancing How To #4 - Calculating a Trial Weight
        Mathematically"), which uses trial weight [oz] = 56 375.5 x rotor
        weight [lb] / (rpm^2 x radius [in]) for a force of 10 % of the
        rotor weight.  Reproduced to 0.2 %.
        """
        grams = trial_mass_from_force(400 * LB, 600, 18 * INCH)
        assert np.isclose(grams / OZ, 3.48, atol=0, rtol=2e-3)

    def test_constant(self):
        """U_T [g·mm] = 8.94e7 M [kg] / n^2 [r/min] for 10 %."""
        unbalance = trial_mass_from_force(1.0, 1.0, 1.0)
        assert np.isclose(unbalance, 8.94e7, rtol=1e-3)

    def test_force_is_the_fraction_of_the_weight(self):
        """The suggested mass gives exactly the requested force."""
        for fraction in (0.05, 0.1, 0.2):
            grams = trial_mass_from_force(250.0, 1480.0, 300.0,
                                          force_fraction=fraction)
            force = centrifugal_force(grams * 300.0, 1480.0)
            assert np.isclose(force, fraction * 250.0 * 9.80665)

    def test_scaling(self):
        base = trial_mass_from_force(100, 1500, 200)
        assert np.isclose(trial_mass_from_force(200, 1500, 200), 2 * base)
        assert np.isclose(trial_mass_from_force(100, 3000, 200), base / 4)
        assert np.isclose(trial_mass_from_force(100, 1500, 400), base / 2)

    @pytest.mark.parametrize("kwargs", [
        dict(rotor_mass=0, speed=1, radius=1),
        dict(rotor_mass=1, speed=0, radius=1),
        dict(rotor_mass=1, speed=1, radius=0),
        dict(rotor_mass=1, speed=1, radius=1, force_fraction=0)])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            trial_mass_from_force(**kwargs)


class TestTrialMassFromTolerance:
    def test_multiple_of_permissible_mass(self):
        """Five to ten times the permissible residual mass."""
        U_per = permissible_unbalance(6.3, 150, 1500)       # 6016 g·mm
        m_per = U_per / 400.0
        assert np.isclose(trial_mass_from_tolerance(U_per, 400.0), 5 * m_per)
        assert np.isclose(
            trial_mass_from_tolerance(U_per, 400.0, multiple=10), 10 * m_per)

    def test_invalid(self):
        with pytest.raises(ValueError, match="must be positive"):
            trial_mass_from_tolerance(0.0, 100.0)
        with pytest.raises(ValueError, match="must be positive"):
            trial_mass_from_tolerance(10.0, 100.0, multiple=0)


class TestAssessTrialRun:
    def test_published_example_is_adequate(self):
        """Example 2 of the B&K application note: 3.4 @ 116° -> 1.8 @ 42°.

        J. Vaughan, "Static and Dynamic Balancing using portable measuring
        equipment", Brüel & Kjær.  The amplitude changes by -47 % and the
        phase by -74°, so the trial run is adequate; the effect vector is
        the V_T used by StaticBalancing.
        """
        V_0, V_1 = VibrationVector(3.4, 116), VibrationVector(1.8, 42)
        a = assess_trial_run(V_0, V_1)
        assert isinstance(a, TrialRunAssessment)
        assert np.isclose(a.amplitude_change, 1.8 / 3.4 - 1)
        assert np.isclose(a.phase_change, -74.0)
        assert a.amplitude_ok and a.phase_ok and a.adequate
        balancer = StaticBalancing(V_0, V_1, MassVector(2.0, 0.0))
        V_T = balancer._compute_V_T()
        assert np.isclose(a.effect.complex_num, V_T.complex_num)
        assert np.isclose(a.effect_ratio, V_T.amplitude / 3.4)

    def test_amplitude_or_phase(self):
        """30 % or 30°: one of the two is enough."""
        V_0 = VibrationVector(10.0, 100.0)
        only_amplitude = assess_trial_run(V_0, VibrationVector(13.5, 105.0))
        assert only_amplitude.amplitude_ok and not only_amplitude.phase_ok
        assert only_amplitude.adequate
        only_phase = assess_trial_run(V_0, VibrationVector(10.5, 60.0))
        assert only_phase.phase_ok and not only_phase.amplitude_ok
        assert only_phase.adequate
        neither = assess_trial_run(V_0, VibrationVector(11.0, 110.0))
        assert not neither.amplitude_ok and not neither.phase_ok
        assert not neither.adequate
        assert np.isclose(neither.amplitude_change, 0.1)
        assert np.isclose(neither.phase_change, 10.0)

    def test_limits_are_parameters(self):
        V_0, V_1 = VibrationVector(10.0, 100.0), VibrationVector(11.0, 110.0)
        assert assess_trial_run(V_0, V_1, min_amplitude_change=0.1).adequate
        assert assess_trial_run(V_0, V_1, min_phase_change=10.0).adequate
        strict = assess_trial_run(VibrationVector(10.0, 0.0),
                                  VibrationVector(12.6, 26.0),
                                  min_amplitude_change=0.25,
                                  min_phase_change=25.0)
        assert strict.amplitude_ok and strict.phase_ok
        with pytest.raises(ValueError, match="non-negative"):
            assess_trial_run(V_0, V_1, min_phase_change=-1.0)

    def test_phase_wraps(self):
        a = assess_trial_run(VibrationVector(5.0, 350.0),
                             VibrationVector(5.0, 20.0))
        assert np.isclose(a.phase_change, 30.0)
        a = assess_trial_run(VibrationVector(5.0, 20.0),
                             VibrationVector(5.0, 350.0))
        assert np.isclose(a.phase_change, -30.0)
        a = assess_trial_run(VibrationVector(5.0, 0.0),
                             VibrationVector(5.0, 180.0))
        assert a.phase_change == 180.0

    def test_zero_initial_reading(self):
        a = assess_trial_run(VibrationVector(0.0, 0.0),
                             VibrationVector(2.0, 50.0))
        assert a.amplitude_change == float("inf")
        assert a.phase_change == 0.0
        assert a.adequate
        a = assess_trial_run(VibrationVector(0.0, 0.0),
                             VibrationVector(0.0, 0.0))
        assert a.amplitude_change == 0.0 and not a.adequate

    def test_synthetic_small_and_large_trial_mass(self):
        """Known unbalance and coefficient: effect ratio = trial / unbalance.

        With a heavy spot of 10 g, a 1 g trial mass in line with it
        changes the reading by 10 % (not adequate); a 4 g trial mass by
        40 % (adequate).  Both give the known correction of 10 g
        opposite the heavy spot.
        """
        alpha = 0.3 * np.exp(1j * np.deg2rad(25.0))
        heavy = 10.0 * np.exp(1j * np.deg2rad(70.0))
        V_0 = VibrationVector.from_complex(alpha * heavy)
        for grams, adequate in ((1.0, False), (4.0, True)):
            trial = MassVector(grams, 70.0)
            V_1 = VibrationVector.from_complex(
                alpha * (heavy + grams * np.exp(1j * np.deg2rad(70.0))))
            a = assess_trial_run(V_0, V_1)
            assert np.isclose(a.effect_ratio, grams / 10.0)
            assert np.isclose(a.amplitude_change, grams / 10.0)
            assert a.adequate is adequate
            comp = StaticBalancing(V_0, V_1, trial).compute_compensation(
                repr=False)
            assert np.isclose(comp.amplitude, 10.0)
            assert np.isclose(comp.phase, 70.0 - 180.0)
