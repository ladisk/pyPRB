"""Tests for pyPRB.tolerance (balance tolerance from a quality grade)."""

import numpy as np
import pytest

from pyPRB import (BALANCE_QUALITY_GRADES, InfluenceCoefficients, MassVector,
                   ToleranceCheck, VibrationVector, allocate_tolerance,
                   check_residual, permissible_mass,
                   permissible_specific_unbalance, permissible_unbalance)


class TestPermissibleUnbalance:
    """U_per = 9549 G M / n in g·mm (M in kg, n in r/min)."""

    def test_constant(self):
        """The constant is 30000/pi = 9549.3 (mm/s, r/min -> g·mm/kg)."""
        assert np.isclose(permissible_specific_unbalance(1.0, 1.0),
                          9549.2966, rtol=1e-7)
        # definition of the grade: e_per [mm] * omega [rad/s] = G [mm/s]
        e_per = permissible_specific_unbalance(2.5, 3000.0)   # µm
        omega = 2 * np.pi * 3000.0 / 60.0
        assert np.isclose(e_per * 1e-3 * omega, 2.5, rtol=1e-12)

    def test_published_fan_example(self):
        """Fan impeller 150 kg, 1500 r/min, G 6.3.

        Worked example published in a balancing-equipment vendor's guide
        to balance quality grades (Vibromera, "ISO 21940 balance quality
        grades"): e_per = 40.1 µm, U_per = 6015 g·mm.  Reproduced to 0.1 %
        (the published total is the rounded 40.1 times 150).
        """
        assert np.isclose(permissible_specific_unbalance(6.3, 1500), 40.1,
                          atol=0, rtol=1e-3)
        assert np.isclose(permissible_unbalance(6.3, 150, 1500), 6015,
                          atol=0, rtol=1e-3)

    def test_published_calculator_example(self):
        """Rotor 120 kg, 1480 r/min, G 6.3.

        Example of a public online balance quality calculator (Sensemore,
        "ISO 21940 balance quality calculator"): e_per = 40.65 g·mm/kg,
        hence U_per = 4878 g·mm.  Reproduced to 0.02 %.
        """
        assert np.isclose(permissible_specific_unbalance(6.3, 1480), 40.65,
                          atol=0, rtol=2e-4)
        assert np.isclose(permissible_unbalance(6.3, 120, 1480), 4878,
                          atol=0, rtol=2e-4)

    def test_published_total_of_two_planes(self):
        """Rotor 2.5 kg, 10 000 r/min, G 6.3: 15.0 g·mm in total.

        Values 8.2 g·mm and 6.8 g·mm for the two planes of such a rotor
        are quoted from Table E.2 of ISO 21940-11:2016/Amd 1:2022 (public
        preview); their sum is the total computed here (to 0.05 g·mm).
        The distances behind the two shares are not public, so the split
        itself is not reproduced.
        """
        assert np.isclose(permissible_unbalance(6.3, 2.5, 10000), 8.2 + 6.8,
                          atol=0.05, rtol=0)

    def test_half_speed_doubles_tolerance(self):
        assert np.isclose(permissible_unbalance(6.3, 120, 740),
                          2 * permissible_unbalance(6.3, 120, 1480))

    def test_as_mass_at_radius(self):
        U = permissible_unbalance(2.5, 25, 3000)
        assert np.isclose(permissible_mass(2.5, 25, 3000, radius=50), U / 50)
        assert np.isclose(U, 199, atol=0, rtol=2e-3)   # 7.96 µm x 25 kg

    def test_grades(self):
        """The series has steps of about 2.5 from 0.4 to 4000."""
        assert BALANCE_QUALITY_GRADES[0] == 0.4
        assert BALANCE_QUALITY_GRADES[-1] == 4000.0
        assert 6.3 in BALANCE_QUALITY_GRADES
        ratios = np.divide(BALANCE_QUALITY_GRADES[1:],
                           BALANCE_QUALITY_GRADES[:-1])
        assert np.allclose(ratios, 2.5, rtol=0.02)

    @pytest.mark.parametrize("args", [(0, 100, 1500), (6.3, 0, 1500),
                                      (6.3, 100, 0), (-1, 100, 1500)])
    def test_invalid(self, args):
        with pytest.raises(ValueError, match="must be positive"):
            permissible_unbalance(*args)
        with pytest.raises(ValueError, match="radius must be positive"):
            permissible_mass(6.3, 100, 1500, radius=0)


class TestAllocateTolerance:
    """Sharing the tolerance between two planes."""

    def test_symmetric(self):
        assert allocate_tolerance(100.0, 5, 5) == (50.0, 50.0)

    def test_proportional_rule_published(self):
        """Planes 10 in apart, centre of gravity 4 in from plane 1: 60/40.

        The rule (each plane's share inversely proportional to its
        distance from the centre of gravity) with a largest ratio of 2:1
        between the planes is given in MIL-STD-167-1A; with the centre of
        gravity 2 in from plane 1 the proportional 80/20 is limited to
        2/3 and 1/3.
        """
        U_1, U_2 = allocate_tolerance(1.0, 4, 6, limits=(1 / 3, 2 / 3))
        assert np.isclose(U_1, 0.6) and np.isclose(U_2, 0.4)
        U_1, U_2 = allocate_tolerance(1.0, 2, 8, limits=(1 / 3, 2 / 3))
        assert np.isclose(U_1, 2 / 3) and np.isclose(U_2, 1 / 3)

    def test_default_limits(self):
        """Commonly published limits 0.3 and 0.7 of the total."""
        U_1, U_2 = allocate_tolerance(200.0, 3, 7)
        assert np.isclose(U_1, 140.0) and np.isclose(U_2, 60.0)
        U_1, U_2 = allocate_tolerance(200.0, 1, 9)        # 0.9 / 0.1
        assert np.isclose(U_1, 140.0) and np.isclose(U_2, 60.0)
        U_1, U_2 = allocate_tolerance(200.0, 9, 1)
        assert np.isclose(U_1, 60.0) and np.isclose(U_2, 140.0)

    def test_no_limits(self):
        U_1, U_2 = allocate_tolerance(200.0, 1, 9, limits=None)
        assert np.isclose(U_1, 180.0) and np.isclose(U_2, 20.0)
        U_1, U_2 = allocate_tolerance(200.0, 0, 9, limits=None)
        assert np.isclose(U_1, 200.0) and U_2 == 0.0

    def test_sum_is_total(self):
        for d in (0.5, 2.0, 4.9, 7.3):
            assert np.isclose(sum(allocate_tolerance(15.04, d, 10 - d)),
                              15.04)

    def test_invalid(self):
        with pytest.raises(ValueError, match="between the two planes"):
            allocate_tolerance(100.0, -1, 5)
        with pytest.raises(ValueError, match="must not coincide"):
            allocate_tolerance(100.0, 0, 0)
        with pytest.raises(ValueError, match="permissible must be positive"):
            allocate_tolerance(0.0, 1, 1)
        with pytest.raises(ValueError, match="limits must satisfy"):
            allocate_tolerance(100.0, 1, 1, limits=(0.8, 0.2))
        with pytest.raises(ValueError, match="limits must satisfy"):
            allocate_tolerance(100.0, 1, 1, limits=(0.6, 0.8))
        with pytest.raises(ValueError, match="limits must satisfy"):
            allocate_tolerance(100.0, 1, 1, limits=(0.2, 1.2))

    @pytest.mark.parametrize("limits", [(0.2, 0.9), (0.1, 0.6), (0.0, 1.0),
                                        (0.5, 0.5), (1 / 3, 2 / 3)])
    @pytest.mark.parametrize("d_1", [0.1, 1.0, 5.0, 9.0, 9.9])
    def test_asymmetric_limits_keep_the_total(self, limits, d_1):
        """The shares never add up to more than the permissible value.

        Limits that are not symmetric around 0.5 used to be applied to
        each plane separately: (0.2, 0.9) with distances 1 and 19 gave
        90 + 20 = 110 % of the permissible unbalance.
        """
        U_1, U_2 = allocate_tolerance(100.0, d_1, 10.0 - d_1, limits=limits)
        assert np.isclose(U_1 + U_2, 100.0)
        low, high = limits
        for U in (U_1, U_2):
            assert low * 100 - 1e-9 <= U <= high * 100 + 1e-9


class TestCheckResidual:
    """Residual against the permissible value."""

    def test_within(self):
        check = check_residual(30.0, 40.0)
        assert isinstance(check, ToleranceCheck)
        assert check.within is True
        assert np.isclose(check.ratio, 0.75)
        assert np.isclose(check.margin, 10.0)
        assert check.residual == 30.0

    def test_limit_and_exceeded(self):
        assert check_residual(40.0, 40.0).within is True
        check = check_residual(50.0, 40.0)
        assert check.within is False
        assert np.isclose(check.ratio, 1.25)
        assert np.isclose(check.margin, -10.0)

    def test_mass_vector_and_none(self):
        check = check_residual(MassVector(0.5, 200), 40.0, radius=100.0)
        assert check.residual == 50.0 and check.within is False
        assert check_residual(None, 40.0).ratio == 0.0
        with pytest.raises(ValueError, match="radius is required"):
            check_residual(MassVector(0.5, 200), 40.0)

    def test_invalid(self):
        with pytest.raises(ValueError, match="non-negative"):
            check_residual(-1.0, 40.0)
        with pytest.raises(ValueError, match="permissible must be positive"):
            check_residual(1.0, 0.0)

    def test_synthetic_job_against_tolerance(self):
        """Known residual unbalance -> known verdict, through a reading.

        A 150 kg rotor at 1500 r/min, G 6.3, masses at 400 mm.  A
        residual of 20 g gives 8000 g·mm, more than the 6016 g·mm
        permitted; 10 g gives 4000 g·mm, which is within.
        """
        coefficients = InfluenceCoefficients([[0.05 * np.exp(0.6j)]])
        U_per = permissible_unbalance(6.3, 150, 1500)
        for grams, within in ((20.0, False), (10.0, True)):
            reading = VibrationVector.from_complex(
                coefficients.matrix[0, 0] * grams * np.exp(2.0j))
            residual = coefficients.residual_unbalance(reading)[0]
            assert np.isclose(residual.amplitude, grams)
            check = check_residual(residual, U_per, radius=400.0)
            assert check.within is within
            assert np.isclose(check.residual, grams * 400.0)
