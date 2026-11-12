"""Tests for pyPRB.weights (the weights actually fitted to the rotor)."""

import numpy as np
import pytest

from pyPRB import (InfluenceCoefficients, MassVector, RoundedMass,
                   VibrationVector, change_radius, combine_masses, mass_error,
                   opposite_position, round_mass, split_mass,
                   split_mass_on_positions)


def _c(mass):
    """MassVector (or None) as a complex number."""
    if mass is None:
        return 0j
    return mass.amplitude * np.exp(1j * np.deg2rad(mass.phase))


class TestCombine:
    def test_right_angle(self):
        total = combine_masses([MassVector(3.0, 0), MassVector(4.0, 90)])
        assert np.isclose(total.amplitude, 5.0)
        assert np.isclose(total.phase, np.rad2deg(np.arctan2(4, 3)))

    def test_cancel_and_skip_none(self):
        assert combine_masses([MassVector(2.0, 10),
                               MassVector(2.0, 190)]) is None
        assert combine_masses([]) is None
        total = combine_masses([MassVector(2.0, 10), None])
        assert np.isclose(total.amplitude, 2.0) and np.isclose(total.phase, 10)
        assert np.isclose(combine_masses(MassVector(2.0, 10)).amplitude, 2.0)

    def test_units(self):
        assert combine_masses([MassVector(1, 0, "oz"),
                               MassVector(1, 90, "oz")]).units == "oz"
        with pytest.raises(ValueError, match="units must match"):
            combine_masses([MassVector(1, 0, "oz"), MassVector(1, 90, "g")])


class TestSplit:
    def test_published_fan_example(self):
        """12-blade fan, 50 g required at 35°, blades at 30° and 60°.

        Worked example "A two-way split on a fan" in a balancing-equipment
        vendor's guide to split correction (Vibromera, "Split Correction
        in Rotor Balancing"): 42.3 g at 30° and 8.7 g at 60°.
        Reproduced to 0.05 g.
        """
        part_1, part_2 = split_mass(MassVector(50.0, 35.0), 30.0, 60.0)
        assert np.isclose(part_1.amplitude, 42.3, atol=0.05, rtol=0)
        assert np.isclose(part_2.amplitude, 8.7, atol=0.05, rtol=0)
        assert part_1.phase == 30.0 and part_2.phase == 60.0

        on_blades = split_mass_on_positions(MassVector(50.0, 35.0), 12)
        assert [blade for blade, _ in on_blades] == [1, 2]
        assert np.isclose(on_blades[0][1].amplitude, 42.3, atol=0.05, rtol=0)
        assert np.isclose(on_blades[1][1].amplitude, 8.7, atol=0.05, rtol=0)

    def test_symmetric_split(self):
        """Midway between positions 2*beta apart: m / (2 cos beta) each."""
        part_1, part_2 = split_mass(MassVector(100.0, 45.0), 30.0, 60.0)
        expected = 100.0 / (2 * np.cos(np.deg2rad(15.0)))
        assert np.isclose(part_1.amplitude, expected)
        assert np.isclose(part_2.amplitude, expected)

    @pytest.mark.parametrize("phase,a1,a2", [
        (35.0, 30.0, 60.0), (-170.0, 170.0, -150.0), (10.0, 80.0, -40.0),
        (200.0, 150.0, 260.0), (5.0, 350.0, 20.0)])
    def test_parts_recombine(self, phase, a1, a2):
        """The parts add up to the original mass (also across ±180°)."""
        mass = MassVector(7.5, phase, units="oz")
        parts = split_mass(mass, a1, a2)
        assert np.isclose(_c(parts[0]) + _c(parts[1]), _c(mass))
        assert all(p.units == "oz" for p in parts)
        assert np.isclose(_c(combine_masses(parts)), _c(mass))

    def test_on_a_position(self):
        part_1, part_2 = split_mass(MassVector(4.0, 30.0), 30.0, 75.0)
        assert part_2 is None
        assert part_1.amplitude == 4.0 and part_1.phase == 30.0
        part_1, part_2 = split_mass(MassVector(4.0, 75.0), 30.0, 75.0)
        assert part_1 is None and np.isclose(part_2.amplitude, 4.0)

    def test_not_between(self):
        with pytest.raises(ValueError, match="does not lie between"):
            split_mass(MassVector(4.0, 90.0), 0.0, 60.0)
        with pytest.raises(ValueError, match="does not lie between"):
            split_mass(MassVector(4.0, 180.0), 0.0, 60.0)

    def test_degenerate_positions(self):
        with pytest.raises(ValueError, match="coincide or be opposite"):
            split_mass(MassVector(4.0, 10.0), 20.0, 20.0)
        with pytest.raises(ValueError, match="coincide or be opposite"):
            split_mass(MassVector(4.0, 10.0), 0.0, 180.0)

    @pytest.mark.parametrize("n,first", [(3, 0.0), (5, 12.0), (8, -20.0),
                                         (12, 0.0), (36, 95.0)])
    def test_positions_recombine(self, n, first):
        """Any angle: two neighbouring positions, sum equals the mass."""
        spacing = 360.0 / n
        for phase in np.arange(-180.0, 180.0, 17.3):
            mass = MassVector(3.0, phase)
            parts = split_mass_on_positions(mass, n, first_position=first)
            assert 1 <= len(parts) <= 2
            assert np.isclose(sum(_c(p) for _, p in parts), _c(mass))
            for index, part in parts:
                assert 0 <= index < n
                offset = (part.phase - first - index * spacing) % 360
                assert min(offset, 360 - offset) < 1e-9
            if len(parts) == 2:
                assert (parts[1][0] - parts[0][0]) % n == 1

    def test_positions_exactly_on_one(self):
        parts = split_mass_on_positions(MassVector(3.0, 90.0), 8)
        assert len(parts) == 1
        assert parts[0][0] == 2
        assert np.isclose(parts[0][1].amplitude, 3.0)
        parts = split_mass_on_positions(MassVector(3.0, -45.0), 8)
        assert [i for i, _ in parts] == [7]

    def test_positions_wrap(self):
        """Between the last position and position 0."""
        parts = split_mass_on_positions(MassVector(3.0, -10.0), 4)
        assert [i for i, _ in parts] == [3, 0]

    def test_positions_invalid(self):
        with pytest.raises(ValueError, match="at least 3"):
            split_mass_on_positions(MassVector(3.0, 0.0), 2)
        with pytest.raises(ValueError, match="at least 3"):
            split_mass_on_positions(MassVector(3.0, 0.0), 4.5)


class TestRadiusAndRemoval:
    def test_change_radius(self):
        """m2 r2 = m1 r1."""
        new = change_radius(MassVector(12.0, 40.0, "oz"), 150.0, 200.0)
        assert np.isclose(new.amplitude * 200.0, 12.0 * 150.0)
        assert new.phase == 40.0 and new.units == "oz"
        back = change_radius(new, 200.0, 150.0)
        assert np.isclose(back.amplitude, 12.0)
        with pytest.raises(ValueError, match="Radii must be positive"):
            change_radius(new, 0.0, 150.0)

    def test_opposite_position(self):
        remove = opposite_position(MassVector(2.0, -30.0))
        assert remove.amplitude == 2.0 and remove.phase == 150.0
        again = opposite_position(remove)
        assert again.phase == -30.0
        # adding m at theta equals removing m at theta + 180°
        assert np.isclose(_c(MassVector(2.0, -30.0)), -_c(remove))


class TestRounding:
    def test_round_down_and_up(self):
        rounded = round_mass(MassVector(8.7, 60.0), step=2.0)
        assert isinstance(rounded, RoundedMass)
        assert rounded.mass.amplitude == 8.0 and rounded.mass.phase == 60.0
        assert np.isclose(rounded.error.amplitude, 0.7)
        assert np.isclose(rounded.error.phase, -120.0)
        rounded = round_mass(MassVector(9.2, 60.0), step=2.0)
        assert rounded.mass.amplitude == 10.0
        assert np.isclose(rounded.error.amplitude, 0.8)
        assert np.isclose(rounded.error.phase, 60.0)

    def test_error_is_rounded_minus_ideal(self):
        ideal = MassVector(42.26, 30.0)
        rounded = round_mass(ideal, step=5.0)
        assert np.isclose(_c(rounded.error), _c(rounded.mass) - _c(ideal))
        assert np.isclose(_c(mass_error(ideal, rounded.mass)),
                          _c(rounded.error))

    def test_exact_and_zero(self):
        rounded = round_mass(MassVector(6.0, 10.0), step=2.0)
        assert rounded.mass.amplitude == 6.0 and rounded.error is None
        rounded = round_mass(MassVector(0.4, 10.0), step=1.0)
        assert rounded.mass is None
        assert np.isclose(rounded.error.amplitude, 0.4)
        assert np.isclose(rounded.error.phase, -170.0)
        with pytest.raises(ValueError, match="step must be positive"):
            round_mass(MassVector(1.0, 0.0), step=0.0)

    def test_mass_error(self):
        """A 15° position error leaves 2 sin(7.5°) of the mass."""
        error = mass_error(MassVector(10.0, 0.0), MassVector(10.0, 15.0))
        assert np.isclose(error.amplitude, 20.0 * np.sin(np.deg2rad(7.5)))
        assert mass_error(MassVector(10.0, 0.0), MassVector(10.0, 0.0)) is None
        parts = split_mass(MassVector(10.0, 20.0), 0.0, 45.0)
        assert mass_error(MassVector(10.0, 20.0), parts) is None
        with pytest.raises(ValueError, match="units must match"):
            mass_error(MassVector(1.0, 0.0), MassVector(1.0, 0.0, "oz"))


class TestWeightsInAJob:
    def test_known_unbalance_fixed_positions_rounded(self):
        """Known unbalance -> split onto blades -> known remaining vibration.

        A heavy spot of 20 g at 100° on a 7-blade fan.  The ideal
        correction of 20 g at -80° is split onto the two nearest blades;
        the split weights cancel the vibration.  Rounded to 1 g, they
        leave exactly the vibration of the rounding errors.
        """
        alpha = 0.12 * np.exp(1j * np.deg2rad(-50.0))
        coefficients = InfluenceCoefficients([[alpha]])
        heavy = 20.0 * np.exp(1j * np.deg2rad(100.0))
        reading = VibrationVector.from_complex(alpha * heavy)

        ideal = coefficients.compute_compensation(reading)[0]
        assert np.isclose(ideal.amplitude, 20.0)
        assert np.isclose(ideal.phase, -80.0)

        parts = split_mass_on_positions(ideal, 7)
        assert [blade for blade, _ in parts] == [5, 6]   # 257.1° and 308.6°
        fitted = [part for _, part in parts]
        after = coefficients.predict_vibration(reading, [fitted])[0]
        assert np.isclose(after.amplitude, 0.0, atol=1e-12)

        rounded = [round_mass(part, step=1.0) for part in fitted]
        errors = combine_masses([r.error for r in rounded])
        after = coefficients.predict_vibration(
            reading, [[r.mass for r in rounded]])[0]
        assert np.isclose(after.complex_num, alpha * _c(errors))
        assert np.isclose(_c(mass_error(ideal, [r.mass for r in rounded])),
                          _c(errors))
