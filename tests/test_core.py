import numpy as np
import pytest

from pyPRB import DynamicBalancing, MassVector, StaticBalancing, VibrationVector


class TestVibrationVector:
    """Tests for the VibrationVector class."""

    def test_creation(self):
        """Test creation of VibrationVector."""
        v = VibrationVector(10.0, 45.0)
        assert v.amplitude == 10.0
        assert v.phase == 45.0

    def test_creation_from_complex(self):
        """Test creation from complex number."""
        v = VibrationVector.from_complex(1 + 1j)
        assert np.isclose(v.amplitude, np.sqrt(2))
        assert np.isclose(v.phase, 45.0)

    def test_negative_amplitude(self):
        """Test that negative amplitude raises ValueError."""
        with pytest.raises(ValueError, match="Amplitude must be non-negative."):
            VibrationVector(-5.0, 30.0)

    def test_phase_wrapping(self):
        """Test that phase wraps correctly to [0, 360)."""
        v1 = VibrationVector(10.0, 370.0)
        assert v1.phase == 10.0  # 370 - 360 == 10

        v2 = VibrationVector(10.0, -30.0)
        assert v2.phase == 330.0  # -30 + 360 == 330

        v3 = VibrationVector(10.0, 720.0)
        assert v3.phase == 0.0  # 720 % 360 == 0

    def test_addition(self):
        """Test addition of two VibrationVectors."""
        v1 = VibrationVector(10.0, 0.0)
        v2 = VibrationVector(10.0, 90.0)
        v3 = v1 + v2
        assert np.isclose(v3.amplitude, np.sqrt(10**2 + 10**2))
        assert np.isclose(v3.phase, 45.0)

    def test_subtraction(self):
        """Test subtraction of two VibrationVectors."""
        v1 = VibrationVector(10.0, 90.0)
        v2 = VibrationVector(10.0, 0.0)
        v3 = v1 - v2
        assert np.isclose(v3.amplitude, np.sqrt(10**2 + 10**2))
        assert np.isclose(v3.phase, 135.0)

    def test_multiplication(self):
        """Test multiplication of VibrationVector."""
        v = VibrationVector(10.0, 45.0)
        v2 = v * 2.0
        assert v2.amplitude == 20.0
        assert v2.phase == 45.0

        v = VibrationVector(10.0, 45.0)
        v2 = 2.0 * v
        assert v2.amplitude == 20.0
        assert v2.phase == 45.0

        v = VibrationVector(10.0, 45.0)
        v2 = -2.0 * v
        assert v2.amplitude == 20.0
        assert v2.phase == 225.0

        v = VibrationVector(10.0, 45.0)
        v1 = VibrationVector(2.0, 90.0)
        v2 = v * v1
        assert np.isclose(v2.amplitude, 20.0)
        assert np.isclose(v2.phase, 135.0)

    def test_division(self):
        """Test division of VibrationVector."""
        v = VibrationVector(10.0, 90.0)
        v2 = v / 2.0
        assert v2.amplitude == 5.0
        assert v2.phase == 90.0

        v = VibrationVector(10.0, 90.0)
        v1 = VibrationVector(2.0, 45.0)
        v2 = v / v1
        assert np.isclose(v2.amplitude, 5.0)
        assert np.isclose(v2.phase, 45.0)

    def test_negation(self):
        """Test negation of VibrationVector."""
        v = VibrationVector(10.0, 45.0)
        v2 = -v
        assert v2.amplitude == 10.0
        assert v2.phase == 225.0


class TestMassVector:
    """Tests for the MassVector class."""

    def test_creation(self):
        """Test creation of MassVector with default units."""
        m = MassVector(2.5, 45.0)
        assert m.amplitude == 2.5
        assert m.phase == 45.0
        assert m.units == "g"

    def test_creation_with_units(self):
        """Test creation of MassVector with custom units."""
        m = MassVector(3.0, 90.0, units="oz")
        assert m.units == "oz"

    def test_invalid_amplitude(self):
        """Test that zero or negative amplitude raises ValueError."""
        with pytest.raises(ValueError, match="Amplitude must be positive."):
            MassVector(0.0, 45.0)
        with pytest.raises(ValueError, match="Amplitude must be positive."):
            MassVector(-1.0, 45.0)

    def test_invalid_units(self):
        """Test that non-string units raises TypeError."""
        with pytest.raises(TypeError, match="Units must be a string."):
            MassVector(2.0, 45.0, units=123)

    @pytest.mark.parametrize("input_phase,expected", [
        (350.0, -10.0),    # 350 -> -10
        (-30.0, -30.0),    # -30 stays -30
        (180.0, 180.0),    # 180 stays 180
        (540.0, 180.0),    # 540 % 360 = 180
    ])
    def test_phase_normalization(self, input_phase, expected):
        """Test that phase is normalized to [-180, 180]."""
        m = MassVector(1.0, input_phase)
        assert m.phase == expected

    def test_repr(self):
        """Test string representation."""
        m = MassVector(2.5, 45.0)
        assert repr(m) == "MassVector(amplitude=2.5, phase=45°, units='g')"


class TestStaticBalancing:
    """Tests for the StaticBalancing class."""

    def test_creation(self):
        """Test creation of StaticBalancing."""
        V_0 = VibrationVector(10.0, 0.0)
        V_1 = VibrationVector(5.0, 90.0)
        trial_mass = MassVector(2.0, 0.0)
        balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)
        assert balancer.V_0 == V_0
        assert balancer.V_1 == V_1
        assert balancer.trial_mass.amplitude == 2.0

    def test_BK_example1(self):
        """Test StaticBalancing with example 1 from B&K application notes."""
        V_0 = VibrationVector(15, 55)
        V_1 = VibrationVector(18, 170)
        trial_mass = MassVector(5.0, 0.0)
        balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)
        comp_mass = balancer.compute_compensation()
        assert np.isclose(comp_mass.amplitude, 2.6, atol=0, rtol=0.05)
        assert np.isclose(comp_mass.phase, 37.0, atol=0, rtol=0.05)

    def test_BK_example2(self):
        """Test StaticBalancing with example 2 from B&K application notes."""
        V_0 = VibrationVector(3.4, 116)
        V_1 = VibrationVector(1.8, 42)
        trial_mass = MassVector(2.0, 0.0)
        balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)
        comp_mass = balancer.compute_compensation()
        assert np.isclose(comp_mass.amplitude, 2.03, atol=0, rtol=0.05)
        assert np.isclose(comp_mass.phase, -31.0, atol=0, rtol=0.05)

    def test_absolute_angle(self):
        """Test that compensation angle is absolute (offset by trial mass position)."""
        V_0 = VibrationVector(3.4, 116)
        V_1 = VibrationVector(1.8, 42)
        # With trial mass at 0°, relative angle is ~-31°
        comp_mass_at_zero = StaticBalancing(
            V_0, V_1, trial_mass=MassVector(2.0, 0.0)
        ).compute_compensation(repr=False)
        # With trial mass at 90°, angle should shift by +90°
        comp_mass_at_90 = StaticBalancing(
            V_0, V_1, trial_mass=MassVector(2.0, 90.0)
        ).compute_compensation(repr=False)
        assert np.isclose(comp_mass_at_zero.amplitude, comp_mass_at_90.amplitude)
        assert np.isclose(comp_mass_at_90.phase, comp_mass_at_zero.phase + 90.0,
                          atol=0, rtol=0.05)


class TestDynamicBalancing:
    """Tests for the DynamicBalancing class."""

    def test_units_mismatch(self):
        """Test that mismatched trial mass units raise ValueError."""
        V_1_0 = VibrationVector(10.0, 0.0)
        V_2_0 = VibrationVector(15.0, 90.0)
        V_1_1 = VibrationVector(5.0, 45.0)
        V_2_1 = VibrationVector(7.0, 135.0)
        V_1_2 = VibrationVector(6.0, 30.0)
        V_2_2 = VibrationVector(8.0, 120.0)
        with pytest.raises(ValueError, match="Trial mass units must match"):
            DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                             V_1_2, V_2_2,
                             trial_mass_1=MassVector(2.0, 0.0, units="g"),
                             trial_mass_2=MassVector(3.0, 0.0, units="oz"))

    def test_creation(self):
        """Test creation of DynamicBalancing."""
        V_1_0 = VibrationVector(10.0, 0.0)
        V_2_0 = VibrationVector(15.0, 90.0)
        V_1_1 = VibrationVector(5.0, 45.0)
        V_2_1 = VibrationVector(7.0, 135.0)
        V_1_2 = VibrationVector(6.0, 30.0)
        V_2_2 = VibrationVector(8.0, 120.0)
        trial_mass_1 = MassVector(2.0, 0.0)
        trial_mass_2 = MassVector(3.0, 0.0)
        balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                                    V_1_2, V_2_2,
                                    trial_mass_1=trial_mass_1,
                                    trial_mass_2=trial_mass_2)
        assert balancer.V_1_0 == V_1_0
        assert balancer.V_2_0 == V_2_0
        assert balancer.V_1_1 == V_1_1
        assert balancer.V_2_1 == V_2_1
        assert balancer.V_1_2 == V_1_2
        assert balancer.V_2_2 == V_2_2
        assert balancer.trial_mass_1.amplitude == 2.0
        assert balancer.trial_mass_2.amplitude == 3.0

    def test_BK_example4(self):
        """Test DynamicBalancing with example 4 from B&K application notes."""
        V_1_0 = VibrationVector(7.2, 238)
        V_2_0 = VibrationVector(13.5, 296)
        V_1_1 = VibrationVector(4.9, 114)
        V_2_1 = VibrationVector(9.2, 347)
        V_1_2 = VibrationVector(4, 79)
        V_2_2 = VibrationVector(12, 292)
        trial_mass_1 = MassVector(2.5, 0.0)
        trial_mass_2 = MassVector(2.5, 0.0)
        balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                                    V_1_2, V_2_2,
                                    trial_mass_1=trial_mass_1,
                                    trial_mass_2=trial_mass_2)
        comp_mass_1, comp_mass_2 = balancer.compute_compensation()
        assert np.isclose(comp_mass_1.amplitude, 2.93, atol=0, rtol=0.05)
        assert np.isclose(comp_mass_1.phase, 50.4, atol=0, rtol=0.05)
        assert np.isclose(comp_mass_2.amplitude, 2.84, atol=0, rtol=0.05)
        assert np.isclose(comp_mass_2.phase, -81.9, atol=0, rtol=0.05)

    def test_BK_example5(self):
        """Test DynamicBalancing with example 5 from B&K application notes."""
        V_1_0 = VibrationVector(170, 112)
        V_2_0 = VibrationVector(53, 78)
        V_1_1 = VibrationVector(235, 94)
        V_2_1 = VibrationVector(58, 68)
        V_1_2 = VibrationVector(185, 115)
        V_2_2 = VibrationVector(77, 104)
        trial_mass_1 = MassVector(1.15, 0.0)
        trial_mass_2 = MassVector(1.15, 0.0)
        balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                                    V_1_2, V_2_2,
                                    trial_mass_1=trial_mass_1,
                                    trial_mass_2=trial_mass_2)
        comp_mass_1, comp_mass_2 = balancer.compute_compensation()
        assert np.isclose(comp_mass_1.amplitude, 1.98, atol=0, rtol=0.05)
        assert np.isclose(comp_mass_1.phase, 236.2-360, atol=0, rtol=0.05)
        assert np.isclose(comp_mass_2.amplitude, 1.07, atol=0, rtol=0.05)
        assert np.isclose(comp_mass_2.phase, 121.8, atol=0, rtol=0.05)

    def test_absolute_angles(self):
        """Test that compensation angles are absolute (offset by trial mass positions)."""
        V_1_0 = VibrationVector(7.2, 238)
        V_2_0 = VibrationVector(13.5, 296)
        V_1_1 = VibrationVector(4.9, 114)
        V_2_1 = VibrationVector(9.2, 347)
        V_1_2 = VibrationVector(4, 79)
        V_2_2 = VibrationVector(12, 292)
        # Baseline at 0°
        comp_mass_1_base, comp_mass_2_base = DynamicBalancing(
            V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2,
            trial_mass_1=MassVector(2.5, 0.0),
            trial_mass_2=MassVector(2.5, 0.0)
        ).compute_compensation(repr=False)
        # With trial masses at 30° and 60°
        comp_mass_1_off, comp_mass_2_off = DynamicBalancing(
            V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2,
            trial_mass_1=MassVector(2.5, 30.0),
            trial_mass_2=MassVector(2.5, 60.0)
        ).compute_compensation(repr=False)
        # Amplitudes should be identical
        assert np.isclose(comp_mass_1_base.amplitude, comp_mass_1_off.amplitude)
        assert np.isclose(comp_mass_2_base.amplitude, comp_mass_2_off.amplitude)
        # Phases should be offset by 30° and 60° respectively
        assert np.isclose(comp_mass_1_off.phase, comp_mass_1_base.phase + 30.0,
                          atol=0, rtol=0.05)
        assert np.isclose(comp_mass_2_off.phase, comp_mass_2_base.phase + 60.0,
                          atol=0, rtol=0.05)
