import numpy as np
import pytest

from pyPRB import DynamicBalancing, StaticBalancing, VibrationVector


class TestVibrationVector:
    """Tests for the VibrationVector class."""

    def test_creation(self):
        """Test creation of VibrationVector."""
        v = VibrationVector(10.0, 45.0)
        assert v.amplitude == 10.0
        assert v.phase == 45.0

    def test_creatin_from_complex(self):
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
        
class TestStaticBalancing:
    """Tests for the StaticBalancing class."""

    def test_creation(self):
        """Test creation of StaticBalancing."""
        V_0 = VibrationVector(10.0, 0.0)
        V_1 = VibrationVector(5.0, 90.0)
        balancer = StaticBalancing(V_0, V_1, mass=2.0)
        assert balancer.V_0 == V_0
        assert balancer.V_1 == V_1
        assert balancer.mass == 2.0
        
    def test_invalid_mass(self):
        """Test that zero or negative mass raises ValueError."""
        V_0 = VibrationVector(10.0, 0.0)
        V_1 = VibrationVector(5.0, 90.0)
        with pytest.raises(ValueError, match="Trial mass must be positive."):
            StaticBalancing(V_0, V_1, mass=0.0)
        with pytest.raises(ValueError, match="Trial mass must be positive."):
            StaticBalancing(V_0, V_1, mass=-1.0)
            
    def test_BK_example1(self):
        """Test StaticBalancing with example 1 from B&K application notes."""
        V_0 = VibrationVector(15, 55)
        V_1 = VibrationVector(18, 170)
        balancer = StaticBalancing(V_0, V_1, mass=5.0)
        compensation_mass, angle = balancer.compute_compensation()
        assert np.isclose(compensation_mass, 2.6, atol=0, rtol=0.05)
        assert np.isclose(angle, 37.0, atol=0, rtol=0.05)
        
    def test_BK_example2(self):
        """Test StaticBalancing with example 2 from B&K application notes."""
        V_0 = VibrationVector(3.4, 116)
        V_1 = VibrationVector(1.8, 42)
        balancer = StaticBalancing(V_0, V_1, mass=2.0)
        compensation_mass, angle = balancer.compute_compensation()
        assert np.isclose(compensation_mass, 2.03, atol=0, rtol=0.05)
        assert np.isclose(angle, -31.0, atol=0, rtol=0.05)
        
        
class TestDynamicBalancing:
    """Tests for the DynamicBalancing class."""

    def test_creation(self):
        """Test creation of DynamicBalancing."""
        V_1_0 = VibrationVector(10.0, 0.0)
        V_2_0 = VibrationVector(15.0, 90.0)
        V_1_1 = VibrationVector(5.0, 45.0)
        V_2_1 = VibrationVector(7.0, 135.0)
        V_1_2 = VibrationVector(6.0, 30.0)
        V_2_2 = VibrationVector(8.0, 120.0)
        balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                                    V_1_2, V_2_2, mass_1_1=2.0, mass_2_2=3.0)
        assert balancer.V_1_0 == V_1_0
        assert balancer.V_2_0 == V_2_0
        assert balancer.V_1_1 == V_1_1
        assert balancer.V_2_1 == V_2_1
        assert balancer.V_1_2 == V_1_2
        assert balancer.V_2_2 == V_2_2
        assert balancer.mass_1_1 == 2.0
        assert balancer.mass_2_2 == 3.0
        
    def test_invalid_masses(self):
        """Test that zero or negative masses raise ValueError."""
        V_1_0 = VibrationVector(10.0, 0.0)
        V_2_0 = VibrationVector(15.0, 90.0)
        V_1_1 = VibrationVector(5.0, 45.0)
        V_2_1 = VibrationVector(7.0, 135.0)
        V_1_2 = VibrationVector(6.0, 30.0)
        V_2_2 = VibrationVector(8.0, 120.0)
        with pytest.raises(ValueError, match="Trial mass must be positive."):
            DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                             V_1_2, V_2_2, mass_1_1=0.0, mass_2_2=3.0)
        with pytest.raises(ValueError, match="Trial mass must be positive."):
            DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                             V_1_2, V_2_2, mass_1_1=-1.0, mass_2_2=3.0)
        with pytest.raises(ValueError, match="Trial mass must be positive."):
            DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                             V_1_2, V_2_2, mass_1_1=2.0, mass_2_2=0.0)
        with pytest.raises(ValueError, match="Trial mass must be positive."):
            DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                             V_1_2, V_2_2, mass_1_1=2.0, mass_2_2=-1.0)

    def test_BK_example4(self):
        """Test DynamicBalancing with example 4 from B&K application notes."""
        V_1_0 = VibrationVector(7.2, 238)
        V_2_0 = VibrationVector(13.5, 296)
        V_1_1 = VibrationVector(4.9, 114)
        V_2_1 = VibrationVector(9.2, 347)
        V_1_2 = VibrationVector(4, 79)
        V_2_2 = VibrationVector(12, 292)
        balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                                    V_1_2, V_2_2, mass_1_1=2.5, mass_2_2=2.5)
        compensation_mass_1, angle_1, compensation_mass_2, angle_2 = balancer.compute_compensation()
        assert np.isclose(compensation_mass_1, 2.93, atol=0, rtol=0.05)
        assert np.isclose(angle_1, 50.4, atol=0, rtol=0.05)
        assert np.isclose(compensation_mass_2, 2.84, atol=0, rtol=0.05)
        assert np.isclose(angle_2, -81.9, atol=0, rtol=0.05)
        
    def test_BK_example5(self):
        """Test DynamicBalancing with example 5 from B&K application notes."""
        V_1_0 = VibrationVector(170, 112)
        V_2_0 = VibrationVector(53, 78)
        V_1_1 = VibrationVector(235, 94)
        V_2_1 = VibrationVector(58, 68)
        V_1_2 = VibrationVector(185, 115)
        V_2_2 = VibrationVector(77, 104)
        balancer = DynamicBalancing(V_1_0, V_2_0, V_1_1, V_2_1,
                                    V_1_2, V_2_2, mass_1_1=1.15, mass_2_2=1.15)
        compensation_mass_1, angle_1, compensation_mass_2, angle_2 = balancer.compute_compensation()
        assert np.isclose(compensation_mass_1, 1.98, atol=0, rtol=0.05)
        assert np.isclose(angle_1, 236.2-360, atol=0, rtol=0.05)
        assert np.isclose(compensation_mass_2, 1.07, atol=0, rtol=0.05)
        assert np.isclose(angle_2, 121.8, atol=0, rtol=0.05)