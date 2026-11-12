import numpy as np
import pytest

import warnings

from pyPRB import (DynamicBalancing, IllConditionedWarning,
                   InfluenceCoefficients, MassVector, MultiPlaneBalancing,
                   StaticBalancing, VibrationVector, convert_angle,
                   convert_convention)


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


def _simulate_runs(alpha, unbalance, trial_masses):
    """Simulate initial and trial runs for a linear rotor model.

    ``alpha`` is the (M x N) complex influence matrix, ``unbalance`` the
    complex initial unbalance in each plane and ``trial_masses`` a list of
    MassVector.  Returns ``V_0`` and ``V_trial`` as VibrationVector lists.
    """
    V_0_c = alpha @ unbalance
    V_0 = [VibrationVector.from_complex(z) for z in V_0_c]
    V_trial = []
    for j, tm in enumerate(trial_masses):
        T_j = tm.amplitude * np.exp(1j * np.deg2rad(tm.phase))
        V_j_c = V_0_c + alpha[:, j] * T_j
        V_trial.append([VibrationVector.from_complex(z) for z in V_j_c])
    return V_0, V_trial


def _as_complex(mass):
    return mass.amplitude * np.exp(1j * np.deg2rad(mass.phase))


class TestMultiPlaneBalancing:
    """Tests for the MultiPlaneBalancing class."""

    def test_creation(self):
        """Test creation and stored dimensions."""
        V_0 = [VibrationVector(1, 0), VibrationVector(2, 0),
               VibrationVector(3, 0)]
        V_trial = [[VibrationVector(1, 10), VibrationVector(2, 20),
                    VibrationVector(3, 30)],
                   [VibrationVector(1, 40), VibrationVector(2, 50),
                    VibrationVector(3, 60)]]
        trial_masses = [MassVector(1.0, 0.0), MassVector(2.0, 0.0)]
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        assert balancer.n_sensors == 3
        assert balancer.n_planes == 2
        assert balancer.influence_coefficients.shape == (3, 2)
        assert "Plane 2" in repr(balancer)
        assert "Point 3" in repr(balancer)

    def test_matches_static_balancing(self):
        """One plane, one point: same result as StaticBalancing."""
        V_0 = VibrationVector(3.4, 116)
        V_1 = VibrationVector(1.8, 42)
        trial_mass = MassVector(2.0, 25.0)
        expected = StaticBalancing(V_0, V_1, trial_mass).compute_compensation(
            repr=False)
        (result,) = MultiPlaneBalancing([V_0], [[V_1]], [trial_mass]
                                        ).compute_compensation(repr=False)
        assert np.isclose(result.amplitude, expected.amplitude)
        assert np.isclose(result.phase, expected.phase)
        assert result.units == expected.units

    @pytest.mark.parametrize("vectors, masses", [
        # B&K example 4
        ([(7.2, 238), (13.5, 296), (4.9, 114), (9.2, 347), (4, 79), (12, 292)],
         [(2.5, 0.0), (2.5, 0.0)]),
        # B&K example 5, trial masses at different positions
        ([(170, 112), (53, 78), (235, 94), (58, 68), (185, 115), (77, 104)],
         [(1.15, 30.0), (1.15, 60.0)]),
    ])
    def test_matches_dynamic_balancing(self, vectors, masses):
        """Two planes, two points: same result as DynamicBalancing."""
        V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2 = [
            VibrationVector(a, p) for a, p in vectors]
        tm1, tm2 = [MassVector(a, p) for a, p in masses]
        expected = DynamicBalancing(
            V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2,
            trial_mass_1=tm1, trial_mass_2=tm2,
        ).compute_compensation(repr=False)
        result = MultiPlaneBalancing(
            [V_1_0, V_2_0],
            [[V_1_1, V_2_1], [V_1_2, V_2_2]],
            [tm1, tm2],
        ).compute_compensation(repr=False)
        for r, e in zip(result, expected):
            assert np.isclose(r.amplitude, e.amplitude)
            assert np.isclose(r.phase, e.phase)

    @pytest.mark.parametrize("vectors, mass, expected", [
        # B&K example 4: 2.93 g @ 50.4°, 2.84 g @ -81.9°
        ([(7.2, 238), (13.5, 296), (4.9, 114), (9.2, 347), (4, 79), (12, 292)],
         2.5, [(2.93, 50.4), (2.84, -81.9)]),
        # B&K example 5: 1.98 g @ 236.2°, 1.07 g @ 121.8°
        ([(170, 112), (53, 78), (235, 94), (58, 68), (185, 115), (77, 104)],
         1.15, [(1.98, 236.2 - 360), (1.07, 121.8)]),
    ])
    def test_BK_two_plane_examples(self, vectors, mass, expected):
        """Two-plane B&K examples give the published compensation masses."""
        V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2 = [
            VibrationVector(a, p) for a, p in vectors]
        comp = MultiPlaneBalancing(
            [V_1_0, V_2_0],
            [[V_1_1, V_2_1], [V_1_2, V_2_2]],
            [MassVector(mass, 0.0), MassVector(mass, 0.0)],
        ).compute_compensation(repr=False)
        for m, (amp, phase) in zip(comp, expected):
            assert np.isclose(m.amplitude, amp, atol=0, rtol=0.05)
            assert np.isclose(m.phase, phase, atol=0, rtol=0.05)

    @pytest.mark.parametrize("n_planes", [3, 4, 6])
    def test_recovers_unbalance_square(self, n_planes):
        """N planes, N points: compensation cancels a known unbalance."""
        rng = np.random.default_rng(n_planes)
        alpha = (rng.standard_normal((n_planes, n_planes))
                 + 1j * rng.standard_normal((n_planes, n_planes)))
        unbalance = (rng.uniform(0.5, 3, n_planes)
                     * np.exp(1j * rng.uniform(0, 2 * np.pi, n_planes)))
        trial_masses = [MassVector(a, p) for a, p in zip(
            rng.uniform(0.5, 2, n_planes), rng.uniform(-180, 180, n_planes))]
        V_0, V_trial = _simulate_runs(alpha, unbalance, trial_masses)

        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        np.testing.assert_allclose(balancer.influence_coefficients, alpha)

        comp = balancer.compute_compensation(repr=False)
        assert len(comp) == n_planes
        np.testing.assert_allclose([_as_complex(m) for m in comp], -unbalance)
        residual = balancer.residual_vibration()
        assert len(residual) == n_planes
        assert max(v.amplitude for v in residual) < 1e-9

    def test_overdetermined_least_squares(self):
        """More points than planes: least-squares solution."""
        rng = np.random.default_rng(42)
        n_sensors, n_planes = 5, 3
        alpha = (rng.standard_normal((n_sensors, n_planes))
                 + 1j * rng.standard_normal((n_sensors, n_planes)))
        unbalance = np.array([1.0 + 0.5j, -0.8 + 1.2j, 0.3 - 0.9j])
        trial_masses = [MassVector(1.0, 0.0), MassVector(1.5, 90.0),
                        MassVector(0.8, -45.0)]

        # Consistent data: the exact unbalance is recovered.
        V_0, V_trial = _simulate_runs(alpha, unbalance, trial_masses)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        comp = balancer.compute_compensation(repr=False)
        np.testing.assert_allclose([_as_complex(m) for m in comp], -unbalance)

        # Add vibration not reachable by the correction planes: the result
        # is the least-squares solution and the residual is orthogonal to
        # the columns of the influence matrix.
        extra = 0.2 * (rng.standard_normal(n_sensors)
                       + 1j * rng.standard_normal(n_sensors))
        V_0 = [VibrationVector.from_complex(v.complex_num + e)
               for v, e in zip(V_0, extra)]
        V_trial = [[VibrationVector.from_complex(v.complex_num + e)
                    for v, e in zip(run, extra)] for run in V_trial]
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        W = np.array([_as_complex(m)
                      for m in balancer.compute_compensation(repr=False)])
        V_0_c = np.array([v.complex_num for v in V_0])
        W_ref = np.linalg.lstsq(alpha, -V_0_c, rcond=None)[0]
        np.testing.assert_allclose(W, W_ref)
        residual = np.array([v.complex_num
                             for v in balancer.residual_vibration()])
        np.testing.assert_allclose(residual, V_0_c + alpha @ W_ref)
        np.testing.assert_allclose(alpha.conj().T @ residual, 0, atol=1e-12)
        assert np.linalg.norm(residual) < np.linalg.norm(V_0_c)

    def test_absolute_angles(self):
        """Rotating a trial mass position rotates its compensation equally."""
        rng = np.random.default_rng(1)
        alpha = (rng.standard_normal((3, 3))
                 + 1j * rng.standard_normal((3, 3)))
        unbalance = np.array([1.0, 1j, -0.5 + 0.5j])
        base_masses = [MassVector(1.0, 0.0)] * 3
        V_0, V_trial = _simulate_runs(alpha, unbalance, base_masses)
        base = MultiPlaneBalancing(V_0, V_trial, base_masses
                                   ).compute_compensation(repr=False)
        # Same measured vectors, but the trial masses are declared at
        # 30°, 60° and 90°: compensations rotate by the same angles.
        shifted_masses = [MassVector(1.0, 30.0), MassVector(1.0, 60.0),
                          MassVector(1.0, 90.0)]
        shifted = MultiPlaneBalancing(V_0, V_trial, shifted_masses
                                      ).compute_compensation(repr=False)
        for b, s, offset in zip(base, shifted, [30.0, 60.0, 90.0]):
            assert np.isclose(s.amplitude, b.amplitude)
            assert np.isclose(_as_complex(s),
                              _as_complex(b) * np.exp(1j * np.deg2rad(offset)))

    def test_units_propagate(self):
        """Compensation masses carry the trial mass units."""
        alpha = np.array([[1.0, 0.2j], [0.1, 1.0 + 0.3j], [0.5j, 0.4]])
        trial_masses = [MassVector(1.0, 0.0, units="oz"),
                        MassVector(2.0, 0.0, units="oz")]
        V_0, V_trial = _simulate_runs(alpha, np.array([1.0, 1j]),
                                      trial_masses)
        comp = MultiPlaneBalancing(V_0, V_trial, trial_masses
                                   ).compute_compensation(repr=False)
        assert all(m.units == "oz" for m in comp)

    def test_print_table(self, capsys):
        """compute_compensation prints one row per plane when repr=True."""
        alpha = np.eye(3, dtype=complex)
        trial_masses = [MassVector(1.0, 0.0)] * 3
        V_0, V_trial = _simulate_runs(alpha, np.array([1.0, 2.0, 3.0]),
                                      trial_masses)
        MultiPlaneBalancing(V_0, V_trial, trial_masses).compute_compensation()
        out = capsys.readouterr().out
        assert "Compensation Mass" in out
        assert "3.00 g" in out

    def test_units_mismatch(self):
        """Mismatched trial mass units raise ValueError."""
        V = VibrationVector(1.0, 0.0)
        with pytest.raises(ValueError, match="Trial mass units must match"):
            MultiPlaneBalancing([V, V], [[V, V], [V, V]],
                                [MassVector(1.0, 0.0, units="g"),
                                 MassVector(1.0, 0.0, units="oz")])

    def test_too_few_measurement_points(self):
        """Fewer measurement points than planes raise ValueError."""
        V = VibrationVector(1.0, 0.0)
        with pytest.raises(ValueError, match="at least the number"):
            MultiPlaneBalancing([V], [[V], [V]],
                                [MassVector(1.0, 0.0), MassVector(1.0, 0.0)])

    def test_trial_run_count_mismatch(self):
        """Number of trial runs must equal number of trial masses."""
        V = VibrationVector(1.0, 0.0)
        with pytest.raises(ValueError, match="Number of trial runs"):
            MultiPlaneBalancing([V, V], [[V, V]],
                                [MassVector(1.0, 0.0), MassVector(1.0, 0.0)])

    def test_trial_run_length_mismatch(self):
        """Each trial run needs one vector per measurement point."""
        V = VibrationVector(1.0, 0.0)
        with pytest.raises(ValueError, match="Trial run 2 has 1"):
            MultiPlaneBalancing([V, V], [[V, V], [V]],
                                [MassVector(1.0, 0.0), MassVector(1.0, 0.0)])

    def test_empty(self):
        """At least one plane and one measurement point are required."""
        with pytest.raises(ValueError, match="At least one"):
            MultiPlaneBalancing([], [], [])

    def test_rank_deficient(self):
        """Trial runs with identical effect cannot separate the planes."""
        V_0 = [VibrationVector(1.0, 0.0), VibrationVector(2.0, 0.0)]
        run = [VibrationVector(2.0, 0.0), VibrationVector(4.0, 0.0)]
        balancer = MultiPlaneBalancing(V_0, [run, run],
                                       [MassVector(1.0, 0.0),
                                        MassVector(1.0, 0.0)])
        with pytest.raises(ValueError, match="rank deficient"):
            balancer.compute_compensation(repr=False)


class TestMultiPlaneConditioning:
    """Warning for ill-conditioned influence coefficient matrices."""

    @staticmethod
    def _nearly_identical_planes(eps):
        """Two planes whose effects on both points differ only by *eps*."""
        alpha = np.array([[1.0, 1.0 + eps], [0.5j, 0.5j * (1.0 - eps)]])
        trial_masses = [MassVector(1.0, 0.0), MassVector(1.0, 0.0)]
        V_0, V_trial = _simulate_runs(alpha, np.array([1.0, 0.5j]),
                                      trial_masses)
        return V_0, V_trial, trial_masses

    def test_condition_number(self):
        V_0, V_trial, trial_masses = self._nearly_identical_planes(1e-3)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        assert np.isclose(balancer.condition_number,
                          np.linalg.cond(balancer.influence_coefficients))
        assert balancer.condition_number > 1e3

    def test_warns_and_points_to_caller(self):
        """The warning is issued and attributed to the user's call."""
        V_0, V_trial, trial_masses = self._nearly_identical_planes(1e-3)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        with pytest.warns(IllConditionedWarning, match="ill-conditioned") as rec:
            balancer.compute_compensation(repr=False)
        assert rec[0].filename == __file__
        with pytest.warns(IllConditionedWarning) as rec:
            balancer.residual_vibration()
        assert rec[0].filename == __file__

    def test_no_warning_for_BK_example(self):
        V = [VibrationVector(a, p) for a, p in
             [(7.2, 238), (13.5, 296), (4.9, 114), (9.2, 347), (4, 79), (12, 292)]]
        balancer = MultiPlaneBalancing([V[0], V[1]], [[V[2], V[3]], [V[4], V[5]]],
                                       [MassVector(2.5, 0.0)] * 2)
        assert balancer.condition_number < 10
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            balancer.compute_compensation(repr=False)

    def test_limit_can_be_changed_or_disabled(self):
        V_0, V_trial, trial_masses = self._nearly_identical_planes(1e-3)
        cond = MultiPlaneBalancing(V_0, V_trial, trial_masses).condition_number
        for limit in (None, 2 * cond):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                    max_condition_number=limit
                                    ).compute_compensation(repr=False)
        with pytest.warns(IllConditionedWarning):
            MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                max_condition_number=0.5 * cond
                                ).compute_compensation(repr=False)

    def test_invalid_limit(self):
        V = VibrationVector(1.0, 0.0)
        with pytest.raises(ValueError, match="max_condition_number"):
            MultiPlaneBalancing([V], [[VibrationVector(2.0, 0.0)]],
                                [MassVector(1.0, 0.0)], max_condition_number=0)


class TestTrialMassLeftOn:
    """Tests for trial masses that stay on the rotor for the next runs."""

    def _case(self, n_planes=3, n_sensors=4, seed=5):
        rng = np.random.default_rng(seed)
        alpha = (rng.normal(size=(n_sensors, n_planes))
                 + 1j * rng.normal(size=(n_sensors, n_planes)))
        unbalance = rng.normal(size=n_planes) + 1j * rng.normal(size=n_planes)
        trial_masses = [MassVector(1.0 + 0.3 * j, 40.0 * j - 30.0)
                        for j in range(n_planes)]
        T = np.array([_as_complex(tm) for tm in trial_masses])
        V_0_c = alpha @ unbalance
        V_0 = [VibrationVector.from_complex(z) for z in V_0_c]
        V_trial = []
        on = np.zeros(n_planes, dtype=complex)
        for j in range(n_planes):
            on[j] = T[j]                   # previous trial masses stay
            V_trial.append([VibrationVector.from_complex(z)
                            for z in V_0_c + alpha @ on])
        return alpha, unbalance, trial_masses, T, V_0, V_trial

    def test_default_is_removed(self):
        """The default keeps the behaviour of removed trial masses."""
        balancer = MultiPlaneBalancing(
            [VibrationVector(1, 0)], [[VibrationVector(2, 30)]],
            [MassVector(1.0, 0.0)])
        assert balancer.trial_left_on is False

    def test_recovers_coefficients_and_unbalance(self):
        """Known unbalance and coefficients are recovered (square case)."""
        alpha, unbalance, trial_masses, T, V_0, V_trial = self._case(3, 3)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                       trial_left_on=True)
        assert np.allclose(balancer.influence_coefficients, alpha)
        comp = balancer.compute_compensation(repr=False)
        assert np.allclose([_as_complex(m) for m in comp], -unbalance)

    def test_wrong_flag_gives_wrong_coefficients(self):
        """Without the flag the same runs give other coefficients."""
        alpha, _, trial_masses, _, V_0, V_trial = self._case(3, 3)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)
        assert np.allclose(balancer.influence_coefficients[:, 0], alpha[:, 0])
        assert not np.allclose(balancer.influence_coefficients, alpha)

    def test_keep_trial_masses(self):
        """Masses to add with the trial masses kept: W - T, zero residual."""
        alpha, unbalance, trial_masses, T, V_0, V_trial = self._case(3, 3)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                       trial_left_on=True)
        add = balancer.compute_compensation(repr=False,
                                            keep_trial_masses=True)
        add_c = np.array([_as_complex(m) for m in add])
        assert np.allclose(add_c, -unbalance - T)
        # total on the rotor: unbalance + trial masses + added = 0
        assert np.allclose(alpha @ (unbalance + T + add_c), 0)

    def test_keep_equals_trim_from_last_run(self):
        """Keeping the trial masses equals a trim from the last run (M > N)."""
        _, _, trial_masses, _, V_0, V_trial = self._case(2, 4)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                       trial_left_on=True)
        add = balancer.compute_compensation(repr=False,
                                            keep_trial_masses=True)
        trim = balancer.get_influence_coefficients().compute_compensation(
            V_trial[-1])
        assert np.allclose([_as_complex(m) for m in add],
                           [_as_complex(m) for m in trim])

    def test_keep_requires_left_on(self):
        """keep_trial_masses needs runs measured with the masses left on."""
        balancer = MultiPlaneBalancing(
            [VibrationVector(1, 0)], [[VibrationVector(2, 30)]],
            [MassVector(1.0, 0.0)])
        with pytest.raises(ValueError, match="requires trial_left_on=True"):
            balancer.compute_compensation(repr=False, keep_trial_masses=True)

    def test_one_plane_left_on_formula(self):
        """One plane with the trial left on: W_add = -V_1 / alpha.

        Data of the fan example in Kelm, Kelm and Pavelek, "Rotor Balancing
        Tutorial", Turbomachinery and Pump Symposia, 2016 (readings
        5.6 @ 135° and 3.3 @ 238°, trial 74 oz·in @ 315°).  The published
        correction with the trial removed is 58.24 oz·in @ 341.9°; with
        the trial left on the mass to add is that correction minus the
        trial mass.
        """
        V_0, V_1 = VibrationVector(5.6, 135), VibrationVector(3.3, 238)
        trial = MassVector(74.0, 315.0, units="oz·in")
        balancer = MultiPlaneBalancing([V_0], [[V_1]], [trial],
                                       trial_left_on=True)
        removed = balancer.compute_compensation(repr=False)[0]
        assert np.isclose(removed.amplitude, 58.24, atol=0, rtol=2e-3)
        assert np.isclose(removed.phase % 360, 341.9, atol=0.1, rtol=0)
        add = balancer.compute_compensation(repr=False,
                                            keep_trial_masses=True)[0]
        alpha = (V_1.complex_num - V_0.complex_num) / _as_complex(trial)
        assert np.isclose(_as_complex(add), -V_1.complex_num / alpha)
        assert np.isclose(_as_complex(add),
                          _as_complex(removed) - _as_complex(trial))

    def test_print_table(self, capsys):
        """The table says that the trial masses were left on."""
        _, _, trial_masses, _, V_0, V_trial = self._case(2, 2)
        balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses,
                                       trial_left_on=True)
        assert "left on" in repr(balancer)
        balancer.compute_compensation(keep_trial_masses=True)
        assert "Compensation Mass" in capsys.readouterr().out


class TestInfluenceCoefficients:
    """Tests for the InfluenceCoefficients class."""

    BK4 = dict(
        V_0=[VibrationVector(7.2, 238), VibrationVector(13.5, 296)],
        V_trial=[[VibrationVector(4.9, 114), VibrationVector(9.2, 347)],
                 [VibrationVector(4.0, 79), VibrationVector(12.0, 292)]],
        trial_masses=[MassVector(2.5, 0.0), MassVector(2.5, 0.0)],
    )

    def test_BK_two_plane_example_from_stored_coefficients(self):
        """Two-plane example of the B&K application note without trial runs.

        J. Vaughan, "Static and Dynamic Balancing using portable measuring
        equipment", Brüel & Kjær application note: published result
        2.93 g @ 50.4° and 2.84 g @ -81.9° (tolerance 5 % as in the
        existing tests; exactly equal to DynamicBalancing).  The
        coefficients go through a plain dictionary and JSON first.
        """
        import json
        coefficients = InfluenceCoefficients.from_trial_runs(**self.BK4)
        stored = json.loads(json.dumps(coefficients.to_dict()))
        loaded = InfluenceCoefficients.from_dict(stored)
        comp = loaded.compute_compensation(self.BK4["V_0"])
        assert np.isclose(comp[0].amplitude, 2.93, atol=0, rtol=0.05)
        assert np.isclose(comp[0].phase, 50.4, atol=0, rtol=0.05)
        assert np.isclose(comp[1].amplitude, 2.84, atol=0, rtol=0.05)
        assert np.isclose(comp[1].phase, -81.9, atol=0, rtol=0.05)

        V = self.BK4["V_0"] + self.BK4["V_trial"][0] + self.BK4["V_trial"][1]
        reference = DynamicBalancing(
            V[0], V[1], V[2], V[3], V[4], V[5],
            *self.BK4["trial_masses"]).compute_compensation(repr=False)
        for c, r in zip(comp, reference):
            assert np.isclose(c.amplitude, r.amplitude, rtol=1e-12)
            assert np.isclose(c.phase, r.phase, rtol=1e-12)

    def test_BK_measured_two_plane_example_from_arrays(self):
        """B&K application note, measured two-plane example (1.15 g trials).

        Published: 1.98 g @ 236.2° and 1.07 g @ 121.8°; reproduced to
        1 % in mass and 0.5° in angle.
        """
        V_0 = [VibrationVector(170, 112), VibrationVector(53, 78)]
        V_trial = [[VibrationVector(235, 94), VibrationVector(58, 68)],
                   [VibrationVector(185, 115), VibrationVector(77, 104)]]
        trial_masses = [MassVector(1.15, 0.0), MassVector(1.15, 0.0)]
        amplitude, phase = InfluenceCoefficients.from_trial_runs(
            V_0, V_trial, trial_masses).to_amplitude_phase()
        comp = InfluenceCoefficients.from_amplitude_phase(
            amplitude, phase).compute_compensation(V_0)
        assert np.isclose(comp[0].amplitude, 1.98, atol=0, rtol=0.01)
        assert np.isclose(comp[0].phase % 360, 236.2, atol=0.5, rtol=0)
        assert np.isclose(comp[1].amplitude, 1.07, atol=0, rtol=0.01)
        assert np.isclose(comp[1].phase % 360, 121.8, atol=0.5, rtol=0)

    def test_matches_multi_plane_property(self):
        """The object holds the matrix of MultiPlaneBalancing."""
        balancer = MultiPlaneBalancing(**self.BK4)
        coefficients = balancer.get_influence_coefficients()
        assert np.array_equal(coefficients.matrix,
                              balancer.influence_coefficients)
        assert coefficients.n_planes == 2
        assert coefficients.n_sensors == 2
        assert coefficients.units == "g"
        assert "n_planes=2" in repr(coefficients)

    def test_round_trips(self):
        """Dictionary and amplitude/phase arrays keep the matrix."""
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        c = InfluenceCoefficients(matrix, units="oz")
        d = c.to_dict()
        assert set(d) == {"amplitude", "phase", "units", "angle_convention"}
        assert isinstance(d["amplitude"], list)
        back = InfluenceCoefficients.from_dict(d)
        assert np.allclose(back.matrix, matrix, rtol=1e-12, atol=1e-12)
        assert back.units == "oz"
        back = InfluenceCoefficients.from_amplitude_phase(
            *c.to_amplitude_phase(), units="oz")
        assert np.allclose(back.matrix, matrix, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("n_planes,n_sensors", [(1, 1), (2, 2), (3, 5)])
    def test_trim_recovers_remaining_unbalance(self, n_planes, n_sensors):
        """A later job: known unbalance -> known correction, no trial runs."""
        rng = np.random.default_rng(10 * n_planes + n_sensors)
        alpha = (rng.normal(size=(n_sensors, n_planes))
                 + 1j * rng.normal(size=(n_sensors, n_planes)))
        coefficients = InfluenceCoefficients(alpha)
        unbalance = rng.normal(size=n_planes) + 1j * rng.normal(size=n_planes)
        check = [VibrationVector.from_complex(z) for z in alpha @ unbalance]
        comp = coefficients.compute_compensation(check)
        assert np.allclose([_as_complex(m) for m in comp], -unbalance)
        heavy = coefficients.residual_unbalance(check)
        assert np.allclose([_as_complex(m) for m in heavy], unbalance)

    def test_predict_vibration(self):
        """Prediction for fitted weights: zero for the ideal, else known."""
        rng = np.random.default_rng(3)
        alpha = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        coefficients = InfluenceCoefficients(alpha)
        unbalance = np.array([1.5 * np.exp(0.4j), 0.8 * np.exp(-2.0j)])
        V = [VibrationVector.from_complex(z) for z in alpha @ unbalance]
        ideal = coefficients.compute_compensation(V)
        after = coefficients.predict_vibration(V, ideal)
        assert np.allclose([v.amplitude for v in after], 0, atol=1e-12)

        # one plane left empty, two weights in the other
        fitted = [None, [MassVector(0.5, 100), MassVector(0.25, -160)]]
        after = coefficients.predict_vibration(V, fitted)
        w2 = _as_complex(fitted[1][0]) + _as_complex(fitted[1][1])
        expected = alpha @ (unbalance + np.array([0, w2]))
        assert np.allclose([v.complex_num for v in after], expected)

    def test_single_objects_for_one_plane(self):
        """One plane and one point accept single objects."""
        c = InfluenceCoefficients(2.0j)
        V = VibrationVector(4.0, 0)
        comp = c.compute_compensation(V)
        assert np.isclose(comp[0].amplitude, 2.0)
        assert np.isclose(comp[0].phase, 90.0)
        after = c.predict_vibration(V, MassVector(2.0, 90.0))
        assert np.isclose(after[0].amplitude, 0, atol=1e-12)
        assert c.predict_vibration(V, None)[0].amplitude == 4.0

    def test_zero_reading_gives_none(self):
        """Nothing to add for a zero reading."""
        c = InfluenceCoefficients([[1.0]])
        assert c.compute_compensation(VibrationVector(0.0, 0.0)) == [None]

    def test_rank_deficient(self):
        """Planes that cannot be told apart raise the existing error."""
        c = InfluenceCoefficients([[1, 2], [2, 4]])
        assert c.condition_number > 1e12
        with pytest.raises(ValueError, match="rank deficient"):
            c.compute_compensation([VibrationVector(1, 0),
                                    VibrationVector(1, 0)])

    def test_ill_conditioned_warns(self):
        """Nearly equal columns warn, with an adjustable limit."""
        matrix = [[1.0, 1.0], [1.0, 1.001]]
        V = [VibrationVector(1, 0), VibrationVector(1, 10)]
        c = InfluenceCoefficients(matrix)
        with pytest.warns(IllConditionedWarning, match="ill-conditioned"):
            c.compute_compensation(V)
        with pytest.warns(IllConditionedWarning):
            assert c.check_conditioning() > 100
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InfluenceCoefficients(matrix, max_condition_number=None
                                  ).compute_compensation(V)
            InfluenceCoefficients(matrix, max_condition_number=1e6
                                  ).compute_compensation(V)
            # well-conditioned: no warning
            InfluenceCoefficients(np.eye(2)).compute_compensation(V)
        assert np.isclose(InfluenceCoefficients(np.eye(2)).condition_number,
                          1.0)

    def test_invalid_input(self):
        """Shapes, units and counts are checked."""
        with pytest.raises(ValueError, match="at least the number"):
            InfluenceCoefficients([[1, 2]])
        with pytest.raises(ValueError, match="two-dimensional"):
            InfluenceCoefficients([1, 2])
        with pytest.raises(ValueError, match="finite"):
            InfluenceCoefficients([[np.nan]])
        with pytest.raises(TypeError, match="Units must be a string"):
            InfluenceCoefficients([[1]], units=1)
        with pytest.raises(ValueError, match="same shape"):
            InfluenceCoefficients.from_amplitude_phase([[1, 2]], [[0]])
        with pytest.raises(ValueError, match="non-negative"):
            InfluenceCoefficients.from_amplitude_phase([[-1]], [[0]])
        with pytest.raises(ValueError, match="max_condition_number"):
            InfluenceCoefficients([[1]], max_condition_number=0)
        c = InfluenceCoefficients(np.eye(2))
        with pytest.raises(ValueError, match="Expected 2 readings"):
            c.compute_compensation([VibrationVector(1, 0)])
        V = [VibrationVector(1, 0), VibrationVector(1, 0)]
        with pytest.raises(ValueError, match="2 correction planes"):
            c.predict_vibration(V, [MassVector(1, 0)])
        with pytest.raises(ValueError, match="units must match"):
            c.predict_vibration(V, [MassVector(1, 0, units="oz"), None])


WITH, AGAINST = "with_rotation", "against_rotation"


def _flip(vector):
    """The same physical vector written in the other convention."""
    return convert_convention(vector, WITH, AGAINST)


class TestAngleConvention:
    """Tests for angles counted with or against the direction of rotation."""

    def test_convert_angle(self):
        """Angles are mirrored about the mark and wrapped to [0, 360)."""
        assert convert_angle(30.0, WITH, AGAINST) == 330.0
        assert convert_angle(330.0, AGAINST, WITH) == 30.0
        assert convert_angle(0.0, WITH, AGAINST) == 0.0
        assert convert_angle(180.0, WITH, AGAINST) == 180.0
        assert convert_angle(-30.0, WITH, WITH) == 330.0
        for angle in (0.0, 12.5, 90.0, 200.0, 359.0):
            there = convert_angle(angle, WITH, AGAINST)
            assert np.isclose(convert_angle(there, AGAINST, WITH), angle)

    def test_convert_convention_types(self):
        """Vectors keep type, amplitude and units."""
        v = convert_convention(VibrationVector(3.4, 116), WITH, AGAINST)
        assert isinstance(v, VibrationVector)
        assert v.amplitude == 3.4
        assert np.isclose(v.phase, 244.0)
        m = convert_convention(MassVector(2.0, -30.0, units="oz"),
                               WITH, AGAINST)
        assert isinstance(m, MassVector)
        assert (m.amplitude, m.units) == (2.0, "oz")
        assert np.isclose(m.phase, 30.0)
        same = convert_convention(MassVector(2.0, -30.0), AGAINST, AGAINST)
        assert np.isclose(same.phase, -30.0)
        with pytest.raises(TypeError, match="VibrationVector or a MassVector"):
            convert_convention(1.0, WITH, AGAINST)

    def test_invalid_convention(self):
        """Unknown conventions are refused everywhere."""
        V, m = VibrationVector(1, 0), MassVector(1, 0)
        with pytest.raises(ValueError, match="angle_convention must be one"):
            convert_angle(1.0, "clockwise", WITH)
        with pytest.raises(ValueError, match="angle_convention must be one"):
            StaticBalancing(V, V, m, angle_convention="cw")
        with pytest.raises(ValueError, match="angle_convention must be one"):
            DynamicBalancing(V, V, V, V, V, V, m, m, angle_convention="cw")
        with pytest.raises(ValueError, match="angle_convention must be one"):
            MultiPlaneBalancing([V], [[V]], [m], angle_convention="cw")
        with pytest.raises(ValueError, match="angle_convention must be one"):
            InfluenceCoefficients([[1]], angle_convention="cw")
        with pytest.raises(ValueError, match="angle_convention must be one"):
            InfluenceCoefficients([[1]]).to_convention("cw")

    def test_default_is_with_rotation(self, capsys):
        """The default convention and the printed text are unchanged."""
        balancer = StaticBalancing(VibrationVector(3.4, 116),
                                   VibrationVector(1.8, 42),
                                   MassVector(2.0, 0.0))
        assert balancer.angle_convention == WITH
        balancer.compute_compensation()
        assert capsys.readouterr().out == (
            "Compensation mass: 2.01 g at position -30.8° on the rotor.\n")
        assert "against" not in repr(balancer)

    def test_static_same_physical_correction(self, capsys):
        """B&K example 2 entered in both conventions: one physical result."""
        V_0, V_1 = VibrationVector(3.4, 116), VibrationVector(1.8, 42)
        trial = MassVector(2.0, 25.0)
        with_rot = StaticBalancing(V_0, V_1, trial).compute_compensation(
            repr=False)
        balancer = StaticBalancing(_flip(V_0), _flip(V_1), _flip(trial),
                                   angle_convention=AGAINST)
        against = balancer.compute_compensation()
        out = capsys.readouterr().out
        assert "against the direction of rotation" in out
        assert "against the direction of rotation" in repr(balancer)
        assert np.isclose(against.amplitude, with_rot.amplitude)
        # the numbers differ ...
        assert not np.isclose(against.phase, with_rot.phase)
        # ... the place on the rotor is the same
        back = convert_convention(against, AGAINST, WITH)
        assert np.isclose(back.phase, with_rot.phase)

    def test_dynamic_same_physical_correction(self, capsys):
        """B&K two-plane example entered in both conventions."""
        V = [VibrationVector(7.2, 238), VibrationVector(13.5, 296),
             VibrationVector(4.9, 114), VibrationVector(9.2, 347),
             VibrationVector(4.0, 79), VibrationVector(12.0, 292)]
        trials = [MassVector(2.5, 10.0), MassVector(2.5, -40.0)]
        with_rot = DynamicBalancing(*V, *trials).compute_compensation(
            repr=False)
        balancer = DynamicBalancing(*[_flip(v) for v in V],
                                    *[_flip(m) for m in trials],
                                    angle_convention=AGAINST)
        against = balancer.compute_compensation()
        assert "against the direction of rotation" in capsys.readouterr().out
        assert "against the direction of rotation" in repr(balancer)
        for a, w in zip(against, with_rot):
            back = convert_convention(a, AGAINST, WITH)
            assert np.isclose(back.amplitude, w.amplitude)
            assert np.isclose(back.phase, w.phase)

    def test_multi_plane_same_physical_correction(self, capsys):
        """Synthetic 3 planes x 4 points, trial masses left on."""
        rng = np.random.default_rng(8)
        alpha = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        unbalance = rng.normal(size=3) + 1j * rng.normal(size=3)
        trials = [MassVector(1.0, 20.0), MassVector(1.5, -70.0),
                  MassVector(0.8, 140.0)]
        V_0, V_trial = _simulate_runs(alpha, unbalance, trials)
        with_rot = MultiPlaneBalancing(V_0, V_trial, trials)
        against = MultiPlaneBalancing(
            [_flip(v) for v in V_0],
            [[_flip(v) for v in run] for run in V_trial],
            [_flip(m) for m in trials], angle_convention=AGAINST)
        comp_w = with_rot.compute_compensation(repr=False)
        comp_a = against.compute_compensation()
        assert "against the direction of rotation" in capsys.readouterr().out
        assert "against the direction of rotation" in repr(against)
        for a, w in zip(comp_a, comp_w):
            back = convert_convention(a, AGAINST, WITH)
            assert np.isclose(back.amplitude, w.amplitude)
            assert np.isclose(back.phase, w.phase)
        # known unbalance -> known correction, in the with-rotation frame
        assert np.allclose([_as_complex(convert_convention(a, AGAINST, WITH))
                            for a in comp_a], -unbalance)
        for a, w in zip(against.residual_vibration(),
                        with_rot.residual_vibration()):
            assert np.isclose(a.amplitude, w.amplitude, atol=1e-12)

    def test_published_instrument_convention_example(self):
        """Fan example given in the instrument convention (against rotation).

        Kelm, Kelm and Pavelek, "Rotor Balancing Tutorial", Turbomachinery
        and Pump Symposia, 2016: readings 5.6 @ 135° and 3.3 @ 238° (phase
        lag), trial 74 oz·in @ 315° against rotation; published correction
        58.24 oz·in @ 341.9°.  Reproduced to 0.2 % and 0.1°.  Counted with
        the rotation, the same place is at 18.1°.
        """
        balancer = StaticBalancing(
            VibrationVector(5.6, 135), VibrationVector(3.3, 238),
            MassVector(74.0, 315.0, units="oz·in"), angle_convention=AGAINST)
        comp = balancer.compute_compensation(repr=False)
        assert np.isclose(comp.amplitude, 58.24, atol=0, rtol=2e-3)
        assert np.isclose(comp.phase % 360, 341.9, atol=0.1, rtol=0)
        with_rot = convert_convention(comp, AGAINST, WITH)
        assert np.isclose(with_rot.phase, 18.1, atol=0.1, rtol=0)

    def test_influence_coefficients_conversion(self):
        """Stored coefficients are converted for readings of the other kind."""
        rng = np.random.default_rng(4)
        alpha = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        unbalance = np.array([1.2 * np.exp(0.7j), 0.4 * np.exp(-2.5j)])
        V = [VibrationVector.from_complex(z) for z in alpha @ unbalance]
        with_rot = InfluenceCoefficients(alpha)
        assert with_rot.angle_convention == WITH
        assert "angle_convention" not in repr(with_rot)
        assert np.array_equal(with_rot.to_convention(WITH).matrix, alpha)

        against = with_rot.to_convention(AGAINST)
        assert against.angle_convention == AGAINST
        assert "against_rotation" in repr(against)
        assert np.allclose(against.matrix, np.conj(alpha))
        comp = against.compute_compensation([_flip(v) for v in V])
        assert np.allclose(
            [_as_complex(convert_convention(m, AGAINST, WITH)) for m in comp],
            -unbalance)

        # the convention is stored and restored
        d = against.to_dict()
        assert d["angle_convention"] == AGAINST
        loaded = InfluenceCoefficients.from_dict(d)
        assert loaded.angle_convention == AGAINST
        assert np.allclose(loaded.to_convention(WITH).matrix, alpha)
        # dictionaries without the key are with the rotation
        del d["angle_convention"]
        assert InfluenceCoefficients.from_dict(d).angle_convention == WITH

    def test_convention_is_passed_on(self):
        """Coefficients taken from a balancer keep its convention."""
        V, m = VibrationVector(1, 0), MassVector(1, 0)
        balancer = MultiPlaneBalancing([V], [[VibrationVector(2, 40)]], [m],
                                       angle_convention=AGAINST)
        assert balancer.get_influence_coefficients().angle_convention == AGAINST
        c = InfluenceCoefficients.from_trial_runs(
            V, VibrationVector(2, 40), m, angle_convention=AGAINST)
        assert c.angle_convention == AGAINST
