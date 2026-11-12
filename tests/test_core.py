import numpy as np
import pytest

from pyPRB import (DynamicBalancing, MassVector, MultiPlaneBalancing,
                   StaticBalancing, VibrationVector)


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
