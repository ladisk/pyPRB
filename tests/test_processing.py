"""Tests for pyPRB.processing using synthetic signals."""

import numpy as np
import pytest

from pyPRB.processing import (
    VibrationExtractor,
    SynchronousAveraging,
    FFTExtraction,
    IQDemodulation,
    LeastSquaresFit,
    ExtractionMethod,
    detect_rotation_freq,
)
from pyPRB.processing._vibration_extractor import _rfft
from pyPRB import VibrationVector


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------

def make_trigger(f0: float, sample_rate: float, duration: float,
                 phase_deg: float = 0.0) -> np.ndarray:
    """Square wave trigger signal at frequency *f0*.

    Uses ``cos`` so that the fundamental's frequency-domain phase
    follows cosine convention.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    return np.sign(np.cos(2 * np.pi * f0 * t + np.deg2rad(phase_deg)))


def make_vibration(
    f0: float,
    amplitude: float,
    phase_deg: float,
    sample_rate: float,
    duration: float,
    noise_level: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Cosine vibration signal at frequency *f0*."""
    t = np.arange(0, duration, 1 / sample_rate)
    phase_rad = np.deg2rad(phase_deg)
    signal = amplitude * np.cos(2 * np.pi * f0 * t + phase_rad)
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        signal = signal + noise_level * rng.standard_normal(len(t))
    return signal


# SA segments at trigger crossings.  The cosine-based make_trigger has its
# rising crossing offset by -90 deg compared to a sine-based trigger.
SA_CROSSING_OFFSET = -90.0


# ===========================================================================
# 1. Low-level helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# Test _rfft
# ---------------------------------------------------------------------------

class TestRfft:
    def test_peak_amplitude(self):
        """A pure cosine should produce a peak at the correct frequency."""
        sr = 1000.0
        f0 = 25.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 3.0 * np.cos(2 * np.pi * f0 * t)

        freq, H = _rfft(signal, sr, remove_dc=True)
        idx = np.argmax(np.abs(H))
        assert np.isclose(freq[idx], f0, atol=1.5)
        assert np.isclose(np.abs(H[idx]), 3.0, rtol=0.05)

    def test_zero_padding(self):
        """Zero-padding should improve frequency resolution by the padding factor."""
        sr = 1000.0
        f0 = 25.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 3.0 * np.cos(2 * np.pi * f0 * t)

        freq_1x, H_1x = _rfft(signal, sr, nfft_mult=1)
        freq_4x, H_4x = _rfft(signal, sr, nfft_mult=4)

        # Frequency resolution should improve by 4x
        df_1x = freq_1x[1] - freq_1x[0]
        df_4x = freq_4x[1] - freq_4x[0]
        assert np.isclose(df_1x / df_4x, 4.0, rtol=0.01)

        # 4x zero-padding should have ~4x more frequency bins
        assert len(freq_4x) > 3 * len(freq_1x)

        # Amplitude at peak should still be correct
        idx = np.argmax(np.abs(H_4x))
        assert np.isclose(freq_4x[idx], f0, atol=0.5)
        assert np.isclose(np.abs(H_4x[idx]), 3.0, rtol=0.05)

    def test_invalid_nfft_mult(self):
        """Non-positive or non-integer nfft_mult should raise ValueError."""
        signal = np.zeros(100)
        with pytest.raises(ValueError, match="nfft_mult must be a positive integer"):
            _rfft(signal, 1000.0, nfft_mult=0)
        with pytest.raises(ValueError, match="nfft_mult must be a positive integer"):
            _rfft(signal, 1000.0, nfft_mult=-1)

    def test_hann_window_amplitude(self):
        """Hann window should still recover correct peak amplitude."""
        sr = 1000.0
        f0 = 25.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 3.0 * np.cos(2 * np.pi * f0 * t)

        freq, H = _rfft(signal, sr, remove_dc=True, window="hann")
        idx = np.argmax(np.abs(H))
        assert np.isclose(freq[idx], f0, atol=1.5)
        assert np.isclose(np.abs(H[idx]), 3.0, rtol=0.05)

    def test_hann_window_with_zero_padding(self):
        """Hann window + zero-padding should give correct amplitude."""
        sr = 1000.0
        f0 = 25.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 5.0 * np.cos(2 * np.pi * f0 * t)

        freq, H = _rfft(signal, sr, nfft_mult=4, remove_dc=True, window="hann")
        idx = np.argmax(np.abs(H))
        assert np.isclose(freq[idx], f0, atol=0.5)
        assert np.isclose(np.abs(H[idx]), 5.0, rtol=0.05)

    def test_invalid_window(self):
        """Unknown window name should raise ValueError."""
        signal = np.zeros(100)
        with pytest.raises(ValueError, match="Unknown window"):
            _rfft(signal, 1000.0, window="kaiser")

    def test_phase_of_peak(self):
        """Complex phase at the peak bin should match the input signal phase."""
        sr = 1000.0
        f0 = 25.0
        phase_deg = 45.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 3.0 * np.cos(2 * np.pi * f0 * t + np.deg2rad(phase_deg))

        freq, H = _rfft(signal, sr, remove_dc=True)
        idx = np.argmax(np.abs(H))
        assert np.isclose(np.degrees(np.angle(H[idx])), phase_deg, atol=2.0)

    def test_dc_removal(self):
        """remove_dc=True should suppress the DC component without affecting the peak."""
        sr = 1000.0
        f0 = 25.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 5.0 + 3.0 * np.cos(2 * np.pi * f0 * t)  # DC offset = 5

        _, H_with_dc = _rfft(signal, sr, remove_dc=False)
        _, H_no_dc = _rfft(signal, sr, remove_dc=True)

        # Without remove_dc, DC bin should be large
        assert np.abs(H_with_dc[0]) > 1.0
        # With remove_dc, DC bin should be near zero
        assert np.abs(H_no_dc[0]) < 0.1
        # Signal peak amplitude should be unaffected
        idx = np.argmax(np.abs(H_no_dc[1:]))  # skip DC bin
        assert np.isclose(np.abs(H_no_dc[idx + 1]), 3.0, rtol=0.05)


# ===========================================================================
# 2. Extraction method unit tests (method-specific)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test IQDemodulation
# ---------------------------------------------------------------------------

class TestIQDemodulation:
    """Tests for IQ demodulation (lock-in amplification) extraction."""

    def test_amplitude_and_phase(self):
        """Extract amplitude and phase from a pure cosine."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        method = IQDemodulation()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_diagnostics_values(self):
        """I/Q diagnostic values should match analytical expectations."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase_deg = 30.0
        phase_rad = np.deg2rad(phase_deg)

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase_deg, sr, dur)

        method = IQDemodulation()
        method.extract(signal, trigger, sr, f0)

        # For A*cos(wt+phi): I ~ A/2*cos(phi), Q ~ A/2*sin(phi)
        assert np.isclose(method.I_signal_, amp / 2 * np.cos(phase_rad), rtol=0.05)
        assert np.isclose(method.Q_signal_, amp / 2 * np.sin(phase_rad), rtol=0.05)
        # n_samples_used_ should be close to total samples
        expected_samples = int(sr * dur)
        assert method.n_samples_used_ == pytest.approx(expected_samples,
                                                       abs=int(sr / f0))

    def test_trim_periods(self):
        """Trimming should handle non-integer-period signal lengths."""
        f0 = 25.0
        sr = 10000.0
        amp = 2.0
        phase = 30.0
        # Duration that does NOT produce an integer number of periods
        dur = 5.03

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        method = IQDemodulation(trim_periods=True)
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_remove_dc(self):
        """DC offset should not affect the result when remove_dc=True."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur) + 5.0  # DC offset

        method = IQDemodulation(remove_dc=True)
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_short_signal_raises(self):
        """Signal shorter than one period with trim_periods=True should raise."""
        f0 = 25.0
        sr = 10000.0
        # Less than one period (1/25 = 0.04 s)
        dur = 0.03

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = IQDemodulation(trim_periods=True)
        with pytest.raises(ValueError, match="shorter than one period"):
            method.extract(signal, trigger, sr, f0)

    def test_remove_dc_false_degrades_with_dc_offset(self):
        """Without DC removal, a DC offset on a non-integer-period signal should degrade results.

        This confirms that ``remove_dc`` is load-bearing: the DC component
        leaks into the IQ estimate when the averaging window does not span
        an exact integer number of periods.
        """
        f0 = 25.0
        sr = 10000.0
        dur = 0.3  # 7.5 periods — half-period remainder maximises leakage
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur) + 10.0  # large DC

        method_on = IQDemodulation(remove_dc=True, trim_periods=False)
        amp_on, _ = method_on.extract(signal, trigger, sr, f0)

        method_off = IQDemodulation(remove_dc=False, trim_periods=False)
        amp_off, _ = method_off.extract(signal, trigger, sr, f0)

        # remove_dc=True should be close to correct
        assert np.isclose(amp_on, amp, rtol=0.05)
        # remove_dc=False should have a noticeably larger error
        assert abs(amp_off - amp) > abs(amp_on - amp)


# ---------------------------------------------------------------------------
# Test LeastSquaresFit
# ---------------------------------------------------------------------------

class TestLeastSquaresFit:
    """Tests for least-squares sine fit extraction."""

    def test_amplitude_and_phase(self):
        """Extract amplitude and phase from a pure cosine."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        method = LeastSquaresFit()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_coefficients_values(self):
        """Fitted A, B coefficients should match analytical expectations.

        For A_peak·cos(ωt + φ) the least-squares model cos/sin columns give:
        A_sig ≈ A_peak·cos(φ), B_sig ≈ -A_peak·sin(φ).
        """
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase_deg = 30.0
        phase_rad = np.deg2rad(phase_deg)

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase_deg, sr, dur)

        method = LeastSquaresFit()
        method.extract(signal, trigger, sr, f0)

        assert np.isclose(method.A_signal_, amp * np.cos(phase_rad), rtol=0.05)
        assert np.isclose(method.B_signal_, -amp * np.sin(phase_rad), rtol=0.05)

    def test_residuals_clean_signal(self):
        """Residuals should be near zero for a clean sinusoidal signal."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 2.0, 30.0, sr, dur)

        method = LeastSquaresFit()
        method.extract(signal, trigger, sr, f0)

        assert np.std(method.residuals_) < 0.01

    def test_residuals_noisy_signal(self):
        """Residual std should approximate the noise level."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        noise_level = 0.5

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 2.0, 30.0, sr, dur,
                                noise_level=noise_level)

        method = LeastSquaresFit()
        method.extract(signal, trigger, sr, f0)

        assert np.isclose(np.std(method.residuals_), noise_level, rtol=0.1)

    def test_dc_coefficient(self):
        """fit_dc=True should recover the DC offset; fit_dc=False gives None."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        dc_offset = 7.3

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 2.0, 30.0, sr, dur) + dc_offset

        method_dc = LeastSquaresFit(fit_dc=True)
        method_dc.extract(signal, trigger, sr, f0)
        assert np.isclose(method_dc.dc_signal_, dc_offset, rtol=0.01)

        method_no_dc = LeastSquaresFit(fit_dc=False)
        method_no_dc.extract(signal, trigger, sr, f0)
        assert method_no_dc.dc_signal_ is None

    def test_fit_dc_false_degrades_with_dc_offset(self):
        """Without DC fitting, a DC offset on a non-integer-period signal
        should degrade amplitude accuracy.
        """
        f0 = 25.0
        sr = 10000.0
        dur = 0.3  # 7.5 periods — half-period remainder
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur) + 10.0

        method_on = LeastSquaresFit(fit_dc=True)
        amp_on, _ = method_on.extract(signal, trigger, sr, f0)

        method_off = LeastSquaresFit(fit_dc=False)
        amp_off, _ = method_off.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_on, amp, rtol=0.05)
        assert abs(amp_off - amp) > abs(amp_on - amp)

    def test_non_integer_periods(self):
        """Least-squares fit should handle non-integer-period durations accurately."""
        f0 = 25.0
        sr = 10000.0
        amp = 2.0
        phase = 30.0
        dur = 5.03  # non-integer number of periods

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        method = LeastSquaresFit()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)


# ---------------------------------------------------------------------------
# Test SynchronousAveraging
# ---------------------------------------------------------------------------

class TestSynchronousAveraging:
    """Tests for synchronous averaging extraction."""

    CROSSING_OFFSET = SA_CROSSING_OFFSET

    def test_rising_edge_positions(self):
        """Rising crossings should be spaced at the expected period."""
        f0, sr, dur = 10.0, 10000.0, 1.0
        trigger = make_trigger(f0, sr, dur)
        method = SynchronousAveraging(edge="rising")
        crossings = method._find_trigger_crossings(trigger)
        expected_period = sr / f0  # 1000 samples
        assert len(crossings) == 10
        periods = np.diff(crossings)
        np.testing.assert_allclose(periods, expected_period, atol=1)

    def test_falling_edge_positions(self):
        """Falling crossings should be spaced at the expected period."""
        f0, sr, dur = 10.0, 10000.0, 1.0
        trigger = make_trigger(f0, sr, dur)
        method = SynchronousAveraging(edge="falling")
        crossings = method._find_trigger_crossings(trigger)
        expected_period = sr / f0
        assert len(crossings) == 10
        periods = np.diff(crossings)
        np.testing.assert_allclose(periods, expected_period, atol=1)

    def test_no_crossings(self):
        """Constant signal should raise ValueError."""
        trigger = np.ones(1000)
        method = SynchronousAveraging(edge="rising")
        with pytest.raises(ValueError, match="No rising crossings"):
            method._find_trigger_crossings(trigger)

    def test_crossing_offset(self):
        """CROSSING_OFFSET makes cosine trigger produce same crossings as sine trigger."""
        f0 = 25.0
        sr = 10000.0
        dur = 1.0

        t = np.arange(0, dur, 1 / sr)
        sin_trigger = np.sign(np.sin(2 * np.pi * f0 * t))
        cos_trigger = make_trigger(f0, sr, dur, phase_deg=self.CROSSING_OFFSET)

        method = SynchronousAveraging(edge="rising")
        sin_crossings = method._find_trigger_crossings(sin_trigger)
        cos_crossings = method._find_trigger_crossings(cos_trigger)

        np.testing.assert_allclose(sin_crossings, cos_crossings, atol=1)

    def test_average_recovers_cosine(self):
        """Averaged waveform should match the input cosine shape."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0
        trigger = make_trigger(f0, sr, dur, phase_deg=self.CROSSING_OFFSET)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = SynchronousAveraging(edge="rising")
        crossings = method._find_trigger_crossings(trigger)
        expected_period = sr / f0
        averaged, segments = method._synchronous_average(signal, crossings, expected_period)

        # Segment count should be close to f0 * dur - 1
        expected_segments = int(f0 * dur) - 1
        assert len(segments) == pytest.approx(expected_segments, abs=2)

        # Averaged length should match expected period
        assert len(averaged) == pytest.approx(expected_period, abs=1)

        # Waveform shape: high correlation with ideal cosine
        angle = np.linspace(0, 2 * np.pi, len(averaged), endpoint=False)
        ideal = np.cos(angle)
        correlation = np.corrcoef(averaged, ideal)[0, 1]
        assert correlation > 0.99

    def test_average_rejects_bad_segments(self):
        """Segments with wrong period should be rejected."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0
        trigger = make_trigger(f0, sr, dur, phase_deg=self.CROSSING_OFFSET)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = SynchronousAveraging(edge="rising")
        crossings = method._find_trigger_crossings(trigger)
        # Use a very different expected period -- should reject most segments
        with pytest.raises(ValueError, match="valid segment"):
            method._synchronous_average(signal, crossings, expected_period=100.0)

    def test_average_noise_reduction(self):
        """Averaging should reduce noise compared to a single segment."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        trigger = make_trigger(f0, sr, dur, phase_deg=self.CROSSING_OFFSET)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur, noise_level=0.5, seed=42)

        method = SynchronousAveraging(edge="rising")
        crossings = method._find_trigger_crossings(trigger)
        expected_period = sr / f0
        averaged, segments = method._synchronous_average(
            signal, crossings, expected_period,
        )

        # Compare RMS error vs ideal cosine
        angle = np.linspace(0, 2 * np.pi, len(averaged), endpoint=False)
        ideal = np.cos(angle)
        avg_rms = np.sqrt(np.mean((averaged - ideal) ** 2))
        single_rms = np.sqrt(np.mean((segments[0] - ideal) ** 2))
        assert avg_rms < single_rms * 0.5

    def test_trigger_jitter(self):
        """Synchronous averaging should handle slight period variations in the trigger."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0
        period = sr / f0

        # Build a trigger with ±3-sample jitter on each crossing
        rng = np.random.default_rng(123)
        n_crossings = int(f0 * dur)
        crossings_ideal = np.arange(n_crossings) * period
        jitter = rng.integers(-3, 4, size=n_crossings)
        crossings_jittered = (crossings_ideal + jitter).astype(int)

        n_samples = int(sr * dur)
        trigger = -np.ones(n_samples)
        for i in range(len(crossings_jittered) - 1):
            start = crossings_jittered[i]
            mid = (crossings_jittered[i] + crossings_jittered[i + 1]) // 2
            trigger[start:mid] = 1.0

        signal = make_vibration(f0, amp, phase, sr, dur)
        method = SynchronousAveraging(edge="rising")
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_extract_fundamental_known(self):
        """Extract fundamental from a single-revolution cosine."""
        N = 400
        amplitude = 2.5
        phase_deg = 45.0
        phase_rad = np.deg2rad(phase_deg)
        angle = np.linspace(0, 2 * np.pi, N, endpoint=False)
        waveform = amplitude * np.cos(angle + phase_rad)

        amp_out, phase_out = SynchronousAveraging._extract_fundamental(waveform)
        assert np.isclose(amp_out, amplitude, rtol=0.01)
        assert np.isclose(phase_out, phase_deg, atol=1.0)

    def test_extract_fundamental_zero_phase(self):
        N = 400
        angle = np.linspace(0, 2 * np.pi, N, endpoint=False)
        waveform = 1.0 * np.cos(angle)

        amp_out, phase_out = SynchronousAveraging._extract_fundamental(waveform)
        assert np.isclose(amp_out, 1.0, rtol=0.01)
        assert np.isclose(phase_out, 0.0, atol=1.0)

    def test_extract_fundamental_negative_phase(self):
        N = 400
        phase_deg = -60.0
        angle = np.linspace(0, 2 * np.pi, N, endpoint=False)
        waveform = 3.0 * np.cos(angle + np.deg2rad(phase_deg))

        amp_out, phase_out = SynchronousAveraging._extract_fundamental(waveform)
        assert np.isclose(amp_out, 3.0, rtol=0.01)
        assert np.isclose(phase_out, phase_deg, atol=1.0)

    def test_extract(self):
        """Full extraction pipeline with known signal."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur, phase_deg=self.CROSSING_OFFSET)
        signal = make_vibration(f0, amp, phase, sr, dur, noise_level=0.01)

        method = SynchronousAveraging(edge="rising")
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_stores_diagnostics(self):
        """Diagnostic attributes should contain correct values after extract."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0

        trigger = make_trigger(f0, sr, dur, phase_deg=self.CROSSING_OFFSET)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = SynchronousAveraging(edge="rising")
        method.extract(signal, trigger, sr, f0)

        # averaged_ should be one revolution long
        expected_len = int(round(sr / f0))
        assert len(method.averaged_) == expected_len

        # segments_ count should match expected revolution count
        expected_revolutions = int(f0 * dur) - 1
        assert len(method.segments_) == pytest.approx(expected_revolutions, abs=2)

        # Each segment should have the same length as averaged_
        for seg in method.segments_:
            assert len(seg) == expected_len

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be"):
            SynchronousAveraging(edge="both")

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError, match="period_tolerance"):
            SynchronousAveraging(period_tolerance=1.5)


# ---------------------------------------------------------------------------
# Test FFTExtraction
# ---------------------------------------------------------------------------

class TestFFTExtraction:
    """Tests for FFT-based extraction."""

    def test_extract_known_amplitude(self):
        """Extract amplitude from a pure cosine."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        method = FFTExtraction()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_non_bin_centered_frequency(self):
        """Non-bin-centered frequency should still extract correct amplitude.

        With nfft_mult=1 and no zero-padding, a frequency that does not
        fall exactly on an FFT bin suffers from spectral leakage, reducing
        the observed peak amplitude.  Zero-padding (nfft_mult=4) should
        recover the correct amplitude.
        """
        f0 = 25.7  # does not fall on an FFT bin for 5 s at 10 kHz
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        # Without zero-padding: amplitude is degraded by leakage
        method_coarse = FFTExtraction(nfft_mult=1)
        amp_coarse, _ = method_coarse.extract(signal, trigger, sr, f0)
        assert amp_coarse < amp * 0.95  # measurable leakage loss

        # With zero-padding: amplitude is recovered
        method_fine = FFTExtraction(nfft_mult=4)
        amp_fine, phase_fine = method_fine.extract(signal, trigger, sr, f0)
        assert np.isclose(amp_fine, amp, rtol=0.05)
        assert np.isclose(phase_fine, phase, atol=5.0)

    def test_different_nfft_mult(self):
        """Different nfft_mult values should all produce correct results."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        for mult in (1, 4, 8):
            method = FFTExtraction(nfft_mult=mult)
            amp_out, phase_out = method.extract(signal, trigger, sr, f0)
            assert np.isclose(amp_out, amp, rtol=0.05), f"nfft_mult={mult}"
            assert np.isclose(phase_out, phase, atol=5.0), f"nfft_mult={mult}"

    def test_invalid_nfft_mult(self):
        with pytest.raises(ValueError, match="nfft_mult must be a positive integer"):
            FFTExtraction(nfft_mult=0)
        with pytest.raises(ValueError, match="nfft_mult must be a positive integer"):
            FFTExtraction(nfft_mult=-1)

    def test_invalid_freq_tolerance(self):
        with pytest.raises(ValueError, match="freq_tolerance must be between"):
            FFTExtraction(freq_tolerance=0.0)
        with pytest.raises(ValueError, match="freq_tolerance must be between"):
            FFTExtraction(freq_tolerance=1.0)
        with pytest.raises(ValueError, match="freq_tolerance must be between"):
            FFTExtraction(freq_tolerance=1.5)

    def test_no_peak_in_tolerance_band(self):
        """Very narrow tolerance with coarse bins should raise ValueError."""
        f0 = 25.3
        sr = 1000.0
        dur = 0.1

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = FFTExtraction(nfft_mult=1, freq_tolerance=0.01)
        with pytest.raises(ValueError, match="No frequency bins"):
            method.extract(signal, trigger, sr, f0)

    def test_stores_diagnostics(self):
        """Diagnostic attributes should contain correct values after extract."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0
        amp = 1.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, 0.0, sr, dur)

        method = FFTExtraction()
        method.extract(signal, trigger, sr, f0)

        # peak_idx_ should point to a frequency near f0
        assert np.isclose(method.freq_[method.peak_idx_], f0, atol=1.0)
        # H_signal_ amplitude at peak should match input amplitude
        assert np.isclose(np.abs(method.H_signal_[method.peak_idx_]), amp, rtol=0.05)
        # Spectra should have same length as frequency axis
        assert len(method.freq_) == len(method.H_signal_) == len(method.H_trigger_)


# ===========================================================================
# 3. VibrationExtractor and detect_rotation_freq
# ===========================================================================

class TestDetectRotationFreq:
    def test_detect_known_frequency(self):
        f0 = 25.0
        sr = 10000.0
        trigger = make_trigger(f0, sr, 2.0)

        detected = detect_rotation_freq(trigger, sample_rate=sr, max_freq=50.0)
        assert np.isclose(detected, f0, rtol=0.02)

    def test_no_max_freq(self):
        """Should find the peak without max_freq constraint."""
        f0 = 25.0
        sr = 10000.0
        trigger = make_trigger(f0, sr, 2.0)

        detected = detect_rotation_freq(trigger, sample_rate=sr)
        assert np.isclose(detected, f0, rtol=0.02)

    def test_different_frequency(self):
        f0 = 40.0
        sr = 10000.0
        trigger = make_trigger(f0, sr, 2.0)

        detected = detect_rotation_freq(trigger, sample_rate=sr, max_freq=50.0)
        assert np.isclose(detected, f0, rtol=0.02)

    def test_with_zero_padding(self):
        """Zero-padding should improve frequency resolution."""
        f0 = 25.0
        sr = 10000.0
        trigger = make_trigger(f0, sr, 2.0)

        detected = detect_rotation_freq(trigger, sample_rate=sr, max_freq=50.0,
                                        nfft_mult=256)
        assert np.isclose(detected, f0, rtol=0.01)

    def test_empty_search_range(self):
        """max_freq below frequency resolution should raise ValueError."""
        sr = 100.0
        trigger = make_trigger(10.0, sr, 1.0)

        with pytest.raises(ValueError, match="No frequency bins found"):
            detect_rotation_freq(trigger, sample_rate=sr, max_freq=0.01)


class TestVibrationExtractor:

    def test_get_vibration_vector(self):
        """End-to-end: synthetic signals -> VibrationVector."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 1.5
        phase = -45.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur, noise_level=0.01)

        extractor = VibrationExtractor(sample_rate=sr, rotation_freq=f0)
        vv = extractor.get_vibration_vector(signal, trigger)

        assert isinstance(vv, VibrationVector)
        assert np.isclose(vv.amplitude, amp, rtol=0.05)
        # Phase wraps to [0, 360), so -45 -> 315
        expected_phase = phase % 360
        assert np.isclose(vv.phase, expected_phase, atol=5.0)

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            VibrationExtractor(sample_rate=-1, rotation_freq=25.0)

    def test_invalid_rotation_freq(self):
        with pytest.raises(ValueError, match="rotation_freq must be positive"):
            VibrationExtractor(sample_rate=10000, rotation_freq=-5)

    def test_custom_method(self):
        """A custom extraction method should be called."""

        class DummyMethod:
            def extract(self, signal, trigger, sample_rate, rotation_freq):
                return 99.0, 123.0

        extractor = VibrationExtractor(
            sample_rate=10000, rotation_freq=25.0, method=DummyMethod()
        )
        vv = extractor.get_vibration_vector(np.zeros(100), np.zeros(100))
        assert np.isclose(vv.amplitude, 99.0)
        assert np.isclose(vv.phase, 123.0)

    def test_detect_then_extract(self):
        """Detect rotation freq, then extract vector."""
        f0 = 30.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 60.0

        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur, noise_level=0.01)

        detected = detect_rotation_freq(trigger, sample_rate=sr, max_freq=50.0)
        extractor = VibrationExtractor(sample_rate=sr, rotation_freq=detected)
        vv = extractor.get_vibration_vector(signal, trigger)

        assert np.isclose(vv.amplitude, amp, rtol=0.05)
        assert np.isclose(vv.phase, phase, atol=5.0)


# ===========================================================================
# 4. Integration tests (all methods)
# ===========================================================================

@pytest.mark.parametrize("cls", [IQDemodulation, LeastSquaresFit,
                                 SynchronousAveraging, FFTExtraction])
def test_protocol_compliance(cls):
    """All extraction methods should satisfy the ExtractionMethod protocol."""
    assert isinstance(cls(), ExtractionMethod)


def _make_method_and_signals(method_cls, method_kwargs, trigger_phase_offset,
                             f0, sr, dur, amp, signal_phase, noise_level=0.0):
    """Helper to create a method instance and matching signals."""
    method = method_cls(**method_kwargs)
    trigger = make_trigger(f0, sr, dur, phase_deg=trigger_phase_offset)
    signal = make_vibration(f0, amp, signal_phase, sr, dur,
                            noise_level=noise_level)
    return method, signal, trigger


@pytest.mark.parametrize("method_cls,method_kwargs,trigger_offset", [
    (IQDemodulation, {}, 0.0),
    (LeastSquaresFit, {}, 0.0),
    (SynchronousAveraging, {"edge": "rising"}, SA_CROSSING_OFFSET),
    (FFTExtraction, {}, 0.0),
])
class TestExtractionIntegration:
    """Integration tests: full .extract() pipeline for all methods."""

    F0 = 25.0
    SR = 10000.0
    DUR = 5.0

    def test_zero_phase(self, method_cls, method_kwargs, trigger_offset):
        """Signal and trigger both at 0 deg should give ~0 phase."""
        method, signal, trigger = _make_method_and_signals(
            method_cls, method_kwargs, trigger_offset,
            self.F0, self.SR, self.DUR, 1.0, 0.0,
        )
        _, phase_out = method.extract(signal, trigger, self.SR, self.F0)
        assert np.isclose(phase_out, 0.0, atol=5.0)

    def test_negative_phase(self, method_cls, method_kwargs, trigger_offset):
        """Negative signal phase should be recovered."""
        method, signal, trigger = _make_method_and_signals(
            method_cls, method_kwargs, trigger_offset,
            self.F0, self.SR, self.DUR, 3.0, -60.0,
        )
        amp_out, phase_out = method.extract(signal, trigger, self.SR, self.F0)
        assert np.isclose(amp_out, 3.0, rtol=0.05)
        assert np.isclose(phase_out, -60.0, atol=5.0)

    def test_with_noise(self, method_cls, method_kwargs, trigger_offset):
        """Extraction should work with moderate noise."""
        method, signal, trigger = _make_method_and_signals(
            method_cls, method_kwargs, trigger_offset,
            self.F0, self.SR, self.DUR, 2.0, 45.0, noise_level=0.1,
        )
        amp_out, phase_out = method.extract(signal, trigger, self.SR, self.F0)
        assert np.isclose(amp_out, 2.0, rtol=0.05)
        assert np.isclose(phase_out, 45.0, atol=5.0)

    def test_phase_relative_to_trigger(self, method_cls, method_kwargs,
                                       trigger_offset):
        """Extracted phase should equal signal_phase - trigger_phase."""
        trigger_phase = 20.0
        signal_phase = 70.0
        amp = 2.0

        trigger = make_trigger(self.F0, self.SR, self.DUR,
                               phase_deg=trigger_phase + trigger_offset)
        signal = make_vibration(self.F0, amp, signal_phase, self.SR, self.DUR,
                                noise_level=0.01)
        method = method_cls(**method_kwargs)
        amp_out, phase_out = method.extract(signal, trigger, self.SR, self.F0)
        expected_phase = signal_phase - trigger_phase  # 50 deg
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, expected_phase, atol=5.0)


# ===========================================================================
# 5. Cross-method agreement
# ===========================================================================

class TestCrossMethodAgreement:
    """All extraction methods should agree on amplitude and phase for the same signal."""

    def test_methods_agree(self):
        """All four extraction methods should produce consistent amplitude
        and phase from the same noisy signal.
        """
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        signal = make_vibration(f0, amp, phase, sr, dur, noise_level=0.1)

        # Each method needs its own trigger (SA requires crossing offset)
        trigger_sa = make_trigger(f0, sr, dur, phase_deg=SA_CROSSING_OFFSET)
        trigger_fft = make_trigger(f0, sr, dur)

        iq_amp, iq_phase = IQDemodulation().extract(
            signal, trigger_fft, sr, f0,
        )
        ls_amp, ls_phase = LeastSquaresFit().extract(
            signal, trigger_fft, sr, f0,
        )
        sa_amp, sa_phase = SynchronousAveraging(edge="rising").extract(
            signal, trigger_sa, sr, f0,
        )
        fft_amp, fft_phase = FFTExtraction().extract(
            signal, trigger_fft, sr, f0,
        )

        # All amplitudes should be within 5% of the true value
        for a in [iq_amp, ls_amp, sa_amp, fft_amp]:
            assert np.isclose(a, amp, rtol=0.05)

        # All phases should agree within 5 degrees
        for p in [iq_phase, ls_phase, sa_phase, fft_phase]:
            assert np.isclose(p, phase, atol=5.0)
