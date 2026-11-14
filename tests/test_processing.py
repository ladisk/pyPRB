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
from pyPRB.processing._vibration_extractor import (
    _edge_reference,
    _find_trigger_edges,
    _rfft,
)
from pyPRB import VibrationVector


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------

def make_trigger(f0: float, sample_rate: float, duration: float,
                 phase_deg: float = 0.0) -> np.ndarray:
    """Square wave trigger signal at frequency *f0*.

    The rising edge is where ``2·π·f0·t + phase_deg`` is a multiple of
    360°, so a vibration ``cos(2·π·f0·t + φ)`` has the phase
    ``φ - phase_deg`` relative to the rising edge.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    return np.sign(np.sin(2 * np.pi * f0 * t + np.deg2rad(phase_deg)))


def make_pulse_trigger(f0: float, sample_rate: float, duration: float,
                       edge_time: float = 0.0, width: float = 0.1,
                       rise_samples: float = 4.0,
                       inverted: bool = False) -> np.ndarray:
    """Once-per-revolution pulse with linear edges of finite rise time.

    The rising edge crosses the midpoint (0.5) exactly at ``edge_time``
    (plus whole revolutions); the pulse lasts *width* of a revolution.
    With ``inverted=True`` the signal is ``1 - pulse`` (active low), so
    the datum is then on the falling edge.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    period = sample_rate / f0
    gap = 1.0 - width
    # Revolutions since the rising edge, in [-gap/2, 1 - gap/2), so that
    # the whole pulse and both of its edges lie inside one interval.
    u = (((t - edge_time) * f0 + gap / 2) % 1.0) - gap / 2
    rising = np.clip(0.5 + u * period / rise_samples, 0.0, 1.0)
    falling = np.clip(0.5 - (u - width) * period / rise_samples, 0.0, 1.0)
    pulse = np.minimum(rising, falling)
    return 1.0 - pulse if inverted else pulse


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

    def test_rising_edge_positions(self):
        """Rising crossings should be spaced at the expected period."""
        f0, sr, dur = 10.0, 10000.0, 1.0
        # Edges inside the record (an edge on the very first sample is not
        # a complete transition and is not counted)
        trigger = make_trigger(f0, sr, dur, phase_deg=-36.0)
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
        with pytest.raises(ValueError, match="constant"):
            method._find_trigger_crossings(trigger)

    def test_no_rising_crossings(self):
        """A single falling step has no rising edge."""
        trigger = np.r_[np.ones(500), np.zeros(500)]
        method = SynchronousAveraging(edge="rising")
        with pytest.raises(ValueError, match="No rising edges"):
            method._find_trigger_crossings(trigger)

    def test_crossings_at_sub_sample_positions(self):
        """Crossings are interpolated between samples."""
        f0, sr, dur = 23.7, 10000.0, 1.0
        edge_time = 0.00123
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=edge_time)
        crossings = SynchronousAveraging()._find_trigger_crossings(trigger)
        expected = (edge_time + np.arange(len(crossings)) / f0) * sr
        np.testing.assert_allclose(crossings, expected, atol=1e-6)

    def test_average_recovers_cosine(self):
        """Averaged waveform should match the input cosine shape."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0
        trigger = make_trigger(f0, sr, dur)
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
        trigger = make_trigger(f0, sr, dur)
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
        trigger = make_trigger(f0, sr, dur)
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

        trigger = make_trigger(f0, sr, dur)
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

        trigger = make_trigger(f0, sr, dur)
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
            detect_rotation_freq(trigger, sample_rate=sr, max_freq=0.01,
                                 method="fft")

    @pytest.mark.parametrize("f0", [23.7, 31.3, 40.0])
    @pytest.mark.parametrize("width", [0.5, 0.2, 0.1, 0.05, 0.02])
    @pytest.mark.parametrize("method, rtol", [("edges", 1e-6),
                                              ("fft", 0.01)])
    def test_short_pulses_off_bin(self, f0, width, method, rtol):
        """Short pulses at frequencies between FFT bins (issue #12)."""
        sr, dur = 10000.0, 5.0
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=0.0013,
                                     width=width)
        detected = detect_rotation_freq(trigger, sample_rate=sr,
                                        max_freq=150.0, method=method)
        assert np.isclose(detected, f0, rtol=rtol)

    def test_fft_without_max_freq(self):
        """The sub-harmonic check also works when searching to Nyquist."""
        trigger = make_pulse_trigger(23.7, 10000.0, 5.0, width=0.05)
        detected = detect_rotation_freq(trigger, sample_rate=10000.0,
                                        method="fft")
        assert np.isclose(detected, 23.7, rtol=0.01)

    def test_fft_pure_sine(self):
        """A sinusoidal trigger has no sub-harmonic content."""
        sr = 10000.0
        t = np.arange(0, 2.0, 1 / sr)
        trigger = np.sin(2 * np.pi * 37.0 * t)
        assert np.isclose(detect_rotation_freq(trigger, sr, method="fft"),
                          37.0, rtol=0.01)
        assert np.isclose(detect_rotation_freq(trigger, sr), 37.0, rtol=1e-6)

    def test_edges_missing_pulse(self):
        """A missing pulse does not change the edge-based estimate."""
        f0, sr, dur = 23.7, 10000.0, 2.0
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=0.0013, width=0.1)
        t = np.arange(len(trigger)) / sr
        revolution = np.floor((t - 0.0013) * f0 + 0.5)
        trigger[revolution == 10] = 0.0
        assert np.isclose(detect_rotation_freq(trigger, sr), f0, rtol=1e-6)

    def test_edges_noisy_trigger(self):
        """Noise on slow edges does not add edges."""
        f0, sr = 23.7, 10000.0
        trigger = make_pulse_trigger(f0, sr, 5.0, edge_time=0.0013,
                                     width=0.05, rise_samples=40)
        trigger += 0.05 * np.random.default_rng(1).standard_normal(len(trigger))
        assert np.isclose(detect_rotation_freq(trigger, sr), f0, rtol=1e-4)

    def test_edges_falling(self):
        """Active-low pulse with edge='falling'."""
        f0, sr = 31.3, 10000.0
        trigger = make_pulse_trigger(f0, sr, 2.0, edge_time=0.0013,
                                     width=0.05, inverted=True)
        assert np.isclose(detect_rotation_freq(trigger, sr, edge="falling"),
                          f0, rtol=1e-6)

    def test_edges_too_few(self):
        trigger = np.r_[np.zeros(100), np.ones(100), np.zeros(100)]
        with pytest.raises(ValueError, match="At least two rising edges"):
            detect_rotation_freq(trigger, 1000.0)

    def test_edges_above_max_freq(self):
        trigger = make_trigger(40.0, 10000.0, 1.0)
        with pytest.raises(ValueError, match="exceeds max_freq"):
            detect_rotation_freq(trigger, 10000.0, max_freq=30.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            detect_rotation_freq(np.zeros(10), 1000.0, method="peak")

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be"):
            detect_rotation_freq(np.zeros(10), 1000.0, edge="both")


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
    (SynchronousAveraging, {"edge": "rising"}, 0.0),
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

        # All methods use the same trigger and the same phase reference
        trigger_sa = make_trigger(f0, sr, dur)
        trigger_fft = trigger_sa

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


# ===========================================================================
# Angle convention
# ===========================================================================

class TestExtractorAngleConvention:
    """Phase lead (with rotation) or phase lag (against rotation)."""

    def test_phase_lag(self):
        """The against-rotation phase is the lag: 360° minus the lead."""
        f0, sr, dur = 25.0, 10000.0, 4.0
        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 2.0, 70.0, sr, dur)
        lead = VibrationExtractor(sr, f0).get_vibration_vector(
            signal, trigger)
        lag = VibrationExtractor(
            sr, f0, angle_convention="against_rotation"
        ).get_vibration_vector(signal, trigger)
        assert np.isclose(lead.phase, 70.0, atol=0.5)
        assert np.isclose(lag.phase, 290.0, atol=0.5)
        assert np.isclose(lag.amplitude, lead.amplitude)

    def test_default_and_invalid(self):
        extractor = VibrationExtractor(10000, 25.0)
        assert extractor.angle_convention == "with_rotation"
        with pytest.raises(ValueError, match="angle_convention must be one"):
            VibrationExtractor(10000, 25.0, angle_convention="clockwise")

    @pytest.mark.parametrize("method_cls", [IQDemodulation, LeastSquaresFit,
                                            FFTExtraction])
    def test_same_physical_correction_from_signals(self, method_cls):
        """Signals -> correction: the same place on the rotor both ways.

        A heavy spot of 3 g sits 140° ahead of the mark (in the direction
        of rotation); the trial mass of 2 g is mounted 60° ahead of the
        mark.  One user counts with the rotation, the other against it.
        Both must be told to add 3 g opposite the heavy spot.
        """
        from pyPRB import MassVector, StaticBalancing, convert_convention
        f0, sr, dur = 25.0, 10000.0, 4.0
        heavy = 3.0 * np.exp(1j * np.deg2rad(140.0))
        trial = 2.0 * np.exp(1j * np.deg2rad(60.0))
        response = 0.8 * np.exp(1j * np.deg2rad(-35.0))  # per gram, with lag

        def record(unbalance):
            v = response * unbalance
            return make_vibration(f0, abs(v), np.rad2deg(np.angle(v)),
                                  sr, dur)

        trigger = make_trigger(f0, sr, dur)
        run_0, run_1 = record(heavy), record(heavy + trial)

        results = {}
        for convention, trial_position in (("with_rotation", 60.0),
                                           ("against_rotation", 300.0)):
            extractor = VibrationExtractor(sr, f0, method=method_cls(),
                                           angle_convention=convention)
            balancer = StaticBalancing(
                extractor.get_vibration_vector(run_0, trigger),
                extractor.get_vibration_vector(run_1, trigger),
                MassVector(2.0, trial_position),
                angle_convention=convention)
            results[convention] = balancer.compute_compensation(repr=False)

        with_rot = results["with_rotation"]
        against = results["against_rotation"]
        assert np.isclose(with_rot.amplitude, 3.0, rtol=1e-3)
        assert np.isclose(with_rot.phase, 140.0 - 180.0, atol=0.1)
        assert np.isclose(against.amplitude, 3.0, rtol=1e-3)
        assert np.isclose(against.phase, 40.0, atol=0.1)       # 360° - 320°
        back = convert_convention(against, "against_rotation",
                                  "with_rotation")
        assert np.isclose(back.phase, with_rot.phase, atol=1e-6)


# ===========================================================================
# The 1x vector with its scatter, speed variation, harmonic content
# ===========================================================================

class TestVectorScatter:
    """The 1x vector over consecutive parts of one record."""

    F0, SR, DUR = 25.0, 10000.0, 8.0

    def _extractor(self, **kwargs):
        return VibrationExtractor(self.SR, self.F0, **kwargs)

    def test_steady_signal(self):
        """A steady record: all parts coincide with the single vector.

        A sinusoidal once-per-revolution reference is used, so that the
        sample-wide timing steps of a square trigger do not show.
        """
        t = np.arange(0, self.DUR, 1 / self.SR)
        trigger = np.cos(2 * np.pi * self.F0 * t)
        signal = make_vibration(self.F0, 2.0, 70.0, self.SR, self.DUR)
        extractor = self._extractor()
        vectors = extractor.get_vibration_vectors(signal, trigger, n_parts=8)
        assert len(vectors) == 8
        assert all(isinstance(v, VibrationVector) for v in vectors)
        single = extractor.get_vibration_vector(signal, trigger)
        scatter = extractor.get_vector_scatter(signal, trigger, n_parts=8)
        assert np.isclose(scatter.mean.amplitude, single.amplitude, rtol=1e-6)
        assert np.isclose(scatter.mean.phase, single.phase, atol=1e-4)
        assert scatter.amplitude_spread < 1e-6
        assert scatter.phase_spread < 1e-4
        assert scatter.max_deviation < 1e-6
        assert scatter.relative_deviation < 1e-6

    def test_published_example_through_signals(self):
        """Example 2 of the B&K application note from noisy records.

        J. Vaughan, "Static and Dynamic Balancing using portable measuring
        equipment", Brüel & Kjær: readings 3.4 @ 116° and 1.8 @ 42° with a
        2 g trial mass, published correction 2.03 g at -31°.  The mean
        vectors of ten parts of each simulated record give the correction
        to 5 % (the tolerance of the existing tests of this example).
        """
        from pyPRB import MassVector, StaticBalancing
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        run_0 = make_vibration(self.F0, 3.4, 116.0, self.SR, self.DUR,
                               noise_level=0.3, seed=1)
        run_1 = make_vibration(self.F0, 1.8, 42.0, self.SR, self.DUR,
                               noise_level=0.3, seed=2)
        extractor = self._extractor()
        V_0 = extractor.get_vector_scatter(run_0, trigger)
        V_1 = extractor.get_vector_scatter(run_1, trigger)
        assert len(V_0.vectors) == 10
        assert 0 < V_0.relative_deviation < 0.05
        comp = StaticBalancing(V_0.mean, V_1.mean,
                               MassVector(2.0, 0.0)).compute_compensation(
            repr=False)
        assert np.isclose(comp.amplitude, 2.03, atol=0, rtol=0.05)
        assert np.isclose(comp.phase, -31.0, atol=0, rtol=0.05)

    def test_known_scatter(self):
        """Known unsteadiness -> known cloud.

        The amplitude grows linearly from 1 to 2 over the record and the
        phase stays at 40°: four parts have the mean amplitudes 1.125,
        1.375, 1.625 and 1.875.
        """
        t = np.arange(0, self.DUR, 1 / self.SR)
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        amplitude = 1.0 + t / self.DUR
        signal = amplitude * np.cos(2 * np.pi * self.F0 * t + np.deg2rad(40))
        scatter = self._extractor().get_vector_scatter(signal, trigger,
                                                       n_parts=4)
        amplitudes = [v.amplitude for v in scatter.vectors]
        assert np.allclose(amplitudes, [1.125, 1.375, 1.625, 1.875],
                           rtol=2e-3)
        assert np.isclose(scatter.mean.amplitude, 1.5, rtol=2e-3)
        assert np.isclose(scatter.mean.phase, 40.0, atol=0.3)
        assert np.isclose(scatter.amplitude_spread, 0.75, rtol=5e-3)
        assert np.isclose(scatter.max_deviation, 0.375, rtol=5e-3)
        assert np.isclose(scatter.relative_deviation, 0.25, rtol=5e-3)
        assert scatter.phase_spread < 0.5

    def test_known_phase_drift(self):
        """The phase drifts by 20° over the record at constant amplitude."""
        t = np.arange(0, self.DUR, 1 / self.SR)
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        phase = np.deg2rad(100.0 + 20.0 * t / self.DUR)
        signal = 2.0 * np.cos(2 * np.pi * self.F0 * t + phase)
        scatter = self._extractor().get_vector_scatter(signal, trigger,
                                                       n_parts=4)
        phases = [v.phase for v in scatter.vectors]
        assert np.allclose(phases, [102.5, 107.5, 112.5, 117.5], atol=0.3)
        assert np.isclose(scatter.phase_spread, 15.0, atol=0.3)
        assert scatter.amplitude_spread < 0.01

    def test_noise_scatter_shrinks_with_longer_parts(self):
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        signal = make_vibration(self.F0, 2.0, 70.0, self.SR, self.DUR,
                                noise_level=1.0, seed=3)
        extractor = self._extractor()
        many = extractor.get_vector_scatter(signal, trigger, n_parts=40)
        few = extractor.get_vector_scatter(signal, trigger, n_parts=4)
        assert many.max_deviation > few.max_deviation > 0
        assert np.isclose(many.mean.amplitude, 2.0, rtol=0.02)

    @pytest.mark.parametrize("method_cls", [IQDemodulation, LeastSquaresFit,
                                            FFTExtraction])
    def test_uses_the_chosen_method(self, method_cls):
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        signal = make_vibration(self.F0, 2.0, 70.0, self.SR, self.DUR,
                                noise_level=0.1)
        scatter = self._extractor(method=method_cls()).get_vector_scatter(
            signal, trigger, n_parts=5)
        assert np.isclose(scatter.mean.amplitude, 2.0, rtol=0.02)
        assert np.isclose(scatter.mean.phase, 70.0, atol=1.0)

    def test_against_rotation(self):
        """The cloud follows the angle convention of the extractor."""
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        signal = make_vibration(self.F0, 2.0, 70.0, self.SR, self.DUR)
        scatter = self._extractor(
            angle_convention="against_rotation").get_vector_scatter(
            signal, trigger, n_parts=4)
        assert np.isclose(scatter.mean.phase, 290.0, atol=0.5)

    def test_invalid(self):
        trigger = make_trigger(self.F0, self.SR, 1.0)
        signal = make_vibration(self.F0, 2.0, 70.0, self.SR, 1.0)
        extractor = self._extractor()
        with pytest.raises(ValueError, match="at least 2"):
            extractor.get_vibration_vectors(signal, trigger, n_parts=1)
        with pytest.raises(ValueError, match="same length"):
            extractor.get_vibration_vectors(signal, trigger[:-5])
        with pytest.raises(ValueError, match="at least two revolutions"):
            extractor.get_vibration_vectors(signal, trigger, n_parts=20)


class TestRotationFreqVariation:
    """Speed found in the trigger record and its variation."""

    def test_steady_speed(self):
        from pyPRB import SpeedVariation, rotation_freq_variation
        trigger = make_trigger(25.0, 10000.0, 4.0, phase_deg=30.0)
        speed = rotation_freq_variation(trigger, 10000.0)
        assert isinstance(speed, SpeedVariation)
        assert np.isclose(speed.mean, 25.0, rtol=1e-4)
        assert speed.n_revolutions == 99
        assert len(speed.values) == 9
        assert speed.relative_range < 1e-3
        assert speed.minimum <= speed.mean <= speed.maximum

    def test_agrees_with_detect_rotation_freq(self):
        from pyPRB import rotation_freq_variation
        trigger = make_trigger(33.3, 10000.0, 6.0)
        f0 = detect_rotation_freq(trigger, 10000.0, max_freq=100.0,
                                  nfft_mult=16)
        speed = rotation_freq_variation(trigger, 10000.0)
        assert np.isclose(speed.mean, f0, rtol=2e-3)
        assert np.isclose(speed.mean, 33.3, rtol=1e-4)

    def test_known_run_up(self):
        """Speed rising linearly from 24 Hz to 26 Hz over 10 s."""
        from pyPRB import rotation_freq_variation
        sr, dur = 10000.0, 10.0
        t = np.arange(0, dur, 1 / sr)
        angle = 2 * np.pi * (24.0 * t + 0.5 * (2.0 / dur) * t ** 2)
        trigger = np.sign(np.sin(angle))
        speed = rotation_freq_variation(trigger, sr, revolutions_per_value=5)
        assert np.isclose(speed.mean, 25.0, rtol=2e-3)
        assert np.isclose(speed.minimum, 24.0, atol=0.06)
        assert np.isclose(speed.maximum, 26.0, atol=0.06)
        assert np.isclose(speed.relative_range, 0.08, atol=0.005)
        assert np.all(np.diff(speed.values) > -0.02)       # rising
        assert np.isclose(speed.std, 2.0 / np.sqrt(12), rtol=0.05)

    def test_smooth_trigger_and_falling_edge(self):
        from pyPRB import rotation_freq_variation
        t = np.arange(0, 4.0, 1 / 2000.0)
        trigger = 2.5 + np.cos(2 * np.pi * 12.5 * t)         # with an offset
        for edge in ("rising", "falling"):
            speed = rotation_freq_variation(trigger, 2000.0,
                                            revolutions_per_value=1,
                                            edge=edge)
            assert np.isclose(speed.mean, 12.5, rtol=1e-5)
            assert speed.relative_range < 1e-4

    def test_invalid(self):
        from pyPRB import rotation_freq_variation
        trigger = make_trigger(25.0, 10000.0, 0.2)           # 5 revolutions
        with pytest.raises(ValueError, match="whole revolutions"):
            rotation_freq_variation(trigger, 10000.0)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            rotation_freq_variation(trigger, 0.0)
        with pytest.raises(ValueError, match="'rising' or 'falling'"):
            rotation_freq_variation(trigger, 10000.0, edge="both")
        with pytest.raises(ValueError, match="positive integer"):
            rotation_freq_variation(trigger, 10000.0,
                                    revolutions_per_value=0)


class TestHarmonicContent:
    """Content at 1x, 2x, 3x and the overall level."""

    F0, SR, DUR = 25.0, 10000.0, 4.0

    def _signal(self, a1=3.0, a2=1.0, a3=0.5, noise=0.0, offset=0.0):
        t = np.arange(0, self.DUR, 1 / self.SR)
        x = (a1 * np.cos(2 * np.pi * self.F0 * t + 0.4)
             + a2 * np.cos(2 * np.pi * 2 * self.F0 * t - 1.0)
             + a3 * np.cos(2 * np.pi * 3 * self.F0 * t + 2.0) + offset)
        if noise:
            x = x + noise * np.random.default_rng(0).standard_normal(len(t))
        return x

    def test_known_harmonics(self):
        from pyPRB import HarmonicContent
        extractor = VibrationExtractor(self.SR, self.F0)
        content = extractor.get_harmonic_content(self._signal(offset=7.0))
        assert isinstance(content, HarmonicContent)
        assert content.orders == (1, 2, 3)
        assert np.allclose(content.amplitudes, [3.0, 1.0, 0.5], rtol=1e-9)
        overall = np.sqrt((9.0 + 1.0 + 0.25) / 2)
        assert np.isclose(content.overall_rms, overall, rtol=1e-9)
        assert np.isclose(content.fraction_1x, 3.0 / np.sqrt(2) / overall)

    def test_agrees_with_extracted_vector(self):
        extractor = VibrationExtractor(self.SR, self.F0)
        signal = self._signal(noise=0.2)
        trigger = make_trigger(self.F0, self.SR, self.DUR)
        vector = extractor.get_vibration_vector(signal, trigger)
        content = extractor.get_harmonic_content(signal)
        assert np.isclose(content.amplitudes[0], vector.amplitude, rtol=1e-6)

    def test_noise_lowers_the_1x_fraction(self):
        extractor = VibrationExtractor(self.SR, self.F0)
        clean = extractor.get_harmonic_content(self._signal(a2=0, a3=0))
        assert np.isclose(clean.fraction_1x, 1.0)
        noisy = extractor.get_harmonic_content(
            self._signal(a2=0, a3=0, noise=3.0))
        expected = (3.0 / np.sqrt(2)) / np.sqrt(4.5 + 9.0)
        assert np.isclose(noisy.fraction_1x, expected, rtol=0.03)

    @pytest.mark.parametrize("orders", [(0.5, 1, 1.5), (1, 2.5), (0, 1), ()])
    def test_non_integer_orders_raise(self, orders):
        extractor = VibrationExtractor(self.SR, self.F0)
        with pytest.raises(ValueError, match="orders must be positive integers"):
            extractor.get_harmonic_content(self._signal(), orders=orders)

    def test_integer_valued_floats_accepted(self):
        extractor = VibrationExtractor(self.SR, self.F0)
        content = extractor.get_harmonic_content(self._signal(),
                                                 orders=(1.0, 2.0))
        assert content.orders == (1, 2)

    def test_other_orders(self):
        extractor = VibrationExtractor(self.SR, self.F0)
        content = extractor.get_harmonic_content(self._signal(),
                                                 orders=(2, 4))
        assert np.allclose(content.amplitudes, [1.0, 0.0], atol=1e-9)
        assert np.isnan(content.fraction_1x)

    def test_invalid(self):
        extractor = VibrationExtractor(self.SR, self.F0)
        with pytest.raises(ValueError, match="positive integers"):
            extractor.get_harmonic_content(self._signal(), orders=(0, 1))
        with pytest.raises(ValueError, match="half the sampling rate"):
            extractor.get_harmonic_content(self._signal(), orders=(1, 300))
        with pytest.raises(ValueError, match="shorter than one revolution"):
            extractor.get_harmonic_content(np.zeros(100))


# ===========================================================================
# 6. Phase reference: the trigger edge
# ===========================================================================

ALL_METHODS = [IQDemodulation, LeastSquaresFit, SynchronousAveraging,
               FFTExtraction]


class TestFindTriggerEdges:
    """Tests for the shared sub-sample edge detector."""

    def test_rising_and_falling(self):
        f0, sr, dur = 23.7, 10000.0, 1.0
        edge_time, width = 0.00123, 0.3
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=edge_time,
                                     width=width)
        rising = _find_trigger_edges(trigger, "rising")
        falling = _find_trigger_edges(trigger, "falling")
        np.testing.assert_allclose(
            rising, (edge_time + np.arange(len(rising)) / f0) * sr, atol=1e-6)
        first_fall = edge_time + width / f0
        np.testing.assert_allclose(
            falling, (first_fall + np.arange(len(falling)) / f0) * sr,
            atol=1e-6)

    def test_threshold_independent_of_levels(self):
        """Offset and gain of the trigger do not move the edges."""
        f0, sr, dur = 25.0, 10000.0, 1.0
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=0.0007, width=0.05)
        np.testing.assert_allclose(_find_trigger_edges(trigger),
                                   _find_trigger_edges(5.0 * trigger - 2.0))

    @pytest.mark.parametrize("edge", ["rising", "falling"])
    def test_hysteresis_suppresses_chatter(self, edge):
        """Noise on slow edges gives no extra edges."""
        f0, sr, dur = 25.0, 10000.0, 1.0
        edge_time, width = 0.005, 0.5
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=edge_time,
                                     width=width, rise_samples=40)
        trigger = trigger + 0.05 * np.random.default_rng(0).standard_normal(
            len(trigger))
        # Plain midpoint crossings would chatter on every edge
        above = trigger > 0.5
        assert np.sum(~above[:-1] & above[1:]) > 2 * int(f0 * dur)
        edges = _find_trigger_edges(trigger, edge)
        assert len(edges) == int(f0 * dur)
        first = edge_time if edge == "rising" else edge_time + width / f0
        expected = (first + np.arange(len(edges)) / f0) * sr
        # Noise of 5 % on a 40-sample edge moves single crossings by a few
        # samples (about 2 samples standard deviation); they must stay on
        # the right edge.
        np.testing.assert_allclose(edges, expected, atol=6.0)

    def test_min_spacing(self):
        """Edges closer than min_spacing to the previous one are dropped."""
        trigger = np.r_[np.zeros(10), np.ones(5), np.zeros(5), np.ones(10),
                        np.zeros(30), np.ones(10)]
        assert len(_find_trigger_edges(trigger)) == 3
        np.testing.assert_allclose(_find_trigger_edges(trigger, min_spacing=20),
                                   [9.5, 59.5])

    def test_constant_trigger(self):
        with pytest.raises(ValueError, match="constant"):
            _find_trigger_edges(np.zeros(100))

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be"):
            _find_trigger_edges(np.r_[0.0, 1.0], edge="both")

    def test_edge_reference_phase(self):
        """The reference cosine has its maxima on the trigger edges."""
        f0, sr, dur = 23.7, 10000.0, 1.0
        edge_time = 0.00123
        trigger = make_pulse_trigger(f0, sr, dur, edge_time=edge_time)
        ref = _edge_reference(trigger, sr, f0)
        t = np.arange(len(trigger)) / sr
        np.testing.assert_allclose(ref, np.cos(2 * np.pi * f0 * (t - edge_time)),
                                   atol=1e-9)


@pytest.mark.parametrize("method_cls", [IQDemodulation, LeastSquaresFit,
                                        FFTExtraction])
def test_invalid_edge_option(method_cls):
    with pytest.raises(ValueError, match="edge must be"):
        method_cls(edge="both")


class TestPhaseReference:
    """All methods measure the phase from the trigger edge (issue #10)."""

    SR = 10000.0
    DUR = 5.0

    def _phases(self, signal, trigger, f0, edge="rising"):
        return [m(edge=edge).extract(signal, trigger, self.SR, f0)
                for m in ALL_METHODS]

    @pytest.mark.parametrize("width", [0.02, 0.05, 0.2, 0.5, 0.8])
    def test_independent_of_pulse_width(self, width):
        """The phase does not depend on the duty cycle of the trigger."""
        f0, amp, phase = 25.0, 3.4, 116.0
        signal = make_vibration(f0, amp, phase, self.SR, self.DUR)
        trigger = make_pulse_trigger(f0, self.SR, self.DUR, width=width)
        for amp_out, phase_out in self._phases(signal, trigger, f0):
            assert np.isclose(amp_out, amp, rtol=1e-3)
            assert np.isclose(phase_out, phase, atol=0.05)

    @pytest.mark.parametrize("f0, edge_time, phase", [
        (25.0, 0.0, 30.0),
        (23.7, 0.00123, -40.0),     # non-integer samples per revolution
        (41.3, 0.0151, 170.0),
    ])
    def test_phase_from_edge(self, f0, edge_time, phase):
        """A vibration cos(ω(t - t_edge) + φ) gives φ in every method."""
        t = np.arange(0, self.DUR, 1 / self.SR)
        signal = 2.0 * np.cos(2 * np.pi * f0 * (t - edge_time)
                              + np.deg2rad(phase))
        trigger = make_pulse_trigger(f0, self.SR, self.DUR,
                                     edge_time=edge_time, width=0.1)
        for amp_out, phase_out in self._phases(signal, trigger, f0):
            assert np.isclose(amp_out, 2.0, rtol=1e-3)
            diff = (phase_out - phase + 180.0) % 360.0 - 180.0
            assert abs(diff) < 0.05

    def test_falling_edge_of_inverted_pulse(self):
        """An active-low pulse with edge='falling' gives the same phase."""
        f0, phase = 25.0, 116.0
        signal = make_vibration(f0, 3.4, phase, self.SR, self.DUR)
        trigger = make_pulse_trigger(f0, self.SR, self.DUR, width=0.05,
                                     inverted=True)
        for _, phase_out in self._phases(signal, trigger, f0, edge="falling"):
            assert np.isclose(phase_out, phase, atol=0.05)

    def test_methods_agree_with_noise_and_harmonics(self):
        """Noisy signal with 2x content and a noisy pulse trigger."""
        f0, amp, phase = 23.7, 1.5, 250.0
        t = np.arange(0, self.DUR, 1 / self.SR)
        rng = np.random.default_rng(5)
        signal = (amp * np.cos(2 * np.pi * f0 * t + np.deg2rad(phase))
                  + 0.4 * np.cos(4 * np.pi * f0 * t + 1.0)
                  + 0.2 * rng.standard_normal(len(t)))
        trigger = make_pulse_trigger(f0, self.SR, self.DUR, width=0.05)
        trigger = trigger + 0.02 * rng.standard_normal(len(t))
        results = self._phases(signal, trigger, f0)
        for amp_out, phase_out in results:
            assert np.isclose(amp_out, amp, rtol=0.02)
            diff = (phase_out - phase + 180.0) % 360.0 - 180.0
            assert abs(diff) < 1.0
        phases = np.array([p for _, p in results])
        spread = np.ptp((phases - phases[0] + 180.0) % 360.0 - 180.0)
        assert spread < 1.0
