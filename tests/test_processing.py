"""Tests for pyPRB.processing using synthetic signals."""

import numpy as np
import pytest

from pyPRB.processing import (
    VibrationExtractor,
    SynchronousAveraging,
    FFTExtraction,
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

    Uses ``sin`` so that the rising crossing aligns with *phase_deg* and
    the extracted phase equals ``signal_phase - trigger_phase``.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    return np.sign(np.sin(2 * np.pi * f0 * t + np.deg2rad(phase_deg)))


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
        """Zero-padding should give finer frequency resolution."""
        sr = 1000.0
        f0 = 25.0
        t = np.arange(0, 1.0, 1 / sr)
        signal = 3.0 * np.cos(2 * np.pi * f0 * t)

        freq_1x, H_1x = _rfft(signal, sr, nfft_mult=1)
        freq_4x, H_4x = _rfft(signal, sr, nfft_mult=4)

        # 4x zero-padding should have 4x more frequency bins
        assert len(freq_4x) > len(freq_1x)
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


# ---------------------------------------------------------------------------
# Test SynchronousAveraging helper methods
# ---------------------------------------------------------------------------

class TestFindTriggerCrossings:
    def setup_method(self):
        self.method = SynchronousAveraging(edge="rising")

    def test_rising_edge_count(self):
        """Square wave at 10 Hz for 1 s should have ~10 rising crossings."""
        trigger = make_trigger(10.0, 10000.0, 1.0)
        crossings = self.method._find_trigger_crossings(trigger)
        assert np.isclose(len(crossings), 10, atol=1)

    def test_falling_edge_count(self):
        trigger = make_trigger(10.0, 10000.0, 1.0)
        method = SynchronousAveraging(edge="falling")
        crossings = method._find_trigger_crossings(trigger)
        assert np.isclose(len(crossings), 10, atol=1)

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be"):
            SynchronousAveraging(edge="both")

    def test_no_crossings(self):
        """Constant signal should raise ValueError."""
        trigger = np.ones(1000)
        with pytest.raises(ValueError, match="No rising crossings"):
            self.method._find_trigger_crossings(trigger)


class TestSynchronousAverage:
    def setup_method(self):
        self.method = SynchronousAveraging(edge="rising")

    def test_recovers_cosine(self):
        """Synchronous average of a pure cosine should recover it."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0
        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        crossings = self.method._find_trigger_crossings(trigger)
        expected_period = sr / f0
        averaged, segments = self.method._synchronous_average(signal, crossings, expected_period)

        # The averaged waveform should have roughly the same amplitude
        assert np.isclose(np.max(averaged), 1.0, rtol=0.1)
        assert len(segments) > 0

    def test_rejects_bad_segments(self):
        """Segments with wrong period should be rejected."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0
        trigger = make_trigger(f0, sr, dur)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        crossings = self.method._find_trigger_crossings(trigger)
        # Use a very different expected period — should reject most segments
        with pytest.raises(ValueError, match="valid segment"):
            self.method._synchronous_average(signal, crossings, expected_period=100.0)


class TestExtractFundamental:
    def test_known_amplitude_and_phase(self):
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

    def test_zero_phase(self):
        N = 400
        angle = np.linspace(0, 2 * np.pi, N, endpoint=False)
        waveform = 1.0 * np.cos(angle)

        amp_out, phase_out = SynchronousAveraging._extract_fundamental(waveform)
        assert np.isclose(amp_out, 1.0, rtol=0.01)
        assert np.isclose(phase_out, 0.0, atol=1.0)

    def test_negative_phase(self):
        N = 400
        phase_deg = -60.0
        angle = np.linspace(0, 2 * np.pi, N, endpoint=False)
        waveform = 3.0 * np.cos(angle + np.deg2rad(phase_deg))

        amp_out, phase_out = SynchronousAveraging._extract_fundamental(waveform)
        assert np.isclose(amp_out, 3.0, rtol=0.01)
        assert np.isclose(phase_out, phase_deg, atol=1.0)


# ---------------------------------------------------------------------------
# Test SynchronousAveraging
# ---------------------------------------------------------------------------

class TestSynchronousAveraging:
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

    def test_phase_relative_to_trigger(self):
        """Extracted phase should equal signal_phase - trigger_phase."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        trigger_phase = 20.0
        signal_phase = 70.0

        trigger = make_trigger(f0, sr, dur, phase_deg=trigger_phase)
        signal = make_vibration(f0, amp, signal_phase, sr, dur, noise_level=0.01)

        method = SynchronousAveraging(edge="rising")
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        expected_phase = signal_phase - trigger_phase  # 50°
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, expected_phase, atol=5.0)

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match="edge must be"):
            SynchronousAveraging(edge="invalid")

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError, match="period_tolerance"):
            SynchronousAveraging(period_tolerance=1.5)


# ---------------------------------------------------------------------------
# Test VibrationExtractor
# ---------------------------------------------------------------------------

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
        """End-to-end: synthetic signals → VibrationVector."""
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
        # Phase wraps to [0, 360), so -45 → 315
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


# ---------------------------------------------------------------------------
# Test FFTExtraction
# ---------------------------------------------------------------------------

class TestFFTExtraction:
    """Tests for FFT-based extraction.

    The ``make_trigger`` helper creates a square wave via ``sin``, whose
    FFT fundamental has a −90° phase offset relative to ``cos``.  With
    the raw phase subtraction in :class:`FFTExtraction`, the expected
    relative phase for a cosine signal and a sin-based trigger is
    ``signal_phase − trigger_phase + 90°``.

    Tests that need a clean phase relationship use cosine signals for
    both signal and trigger (via ``make_vibration``).
    """

    def test_protocol_compliance(self):
        """FFTExtraction should satisfy the ExtractionMethod protocol."""
        assert isinstance(FFTExtraction(), ExtractionMethod)

    def test_extract_known_amplitude(self):
        """Extract amplitude from a pure cosine."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        method = FFTExtraction()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)

        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_phase_relative_to_trigger(self):
        """Extracted phase should equal signal_phase - trigger_phase."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        trigger_phase = 20.0
        signal_phase = 70.0

        trigger = make_vibration(f0, 1.0, trigger_phase, sr, dur)
        signal = make_vibration(f0, amp, signal_phase, sr, dur)

        method = FFTExtraction()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        expected_phase = signal_phase - trigger_phase  # 50 deg
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, expected_phase, atol=5.0)

    def test_zero_phase(self):
        """Signal and trigger both at 0 deg should give ~0 phase."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = FFTExtraction()
        _, phase_out = method.extract(signal, trigger, sr, f0)
        assert np.isclose(phase_out, 0.0, atol=5.0)

    def test_negative_phase(self):
        """Negative signal phase should be recovered."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
        signal = make_vibration(f0, 3.0, -60.0, sr, dur)

        method = FFTExtraction()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        assert np.isclose(amp_out, 3.0, rtol=0.05)
        assert np.isclose(phase_out, -60.0, atol=5.0)

    def test_with_noise(self):
        """Extraction should work with moderate noise."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 45.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur, noise_level=0.1)

        method = FFTExtraction()
        amp_out, phase_out = method.extract(signal, trigger, sr, f0)
        assert np.isclose(amp_out, amp, rtol=0.05)
        assert np.isclose(phase_out, phase, atol=5.0)

    def test_different_nfft_mult(self):
        """Different nfft_mult values should all produce correct results."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 2.0
        phase = 30.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
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

    def test_end_to_end_with_vibration_extractor(self):
        """FFTExtraction plugged into VibrationExtractor should work."""
        f0 = 25.0
        sr = 10000.0
        dur = 5.0
        amp = 1.5
        phase = -45.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
        signal = make_vibration(f0, amp, phase, sr, dur)

        extractor = VibrationExtractor(
            sample_rate=sr, rotation_freq=f0, method=FFTExtraction(),
        )
        vv = extractor.get_vibration_vector(signal, trigger)

        assert isinstance(vv, VibrationVector)
        assert np.isclose(vv.amplitude, amp, rtol=0.05)
        expected_phase = phase % 360  # -45 -> 315
        assert np.isclose(vv.phase, expected_phase, atol=5.0)

    def test_stores_diagnostics(self):
        """Diagnostic attributes should be populated after extract."""
        f0 = 25.0
        sr = 10000.0
        dur = 2.0

        trigger = make_vibration(f0, 1.0, 0.0, sr, dur)
        signal = make_vibration(f0, 1.0, 0.0, sr, dur)

        method = FFTExtraction()
        method.extract(signal, trigger, sr, f0)

        assert method.freq_ is not None
        assert method.H_signal_ is not None
        assert method.H_trigger_ is not None
        assert method.peak_idx_ is not None
        assert isinstance(method.peak_idx_, int)
