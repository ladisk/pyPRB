"""VibrationExtractor class and rotation frequency detection."""

from __future__ import annotations

import numpy as np

from typing import Protocol, runtime_checkable

from ..core import VibrationVector


@runtime_checkable
class ExtractionMethod(Protocol):
    """Protocol for vibration extraction methods.

    Any object with an ``extract`` method matching this signature can be
    used as the *method* argument to :class:`VibrationExtractor`.
    """

    def extract(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
        sample_rate: float,
        rotation_freq: float,
    ) -> tuple[float, float]:
        """Extract the fundamental (1x) vibration component.

        Parameters
        ----------
        signal : np.ndarray
            Response signal (acceleration, velocity, displacement, …).
        trigger : np.ndarray
            Trigger signal from a photoelectric probe or similar sensor
            that produces one pulse per revolution.
        sample_rate : float
            Sampling rate in Hz.
        rotation_freq : float
            Rotation frequency in Hz.

        Returns
        -------
        amplitude : float
            Peak amplitude of the fundamental (1×) component.
        phase_deg : float
            Phase of the fundamental component in degrees.
        """
        ...


_WINDOWS: dict[str, np.ufunc] = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "bartlett": np.bartlett,
    "blackman": np.blackman,
}


def _rfft(
    signal: np.ndarray,
    sample_rate: float,
    nfft_mult: int = 1,
    remove_dc: bool = False,
    window: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a scaled one-sided FFT of *signal*.

    The result is scaled so that ``|H[k]|`` equals the peak amplitude
    of the corresponding sinusoidal component.

    Parameters
    ----------
    signal : np.ndarray
        Input time series.
    sample_rate : float
        Sampling rate in Hz.
    nfft_mult : int
        Zero-padding multiplier.  The FFT length is
        ``len(signal) * nfft_mult``.
    remove_dc : bool
        If ``True``, subtract the mean of the signal before computing
        the FFT to suppress the DC component.
    window : str or None
        Window function to apply before the FFT.  Supported values are
        ``'hann'``, ``'hamming'``, ``'bartlett'``, and ``'blackman'``.
        ``None`` (default) applies no window.  The amplitude scaling is
        compensated for the coherent gain of the window so that
        ``|H[k]|`` still returns the correct peak amplitude.

    Returns
    -------
    freq : np.ndarray
        Frequency axis in Hz.
    H : np.ndarray
        Complex spectrum, scaled to peak amplitude.
    """
    if not isinstance(nfft_mult, int) or nfft_mult < 1:
        raise ValueError(
            f"nfft_mult must be a positive integer, got {nfft_mult!r}."
        )
    if remove_dc:
        signal = signal - np.mean(signal)

    M = len(signal)
    coherent_gain = 1.0
    if window is not None:
        if window not in _WINDOWS:
            raise ValueError(
                f"Unknown window {window!r}. "
                f"Supported windows: {sorted(_WINDOWS)}."
            )
        w = _WINDOWS[window](M)
        signal = signal * w
        coherent_gain = float(np.mean(w))

    N = M * nfft_mult
    H = np.fft.rfft(signal, n=N) * 2 / N * nfft_mult / coherent_gain
    freq = np.fft.rfftfreq(N, d=1 / sample_rate)
    return freq, H


def _import_matplotlib():
    """Import matplotlib or raise a helpful error."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install pyPRB[plot]"
        ) from None


def detect_rotation_freq(
    trigger: np.ndarray,
    sample_rate: float,
    max_freq: float | None = None,
    nfft_mult: int = 1,
    plot: bool = False,
) -> float:
    """Detect the rotation frequency from a trigger signal.

    Computes a zero-padded FFT of *trigger* and returns the frequency
    of the highest peak.

    Parameters
    ----------
    trigger : np.ndarray
        Trigger signal from a photoelectric probe or similar sensor.
    sample_rate : float
        Sampling rate in Hz.
    max_freq : float, optional
        Upper frequency limit for the search in Hz.  If ``None``
        (default), the entire spectrum is searched.
    nfft_mult : int
        Zero-padding multiplier for the FFT.  Higher values give finer
        frequency resolution.  Default is 1 (no zero-padding).
    plot : bool
        If ``True``, show a diagnostic figure with the frequency
        spectrum and a time-domain overlay of the detected sinusoid.
        Requires matplotlib.

    Returns
    -------
    float
        Detected rotation frequency in Hz.

    Raises
    ------
    ValueError
        If no valid peak is found.

    Examples
    --------
    >>> f0 = detect_rotation_freq(trigger_signal, sample_rate=10000)
    >>> f0 = detect_rotation_freq(trigger_signal, sample_rate=10000,
    ...                           max_freq=50.0, nfft_mult=256)
    """
    freq, H = _rfft(trigger, sample_rate, nfft_mult=nfft_mult, remove_dc=True)

    mask = freq > 0
    if max_freq is not None:
        mask = mask & (freq <= max_freq)

    if not np.any(mask):
        raise ValueError(
            "No frequency bins found in the search range. "
            "Check sample_rate and max_freq."
        )

    magnitudes = np.abs(H[mask])
    peak_idx = np.argmax(magnitudes)
    f0 = float(freq[mask][peak_idx])

    if plot:
        peak_bin = H[mask][peak_idx]
        _plot_rotation_freq(trigger, sample_rate, freq[mask], magnitudes,
                            f0, peak_bin)

    return f0


def _plot_rotation_freq(
    trigger: np.ndarray,
    sample_rate: float,
    freq: np.ndarray,
    magnitudes: np.ndarray,
    f0: float,
    peak_bin: complex,
    n_periods: int = 5,
) -> None:
    """Plot frequency spectrum and time-domain overlay for rotation detection.

    Parameters
    ----------
    trigger : np.ndarray
        Raw trigger signal.
    sample_rate : float
        Sampling rate in Hz.
    freq : np.ndarray
        Frequency axis (masked region).
    magnitudes : np.ndarray
        FFT magnitudes (masked region).
    f0 : float
        Detected rotation frequency in Hz.
    peak_bin : complex
        Complex FFT value at the detected peak.
    n_periods : int
        Number of periods to show in the time-domain plot.
    """
    plt = _import_matplotlib()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # -- Frequency domain --
    ax1.plot(freq, magnitudes, color="steelblue")
    ax1.axvline(f0, color="red", linestyle="--",
                label=f"Detected: {f0:.2f} Hz")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Trigger signal spectrum")
    ax1.legend()

    # -- Time domain --
    t_end = n_periods / f0
    N = int(sample_rate * t_end)
    N = min(N, len(trigger))
    time = np.arange(N) / sample_rate

    amplitude = float(np.abs(peak_bin))
    phase_rad = float(np.angle(peak_bin))
    reconstructed = amplitude * np.cos(2 * np.pi * f0 * time + phase_rad)

    ax2.plot(time, trigger[:N], color="steelblue", label="Trigger")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time, reconstructed, color="red", linestyle="--",
                  label=f"Reconstructed {f0:.2f} Hz")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Trigger amplitude")
    ax2_twin.set_ylabel("Reconstructed amplitude")
    ax2.set_title(f"Time domain — first {n_periods} periods")
    ax2_twin.legend(loc="upper right")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show(block=True)


class VibrationExtractor:
    """Extracts vibration vectors from time-domain signals.

    Orchestrates the signal processing pipeline that converts raw numpy
    arrays (response signal + trigger signal) into
    :class:`~pyPRB.core.VibrationVector` objects suitable for use with
    :class:`~pyPRB.core.StaticBalancing` or
    :class:`~pyPRB.core.DynamicBalancing`.

    Parameters
    ----------
    sample_rate : float
        Sampling rate of the signals in Hz.
    rotation_freq : float
        Rotation frequency in Hz.  Use :func:`detect_rotation_freq` if
        unknown.
    method : ExtractionMethod, optional
        Extraction method to use.  Defaults to
        :class:`IQDemodulation` with default settings.

    Examples
    --------
    >>> f0 = detect_rotation_freq(trigger_signal, sample_rate=10000)
    >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=f0)
    >>> V_0 = extractor.get_vibration_vector(vibration_signal, trigger_signal)
    """

    def __init__(
        self,
        sample_rate: float,
        rotation_freq: float,
        method: ExtractionMethod | None = None,
    ):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
        if rotation_freq <= 0:
            raise ValueError(
                f"rotation_freq must be positive, got {rotation_freq}."
            )

        self.sample_rate = float(sample_rate)
        self.rotation_freq = float(rotation_freq)
        if method is None:
            from .iq_demodulation import IQDemodulation
            method = IQDemodulation()
        self.method = method

    def get_vibration_vector(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
    ) -> VibrationVector:
        """Extract the fundamental vibration vector from a measurement.

        Parameters
        ----------
        signal : np.ndarray
            Response signal (acceleration, velocity, displacement, …).
        trigger : np.ndarray
            Trigger signal from e.g. a photoelectric probe.

        Returns
        -------
        VibrationVector
            Vibration vector with amplitude and phase of the fundamental
            (once-per-revolution) component.
        """
        amplitude, phase_deg = self.method.extract(
            signal, trigger, self.sample_rate, self.rotation_freq
        )
        return VibrationVector(amplitude=amplitude, phase=phase_deg)
