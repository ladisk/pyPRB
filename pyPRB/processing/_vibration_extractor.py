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
            Phase of the fundamental component in degrees, relative to
            the trigger edge: a response ``A·cos(θ + φ)``, where ``θ`` is
            the rotation angle measured from the trigger edge, gives
            ``φ``.
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


_EDGES = ("rising", "falling")


def _check_edge(edge: str) -> None:
    """Raise ValueError if *edge* is not a valid trigger edge."""
    if edge not in _EDGES:
        raise ValueError(f"edge must be 'rising' or 'falling', got '{edge}'.")


def _find_trigger_edges(
    trigger: np.ndarray,
    edge: str = "rising",
    min_spacing: float | None = None,
) -> np.ndarray:
    """Locate trigger edges with sub-sample resolution.

    The threshold is the midpoint between the minimum and the maximum of
    the trigger, so the result does not depend on the pulse width or on
    the signal levels.  The position of each edge is linearly
    interpolated between the two samples on either side of the
    threshold.  An edge is only accepted when the trigger moves all the
    way from below 25 % to above 75 % of its range (or the reverse for a
    falling edge); this hysteresis rejects extra crossings caused by
    noise on a slow edge.  If noise makes the trigger cross the midpoint
    several times within one edge, the mean of the first and the last
    crossing is used.

    Parameters
    ----------
    trigger : np.ndarray
        Trigger signal (one pulse per revolution).
    edge : str
        ``'rising'`` (default) or ``'falling'``.
    min_spacing : float, optional
        Minimum distance between two edges in samples.  An edge closer
        than this to the previously accepted edge is ignored, which
        suppresses multiple crossings caused by noise on a slow edge.

    Returns
    -------
    np.ndarray
        Fractional sample positions of the edges.

    Raises
    ------
    ValueError
        If the trigger is constant or contains no edge of the given type.
    """
    _check_edge(edge)
    trigger = np.asarray(trigger, dtype=float)
    lo, hi = float(np.min(trigger)), float(np.max(trigger))
    if not hi > lo:
        raise ValueError("Trigger signal is constant; no edges found.")
    threshold = 0.5 * (lo + hi)
    if edge == "falling":
        trigger = -trigger
        threshold = -threshold
        lo, hi = -hi, -lo
    upper = lo + 0.75 * (hi - lo)
    lower = lo + 0.25 * (hi - lo)

    # Schmitt trigger: the state changes only when a band limit is passed.
    is_high = trigger > upper
    decided = is_high | (trigger < lower)
    last = np.maximum.accumulate(np.where(decided, np.arange(len(trigger)), 0))
    state = is_high[last] & decided[last]
    switch_on = np.where(~state[:-1] & state[1:])[0] + 1   # first high sample

    # Midpoint crossings (interpolated) inside each transition, i.e.
    # between the last sample below the lower limit and the switch-on.
    # With noise there can be several; the edge is the mean of the first
    # and the last one.
    above = trigger > threshold
    idx = np.where(~above[:-1] & above[1:])[0]
    x0 = trigger[idx]
    x1 = trigger[idx + 1]
    crossings = idx + (threshold - x0) / (x1 - x0)

    last_low = np.maximum.accumulate(
        np.where(trigger < lower, np.arange(len(trigger)), -1))
    starts = last_low[switch_on]
    valid = starts >= 0
    first = np.searchsorted(idx, starts[valid])
    last = np.searchsorted(idx, switch_on[valid]) - 1
    ok = last >= first
    if not np.any(ok):
        raise ValueError(
            f"No {edge} edges found in the trigger signal. "
            "Check that the trigger channel contains a periodic pulse."
        )
    edges = 0.5 * (crossings[first[ok]] + crossings[last[ok]])

    if min_spacing is not None and len(edges) > 1:
        keep = [edges[0]]
        for e in edges[1:]:
            if e - keep[-1] >= min_spacing:
                keep.append(e)
        edges = np.array(keep)
    return edges


def _edge_reference(
    trigger: np.ndarray,
    sample_rate: float,
    rotation_freq: float,
    edge: str = "rising",
) -> np.ndarray:
    """Unit cosine at the rotation frequency with zero phase at the trigger edges.

    The phase reference of all extraction methods is the trigger edge
    (the datum mark passing the probe).  The reference phase is the
    circular mean over all edges, which makes it robust to edge jitter.
    Processing this reference in the same way as the response signal
    and subtracting its phase gives the phase relative to the edge.

    Parameters
    ----------
    trigger : np.ndarray
        Trigger signal.
    sample_rate : float
        Sampling rate in Hz.
    rotation_freq : float
        Rotation frequency in Hz.
    edge : str
        ``'rising'`` (default) or ``'falling'``.

    Returns
    -------
    np.ndarray
        ``cos(ω (t - t_edge))`` with the same length as *trigger*.
    """
    samples_per_period = sample_rate / rotation_freq
    edges = _find_trigger_edges(trigger, edge,
                                min_spacing=0.5 * samples_per_period)
    omega = 2 * np.pi / samples_per_period          # rad per sample
    theta_0 = np.angle(np.mean(np.exp(1j * omega * edges)))
    n = np.arange(len(trigger))
    return np.cos(omega * n - theta_0)


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
