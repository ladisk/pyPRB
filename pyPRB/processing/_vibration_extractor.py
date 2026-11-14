"""VibrationExtractor class and rotation frequency detection."""

from __future__ import annotations

import numpy as np

from typing import NamedTuple, Protocol, runtime_checkable

from ..core import AGAINST_ROTATION, WITH_ROTATION, VibrationVector
from ..core import _check_convention


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


def _rotation_freq_from_edges(
    trigger: np.ndarray,
    sample_rate: float,
    edge: str = "rising",
) -> float:
    """Rotation frequency from the spacing of the trigger edges.

    The edge positions are fitted with a straight line against the
    revolution number (least squares), so a missing pulse does not bias
    the result.
    """
    edges = _find_trigger_edges(trigger, edge)
    if len(edges) < 2:
        raise ValueError(
            f"At least two {edge} edges are needed to detect the rotation "
            f"frequency, found {len(edges)}. Provide a longer signal."
        )
    period_0 = float(np.median(np.diff(edges)))
    revolution = np.round((edges - edges[0]) / period_0)
    # Keep the first edge of each revolution (extra edges are not expected
    # after the hysteresis, but are harmless here).
    revolution, first = np.unique(revolution, return_index=True)
    edges = edges[first]
    if len(edges) < 2:
        raise ValueError("Could not determine the trigger period.")
    period = np.polyfit(revolution, edges, 1)[0]
    return float(sample_rate / period)


def _rotation_freq_from_fft(
    trigger: np.ndarray,
    sample_rate: float,
    max_freq: float | None,
    nfft_mult: int,
    subharmonic_ratio: float = 0.5,
    max_subharmonic: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Rotation frequency from the spectrum of the trigger.

    Finds the highest peak and then returns the lowest sub-harmonic
    ``f_peak / k`` (``k <= max_subharmonic``) whose magnitude is at least
    *subharmonic_ratio* of the peak.  For a short pulse the harmonics are
    nearly as high as the fundamental, so the highest peak alone is
    often a harmonic.

    Returns the frequency, the frequency axis and the magnitudes of the
    search range, and the index of the selected line in that range.
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

    freq_m = freq[mask]
    magnitudes = np.abs(H[mask])
    peak_idx = int(np.argmax(magnitudes))
    peak = magnitudes[peak_idx]
    df = freq_m[1] - freq_m[0] if len(freq_m) > 1 else freq_m[0]

    selected = peak_idx
    for k in range(max_subharmonic, 1, -1):
        target = freq_m[peak_idx] / k
        if target < freq_m[0] - df:
            continue
        centre = int(np.argmin(np.abs(freq_m - target)))
        lo, hi = max(centre - 1, 0), min(centre + 2, len(freq_m))
        local = lo + int(np.argmax(magnitudes[lo:hi]))
        if magnitudes[local] >= subharmonic_ratio * peak:
            selected = local
            break
    return float(freq_m[selected]), freq_m, magnitudes, selected


def detect_rotation_freq(
    trigger: np.ndarray,
    sample_rate: float,
    max_freq: float | None = None,
    nfft_mult: int = 1,
    plot: bool = False,
    method: str = "edges",
    edge: str = "rising",
) -> float:
    """Detect the rotation frequency from a trigger signal.

    Two methods are available:

    ``'edges'`` (default)
        The trigger edges are located with sub-sample resolution and
        hysteresis (as for the phase reference of the extraction
        methods), and the period is the slope of a straight-line fit of
        the edge positions against the revolution number.  The result
        does not depend on the pulse width or on the FFT resolution.
    ``'fft'``
        The highest peak of the (optionally zero-padded) spectrum of
        the trigger, replaced by its lowest sub-harmonic ``f/k``
        (``k <= 5``) that reaches at least half of the peak.  For a
        short pulse the harmonics are nearly as high as the
        fundamental, so the highest peak alone is often a harmonic.

    Parameters
    ----------
    trigger : np.ndarray
        Trigger signal from a photoelectric probe or similar sensor
        (one pulse per revolution).
    sample_rate : float
        Sampling rate in Hz.
    max_freq : float, optional
        Upper frequency limit in Hz.  With ``method='fft'`` the
        spectrum is searched up to this frequency; with
        ``method='edges'`` a result above it raises ``ValueError``.
        If ``None`` (default), there is no limit.
    nfft_mult : int
        Zero-padding multiplier for the FFT (``method='fft'`` and the
        plot).  Default is 1 (no zero-padding).
    plot : bool
        If ``True``, show a diagnostic figure with the frequency
        spectrum and a time-domain overlay of the detected sinusoid.
        Requires matplotlib.
    method : str
        ``'edges'`` (default) or ``'fft'``.
    edge : str
        Trigger edge used by ``method='edges'``: ``'rising'``
        (default) or ``'falling'``.

    Returns
    -------
    float
        Detected rotation frequency in Hz.

    Raises
    ------
    ValueError
        If no valid edge or peak is found, or the result exceeds
        *max_freq*.

    Examples
    --------
    >>> f0 = detect_rotation_freq(trigger_signal, sample_rate=10000)
    >>> f0 = detect_rotation_freq(trigger_signal, sample_rate=10000,
    ...                           method='fft', max_freq=50.0, nfft_mult=256)
    """
    if method not in ("edges", "fft"):
        raise ValueError(f"method must be 'edges' or 'fft', got '{method}'.")
    _check_edge(edge)

    if method == "edges":
        f0 = _rotation_freq_from_edges(trigger, sample_rate, edge)
        if max_freq is not None and f0 > max_freq:
            raise ValueError(
                f"Detected rotation frequency {f0:.4g} Hz exceeds "
                f"max_freq={max_freq:g} Hz. Check the trigger signal."
            )
    else:
        f0, _, _, _ = _rotation_freq_from_fft(trigger, sample_rate,
                                              max_freq, nfft_mult)

    if plot:
        freq, H = _rfft(trigger, sample_rate, nfft_mult=nfft_mult,
                        remove_dc=True)
        mask = freq > 0
        if max_freq is not None:
            mask &= freq <= max(max_freq, f0)
        freq_m, H_m = freq[mask], H[mask]
        line = int(np.argmin(np.abs(freq_m - f0)))
        _plot_rotation_freq(trigger, sample_rate, freq_m, np.abs(H_m),
                            f0, H_m[line])

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


class SpeedVariation(NamedTuple):
    """Result of :func:`rotation_freq_variation`."""

    #: Mean rotation frequency over the record in Hz.
    mean: float
    #: Lowest value in Hz.
    minimum: float
    #: Highest value in Hz.
    maximum: float
    #: Standard deviation of the values in Hz.
    std: float
    #: ``(maximum - minimum) / mean``.
    relative_range: float
    #: Rotation frequency in Hz of each block of revolutions, in time order.
    values: np.ndarray
    #: Number of whole revolutions found in the record.
    n_revolutions: int


def rotation_freq_variation(
    trigger: np.ndarray,
    sample_rate: float,
    revolutions_per_value: int = 10,
    edge: str = "rising",
) -> SpeedVariation:
    """Measure the rotation frequency and its variation over a record.

    The times of the once-per-revolution trigger edges are located (with
    hysteresis and linear interpolation between samples, as for the
    phase reference of the extraction methods) and the rotation frequency is
    computed for consecutive blocks of revolutions.  A balancing reading
    is only meaningful if the speed is steady during the record and the
    same in all runs.

    Parameters
    ----------
    trigger : np.ndarray
        Trigger signal with one pulse per revolution.
    sample_rate : float
        Sampling rate in Hz.
    revolutions_per_value : int
        Number of revolutions over which each frequency value is taken.
        Default is 10.  One revolution gives the finest time resolution,
        but the timing error of one sample then shows up as apparent
        speed variation of about ``rotation_freq / sample_rate``.
    edge : str
        Trigger edge to use, ``'rising'`` (default) or ``'falling'``.

    Returns
    -------
    SpeedVariation
        Named tuple ``(mean, minimum, maximum, std, relative_range,
        values, n_revolutions)``, frequencies in Hz.

    Raises
    ------
    ValueError
        If the record holds fewer than ``revolutions_per_value``
        revolutions.

    Examples
    --------
    >>> import numpy as np
    >>> from pyPRB import rotation_freq_variation
    >>> t = np.arange(0, 4, 1 / 10000)
    >>> trigger = np.sign(np.sin(2 * np.pi * 25.0 * t + 0.3))
    >>> speed = rotation_freq_variation(trigger, sample_rate=10000)
    >>> round(speed.mean, 2), speed.n_revolutions
    (25.0, 99)
    >>> bool(speed.relative_range < 0.001)
    True
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    _check_edge(edge)
    if (not isinstance(revolutions_per_value, (int, np.integer))
            or revolutions_per_value < 1):
        raise ValueError(
            "revolutions_per_value must be a positive integer, got "
            f"{revolutions_per_value!r}.")

    try:
        edges = _find_trigger_edges(trigger, edge)
    except ValueError:
        edges = np.empty(0)
    if len(edges) < revolutions_per_value + 1:
        raise ValueError(
            f"Found {max(len(edges) - 1, 0)} whole revolutions in the trigger "
            f"signal; at least {revolutions_per_value} are needed.")
    times = edges / sample_rate

    block_times = times[::revolutions_per_value]
    values = revolutions_per_value / np.diff(block_times)
    mean = (len(times) - 1) / (times[-1] - times[0])
    return SpeedVariation(
        mean=float(mean),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        std=float(np.std(values)),
        relative_range=float((np.max(values) - np.min(values)) / mean),
        values=values,
        n_revolutions=int(len(times) - 1),
    )


class VectorScatter(NamedTuple):
    """Result of :meth:`VibrationExtractor.get_vector_scatter`."""

    #: 1x vector of each consecutive part of the record (the cloud).
    vectors: list
    #: Mean vector (the complex mean of the cloud).
    mean: VibrationVector
    #: Largest minus smallest amplitude in the cloud.
    amplitude_spread: float
    #: Range of the phases about the mean vector in degrees
    #: (``nan`` if the mean vector is zero).
    phase_spread: float
    #: Largest distance of a vector from the mean vector.
    max_deviation: float
    #: ``max_deviation`` divided by the amplitude of the mean vector.
    relative_deviation: float


class HarmonicContent(NamedTuple):
    """Result of :meth:`VibrationExtractor.get_harmonic_content`."""

    #: Orders (multiples of the rotation frequency).
    orders: tuple
    #: Peak amplitude at each order.
    amplitudes: tuple
    #: Overall level: r.m.s. of the signal without its mean.
    overall_rms: float
    #: R.m.s. of the 1x component divided by the overall level
    #: (``nan`` if order 1 is not among the orders).
    fraction_1x: float


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
    angle_convention : {"with_rotation", "against_rotation"}, optional
        With "with_rotation" (default) the phase of the returned vectors
        is the angle by which the vibration leads the trigger, which goes
        with weight positions counted in the direction of rotation.  With
        "against_rotation" it is the phase lag, which goes with weight
        positions counted against the direction of rotation, as many
        balancing instruments do.

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
        angle_convention: str = WITH_ROTATION,
    ):
        self.angle_convention = _check_convention(angle_convention)
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
        if self.angle_convention == AGAINST_ROTATION:
            phase_deg = -phase_deg
        return VibrationVector(amplitude=amplitude, phase=phase_deg)

    def get_vibration_vectors(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
        n_parts: int = 10,
    ) -> list:
        """Extract the 1x vector of consecutive parts of one record.

        The record is cut into ``n_parts`` parts of equal length and the
        vector of each part is extracted like in
        :meth:`get_vibration_vector`.  For a steady machine all vectors
        coincide; their scatter shows how repeatable a reading is.

        Parameters
        ----------
        signal : np.ndarray
            Response signal.
        trigger : np.ndarray
            Trigger signal of the same length.
        n_parts : int
            Number of parts, at least 2.  Default is 10.  Each part must
            hold at least two revolutions.

        Returns
        -------
        list of VibrationVector
            One vector per part, in time order.
        """
        signal = np.asarray(signal)
        trigger = np.asarray(trigger)
        if not isinstance(n_parts, (int, np.integer)) or n_parts < 2:
            raise ValueError(
                f"n_parts must be an integer of at least 2, got {n_parts!r}.")
        if len(signal) != len(trigger):
            raise ValueError(
                f"signal and trigger must have the same length, got "
                f"{len(signal)} and {len(trigger)}.")
        part = len(signal) // n_parts
        if part < 2 * self.sample_rate / self.rotation_freq:
            raise ValueError(
                f"Each of the {n_parts} parts must hold at least two "
                f"revolutions; the record is too short ({part} samples per "
                "part). Use fewer parts or a longer record.")
        return [
            self.get_vibration_vector(signal[k * part:(k + 1) * part],
                                      trigger[k * part:(k + 1) * part])
            for k in range(n_parts)
        ]

    def get_vector_scatter(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
        n_parts: int = 10,
    ) -> VectorScatter:
        """Extract the 1x vector with its scatter over one record.

        Parameters
        ----------
        signal : np.ndarray
            Response signal.
        trigger : np.ndarray
            Trigger signal of the same length.
        n_parts : int
            Number of consecutive parts, at least 2.  Default is 10.

        Returns
        -------
        VectorScatter
            Named tuple ``(vectors, mean, amplitude_spread, phase_spread,
            max_deviation, relative_deviation)``.  ``vectors`` is the
            cloud of the parts, ``mean`` their mean vector and
            ``max_deviation`` the radius of the circle about the mean
            vector that holds all of them.

        Notes
        -----
        The scatter over the parts of one record shows noise and
        unsteadiness during the record.  It does not show what changes
        between runs (speed, temperature, a mass that moved).  The
        rotation frequency is taken as constant over the record; see
        :func:`rotation_freq_variation`.

        Examples
        --------
        >>> import numpy as np
        >>> from pyPRB import VibrationExtractor
        >>> t = np.arange(0, 4, 1 / 10000)
        >>> trigger = np.sin(2 * np.pi * 25.0 * t)   # rising edge at t = 0
        >>> signal = 3.4 * np.cos(2 * np.pi * 25.0 * t + np.deg2rad(116))
        >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=25.0)
        >>> scatter = extractor.get_vector_scatter(signal, trigger, n_parts=8)
        >>> len(scatter.vectors)
        8
        >>> print(scatter.mean)
        VibrationVector(amplitude=3.4, phase=116°)
        >>> bool(scatter.relative_deviation < 1e-6)
        True
        """
        vectors = self.get_vibration_vectors(signal, trigger, n_parts)
        z = np.array([v.complex_num for v in vectors])
        mean = complex(np.mean(z))
        amplitudes = np.abs(z)
        max_deviation = float(np.max(np.abs(z - mean)))
        if abs(mean) > 0:
            deviation = np.rad2deg(np.angle(z / mean))
            phase_spread = float(np.max(deviation) - np.min(deviation))
            relative_deviation = max_deviation / abs(mean)
        else:
            phase_spread = float("nan")
            relative_deviation = float("inf") if max_deviation > 0 else 0.0
        return VectorScatter(
            vectors=vectors,
            mean=VibrationVector.from_complex(mean),
            amplitude_spread=float(np.max(amplitudes) - np.min(amplitudes)),
            phase_spread=phase_spread,
            max_deviation=max_deviation,
            relative_deviation=float(relative_deviation),
        )

    def get_harmonic_content(
        self,
        signal: np.ndarray,
        orders: tuple = (1, 2, 3),
    ) -> HarmonicContent:
        """Measure the content at multiples of the rotation frequency.

        Unbalance shows at 1x.  If the 1x component is only a small part
        of the overall level, or 2x and 3x are large, the vibration has
        other causes as well and balancing alone will not remove it.

        The amplitudes are obtained by correlation with a sine and a
        cosine at each order over a whole number of revolutions,
        independently of the extraction method of the extractor.

        Parameters
        ----------
        signal : np.ndarray
            Response signal.
        orders : sequence of int
            Orders to evaluate.  Default is ``(1, 2, 3)``.

        Returns
        -------
        HarmonicContent
            Named tuple ``(orders, amplitudes, overall_rms, fraction_1x)``:
            the peak amplitude at each order, the overall r.m.s. level of
            the signal (mean removed) and the r.m.s. of the 1x component
            as a fraction of it.

        Examples
        --------
        >>> import numpy as np
        >>> from pyPRB import VibrationExtractor
        >>> t = np.arange(0, 4, 1 / 10000)
        >>> signal = (3.0 * np.cos(2 * np.pi * 25.0 * t + 1.0)
        ...           + 1.0 * np.cos(2 * np.pi * 50.0 * t))
        >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=25.0)
        >>> content = extractor.get_harmonic_content(signal)
        >>> [round(a, 3) for a in content.amplitudes]
        [3.0, 1.0, 0.0]
        >>> round(content.fraction_1x, 3)
        0.949
        """
        orders = tuple(orders)
        if len(orders) == 0 or any(
                not float(k).is_integer() or k < 1 for k in orders):
            raise ValueError(
                f"orders must be positive integers, got {orders!r}.")
        orders = tuple(int(k) for k in orders)
        if max(orders) * self.rotation_freq >= self.sample_rate / 2:
            raise ValueError(
                f"Order {max(orders)} is above half the sampling rate.")
        signal = np.asarray(signal, dtype=float)
        period = self.sample_rate / self.rotation_freq
        n_periods = int(len(signal) / period)
        if n_periods < 1:
            raise ValueError(
                "The signal is shorter than one revolution.")
        x = signal[:int(round(n_periods * period))]
        x = x - np.mean(x)
        t = np.arange(len(x)) / self.sample_rate
        amplitudes = tuple(
            float(2 * np.abs(np.mean(
                x * np.exp(-2j * np.pi * k * self.rotation_freq * t))))
            for k in orders)
        overall_rms = float(np.sqrt(np.mean(x ** 2)))
        if 1 in orders and overall_rms > 0:
            fraction_1x = amplitudes[orders.index(1)] / np.sqrt(2) / overall_rms
        else:
            fraction_1x = float("nan")
        return HarmonicContent(orders=orders, amplitudes=amplitudes,
                               overall_rms=overall_rms,
                               fraction_1x=float(fraction_1x))
