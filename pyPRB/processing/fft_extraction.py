"""FFT-based vibration extraction method.

Extracts the vibration component at the rotation frequency by computing
the FFT of both the response and trigger signals, then deriving
relative phase by subtracting the trigger phase from the signal phase.
"""

from __future__ import annotations

import numpy as np

from ._vibration_extractor import _rfft, _import_matplotlib, _WINDOWS


class FFTExtraction:
    """Extracts the vibration component at the rotation frequency via FFT.

    Computes the one-sided FFT of both the response signal and the
    trigger signal, locates the spectral peak nearest to the expected
    rotation frequency, and extracts amplitude and phase.  The relative
    phase is obtained by subtracting the trigger phase from the signal
    phase at the detected peak.

    A window function (controlled by *window*) is applied before the
    FFT to reduce spectral leakage.  Zero-padding (controlled by
    *nfft_mult*) improves frequency resolution.

    Parameters
    ----------
    nfft_mult : int
        Zero-padding multiplier for the FFT.  Higher values give finer
        frequency resolution.  Default is 4.
    freq_tolerance : float
        Relative tolerance for finding the rotation frequency peak.
        The search band is
        ``[rotation_freq * (1 - freq_tolerance),
        rotation_freq * (1 + freq_tolerance)]``.
        Default is 0.1 (±10 %).
    window : str
        Window function to apply before the FFT.  Supported values are
        ``'hann'`` (default), ``'hamming'``, ``'bartlett'``, and
        ``'blackman'``.
    plotting : bool
        If ``True``, show a diagnostic spectrum plot during each call
        to :meth:`extract`.  Requires matplotlib
        (install with ``pip install pyPRB[plot]``).

    Attributes
    ----------
    freq_ : np.ndarray or None
        Frequency axis from the last call to :meth:`extract`.
    H_signal_ : np.ndarray or None
        Complex signal spectrum from the last call to :meth:`extract`.
    H_trigger_ : np.ndarray or None
        Complex trigger spectrum from the last call to :meth:`extract`.
    peak_idx_ : int or None
        Index of the detected peak in the spectrum arrays.

    Examples
    --------
    >>> from pyPRB.processing import FFTExtraction, VibrationExtractor
    >>> method = FFTExtraction(nfft_mult=8, freq_tolerance=0.05)
    >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=25.0,
    ...                                method=method)
    """

    def __init__(
        self,
        nfft_mult: int = 4,
        freq_tolerance: float = 0.1,
        window: str = "hann",
        plotting: bool = False,
    ):
        if not isinstance(nfft_mult, int) or nfft_mult < 1:
            raise ValueError(
                f"nfft_mult must be a positive integer, got {nfft_mult!r}."
            )
        if not 0 < freq_tolerance < 1:
            raise ValueError(
                f"freq_tolerance must be between 0 and 1, "
                f"got {freq_tolerance}."
            )
        if window not in _WINDOWS:
            raise ValueError(
                f"Unknown window {window!r}. "
                f"Supported windows: {sorted(_WINDOWS)}."
            )
        self.nfft_mult = nfft_mult
        self.freq_tolerance = freq_tolerance
        self.window = window
        self.plotting = plotting
        self.freq_: np.ndarray | None = None
        self.H_signal_: np.ndarray | None = None
        self.H_trigger_: np.ndarray | None = None
        self.peak_idx_: int | None = None

    def extract(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
        sample_rate: float,
        rotation_freq: float,
    ) -> tuple[float, float]:
        """Extract vibration amplitude and phase at the rotation frequency.

        Parameters
        ----------
        signal : np.ndarray
            Response signal (acceleration, velocity, …).
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
            Peak amplitude at the rotation frequency.
        phase_deg : float
            Phase at the rotation frequency in degrees, relative to
            the trigger signal.
        """
        freq, H_sig = _rfft(
            signal, sample_rate,
            nfft_mult=self.nfft_mult, remove_dc=True, window=self.window,
        )
        _, H_trig = _rfft(
            trigger, sample_rate,
            nfft_mult=self.nfft_mult, remove_dc=True, window=self.window,
        )

        # Find peak within tolerance band
        f_lo = rotation_freq * (1 - self.freq_tolerance)
        f_hi = rotation_freq * (1 + self.freq_tolerance)
        mask = (freq >= f_lo) & (freq <= f_hi)

        if not np.any(mask):
            raise ValueError(
                f"No frequency bins in the search band "
                f"[{f_lo:.2f}, {f_hi:.2f}] Hz. "
                f"Check rotation_freq ({rotation_freq:.2f} Hz) and "
                f"freq_tolerance ({self.freq_tolerance})."
            )

        global_indices = np.where(mask)[0]
        idx_in_mask = np.argmax(np.abs(H_sig[mask]))
        peak_idx = global_indices[idx_in_mask]

        amplitude = float(np.abs(H_sig[peak_idx]))
        sig_phase = np.angle(H_sig[peak_idx])
        trig_phase = np.angle(H_trig[peak_idx])
        phase_deg = float(np.degrees(sig_phase - trig_phase))

        # Store diagnostics
        self.freq_ = freq
        self.H_signal_ = H_sig
        self.H_trigger_ = H_trig
        self.peak_idx_ = int(peak_idx)

        if self.plotting:
            self._plot_spectrum(
                freq, H_sig, H_trig, peak_idx,
                rotation_freq, f_lo, f_hi, amplitude, phase_deg,
            )

        return amplitude, phase_deg

    @staticmethod
    def _plot_spectrum(
        freq: np.ndarray,
        H_sig: np.ndarray,
        H_trig: np.ndarray,
        peak_idx: int,
        rotation_freq: float,
        f_lo: float,
        f_hi: float,
        amplitude: float,
        phase_deg: float,
    ) -> None:
        """Plot signal and trigger spectra with the detected peak."""
        plt = _import_matplotlib()

        sig_phase_deg = float(np.degrees(np.angle(H_sig[peak_idx])))
        trig_phase_deg = float(np.degrees(np.angle(H_trig[peak_idx])))
        peak_freq = freq[peak_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # -- Signal spectrum --
        ax1.plot(freq, np.abs(H_sig), color="steelblue")
        ax1.axvspan(f_lo, f_hi, alpha=0.15, color="orange",
                    label="Search band")
        ax1.plot(peak_freq, np.abs(H_sig[peak_idx]), "ro",
                 label=f"Peak @ {peak_freq:.2f} Hz")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(
            f"Signal spectrum \u2014 "
            f"amplitude = {amplitude:.4g}, "
            f"phase = {sig_phase_deg:.1f}\u00b0"
        )
        ax1.legend()

        # -- Trigger spectrum --
        ax2.plot(freq, np.abs(H_trig), color="steelblue")
        ax2.axvspan(f_lo, f_hi, alpha=0.15, color="orange",
                    label="Search band")
        ax2.plot(peak_freq, np.abs(H_trig[peak_idx]), "ro",
                 label=f"Peak @ {peak_freq:.2f} Hz")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.set_title(
            f"Trigger spectrum \u2014 "
            f"phase = {trig_phase_deg:.1f}\u00b0"
        )
        ax2.legend()

        # Limit x-axis to 2x rotation_freq above and below the peak
        ax2.set_xlim(max(0, rotation_freq - 2 * rotation_freq),
                     rotation_freq + 2 * rotation_freq)  # [0, 3·f0]

        fig.suptitle(
            f"FFT extraction \u2014 "
            f"relative phase = {phase_deg:.1f}\u00b0",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show(block=True)
