"""Least-squares sine fit extraction method.

Extracts the vibration component at the rotation frequency by fitting
a cosine–sine model to the signal via linear least squares.
Statistically optimal under Gaussian noise and handles non-integer
periods without spectral leakage.
"""

from __future__ import annotations

import numpy as np

from ._vibration_extractor import _import_matplotlib


class LeastSquaresFit:
    """Extracts the vibration component at the rotation frequency via least-squares fit.

    Fits the model ``A·cos(ωt) + B·sin(ωt) [+ C]`` to the signal using
    :func:`numpy.linalg.lstsq`.  The amplitude and phase are derived from
    the fitted coefficients.  Unlike FFT- or IQ-based methods, the
    least-squares approach handles non-integer periods correctly —
    there is no spectral leakage even when the measurement window does
    not span an exact number of revolutions.

    The relative phase is obtained by performing the same fit on the
    trigger signal and subtracting the trigger phase from the signal
    phase.

    Parameters
    ----------
    fit_dc : bool
        If ``True`` (default), include a constant (DC) term in the
        model: ``A·cos(ωt) + B·sin(ωt) + C``.  This prevents a DC
        offset in the signal from biasing the amplitude estimate.
    plotting : bool
        If ``True``, show a diagnostic figure during each call to
        :meth:`extract`.  Requires matplotlib
        (install with ``pip install pyPRB[plot]``).

    Attributes
    ----------
    A_signal_ : float or None
        Cosine coefficient of the signal fit from the last call to
        :meth:`extract`.
    B_signal_ : float or None
        Sine coefficient of the signal fit from the last call to
        :meth:`extract`.
    A_trigger_ : float or None
        Cosine coefficient of the trigger fit from the last call to
        :meth:`extract`.
    B_trigger_ : float or None
        Sine coefficient of the trigger fit from the last call to
        :meth:`extract`.
    dc_signal_ : float or None
        Fitted DC offset of the signal (``None`` if *fit_dc* is
        ``False``).
    residuals_ : np.ndarray or None
        Residual vector ``signal − X @ coeffs`` from the last call to
        :meth:`extract`.

    Examples
    --------
    >>> from pyPRB.processing import LeastSquaresFit, VibrationExtractor
    >>> method = LeastSquaresFit(fit_dc=True)
    >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=25.0,
    ...                                method=method)
    """

    def __init__(
        self,
        fit_dc: bool = True,
        plotting: bool = False,
    ):
        self.fit_dc = fit_dc
        self.plotting = plotting

        # Diagnostic attributes (None until extract() is called)
        self.A_signal_: float | None = None
        self.B_signal_: float | None = None
        self.A_trigger_: float | None = None
        self.B_trigger_: float | None = None
        self.dc_signal_: float | None = None
        self.residuals_: np.ndarray | None = None

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
        n = len(signal)
        t = np.arange(n) / sample_rate
        omega_t = 2 * np.pi * rotation_freq * t

        # Design matrix
        cos_col = np.cos(omega_t)
        sin_col = np.sin(omega_t)
        if self.fit_dc:
            X = np.column_stack([cos_col, sin_col, np.ones(n)])
        else:
            X = np.column_stack([cos_col, sin_col])

        # Fit signal
        coeffs_sig, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
        A_sig = float(coeffs_sig[0])
        B_sig = float(coeffs_sig[1])

        # For x(t) = A_peak·cos(ωt + φ) = A_peak·cos(φ)·cos(ωt) - A_peak·sin(φ)·sin(ωt)
        # So: A_sig = A_peak·cos(φ), B_sig = -A_peak·sin(φ)
        # amplitude = sqrt(A² + B²), phase = atan2(-B, A)
        amplitude = float(np.sqrt(A_sig**2 + B_sig**2))
        sig_phase = np.arctan2(-B_sig, A_sig)

        # Fit trigger
        coeffs_trig, _, _, _ = np.linalg.lstsq(X, trigger, rcond=None)
        A_trig = float(coeffs_trig[0])
        B_trig = float(coeffs_trig[1])
        trig_phase = np.arctan2(-B_trig, A_trig)

        # Relative phase
        phase_deg = float(np.degrees(sig_phase - trig_phase))

        # Store diagnostics
        self.A_signal_ = A_sig
        self.B_signal_ = B_sig
        self.A_trigger_ = A_trig
        self.B_trigger_ = B_trig
        self.dc_signal_ = float(coeffs_sig[2]) if self.fit_dc else None
        self.residuals_ = signal - X @ coeffs_sig

        if self.plotting:
            self._plot_fit(
                signal, t, amplitude, phase_deg, sig_phase,
                rotation_freq, self.residuals_,
            )

        return amplitude, phase_deg

    @staticmethod
    def _plot_fit(
        signal: np.ndarray,
        t: np.ndarray,
        amplitude: float,
        phase_deg: float,
        sig_phase: float,
        rotation_freq: float,
        residuals: np.ndarray,
        n_periods: int = 5,
    ) -> None:
        """Plot time-domain overlay and residuals."""
        plt = _import_matplotlib()

        # Wrap phase for display
        display_phase = phase_deg % 360
        if display_phase > 180:
            display_phase -= 360

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # -- Left: time-domain overlay (first n_periods) --
        t_end = n_periods / rotation_freq
        n_show = min(len(t), int(np.searchsorted(t, t_end)))
        if n_show == 0:
            n_show = len(t)

        ax1.plot(t[:n_show], signal[:n_show], color="steelblue",
                 label="Signal")
        reconstructed = amplitude * np.cos(
            2 * np.pi * rotation_freq * t[:n_show] + sig_phase
        )
        ax1.plot(t[:n_show], reconstructed, color="red", linestyle="--",
                 label=f"LS fit: {amplitude:.4g} @ {display_phase:.1f}\u00b0")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Time domain \u2014 first {n_periods} periods")
        ax1.legend()

        # -- Right: residuals --
        ax2.plot(t, residuals, color="steelblue", linewidth=0.5)
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Residual")
        ax2.set_title(f"Residuals \u2014 std = {np.std(residuals):.4g}")

        plt.tight_layout()
        plt.show(block=True)
