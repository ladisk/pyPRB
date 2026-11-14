"""IQ demodulation (lock-in amplification) extraction method.

Extracts the vibration component at the rotation frequency by
multiplying the signal with in-phase (cosine) and quadrature (sine)
references at the target frequency and averaging.  Everything not at
the target frequency averages to zero, yielding a narrow-band estimate
of amplitude and phase.
"""

from __future__ import annotations

import numpy as np

from ._vibration_extractor import (_check_edge, _edge_reference,
                                   _import_matplotlib)


class IQDemodulation:
    """Extracts the vibration component at the rotation frequency via IQ demodulation.

    The signal is multiplied by cosine (in-phase) and sine (quadrature)
    references at the rotation frequency and averaged over the
    measurement window.  The averaging acts as an extremely narrow
    bandpass filter centred on the target frequency (bandwidth ≈ 1/T,
    where T is the averaging time).

    The phase is measured from the trigger edge (the datum mark passing
    the probe): the edges are located with sub-sample resolution, a unit
    cosine with zero phase at the edges is demodulated in the same way
    as the signal, and its phase is subtracted from the signal phase.
    The result therefore does not depend on the pulse width of the
    trigger.

    Parameters
    ----------
    remove_dc : bool
        If ``True`` (default), subtract the mean of both signal and
        trigger before demodulation.  This prevents a DC offset from
        leaking into the IQ estimate when the averaging window does not
        span an exact integer number of periods.
    trim_periods : bool
        If ``True`` (default), trim the signal to the nearest integer
        number of complete periods before demodulation.  This
        eliminates spectral leakage from partial periods and ensures
        that the in-phase and quadrature averages are unbiased.
    edge : str
        Trigger edge used as the phase reference: ``'rising'`` (default)
        or ``'falling'``.
    plotting : bool
        If ``True``, show a diagnostic figure during each call to
        :meth:`extract`.  Requires matplotlib
        (install with ``pip install pyPRB[plot]``).

    Attributes
    ----------
    I_signal_ : float or None
        In-phase component of the signal demodulation from the last
        call to :meth:`extract`.
    Q_signal_ : float or None
        Quadrature component of the signal demodulation from the last
        call to :meth:`extract`.
    I_trigger_ : float or None
        In-phase component of the demodulated edge reference from the
        last call to :meth:`extract`.
    Q_trigger_ : float or None
        Quadrature component of the demodulated edge reference from the
        last call to :meth:`extract`.
    n_samples_used_ : int or None
        Number of samples used after optional trimming from the last
        call to :meth:`extract`.

    Examples
    --------
    >>> from pyPRB.processing import IQDemodulation, VibrationExtractor
    >>> method = IQDemodulation(remove_dc=True, trim_periods=True)
    >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=25.0,
    ...                                method=method)
    """

    def __init__(
        self,
        remove_dc: bool = True,
        trim_periods: bool = True,
        edge: str = "rising",
        plotting: bool = False,
    ):
        _check_edge(edge)
        self.remove_dc = remove_dc
        self.edge = edge
        self.trim_periods = trim_periods
        self.plotting = plotting

        # Diagnostic attributes (None until extract() is called)
        self.I_signal_: float | None = None
        self.Q_signal_: float | None = None
        self.I_trigger_: float | None = None
        self.Q_trigger_: float | None = None
        self.n_samples_used_: int | None = None

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
            the trigger edge.

        Raises
        ------
        ValueError
            If *trim_periods* is ``True`` and the signal is shorter
            than one complete period.
        """
        samples_per_period = sample_rate / rotation_freq
        n = len(signal)

        # Trim to integer number of periods
        if self.trim_periods:
            n_periods = int(n / samples_per_period)
            if n_periods < 1:
                raise ValueError(
                    f"Signal is shorter than one period at {rotation_freq} Hz "
                    f"({n} samples < {samples_per_period:.0f} samples/period). "
                    f"Provide a longer signal or set trim_periods=False."
                )
            n = int(round(n_periods * samples_per_period))

        reference = _edge_reference(trigger, sample_rate, rotation_freq,
                                    self.edge)
        sig = np.asarray(signal[:n], dtype=float).copy()
        trig = reference[:n].copy()

        # Remove DC
        if self.remove_dc:
            sig -= np.mean(sig)
            trig -= np.mean(trig)

        # Reference sinusoids
        t = np.arange(n) / sample_rate
        cos_ref = np.cos(2 * np.pi * rotation_freq * t)
        sin_ref = np.sin(2 * np.pi * rotation_freq * t)

        # IQ demodulation — signal
        # For x(t) = A·cos(ωt + φ):
        #   mean(x·cos(ωt)) =  A/2·cos(φ)
        #   mean(x·sin(ωt)) = -A/2·sin(φ)
        # Negate Q so that atan2(Q, I) = atan2(sin(φ), cos(φ)) = φ.
        I_sig = float(np.mean(sig * cos_ref))
        Q_sig = float(-np.mean(sig * sin_ref))
        amplitude = 2.0 * np.sqrt(I_sig**2 + Q_sig**2)
        sig_phase = np.arctan2(Q_sig, I_sig)

        # IQ demodulation — edge reference
        I_trig = float(np.mean(trig * cos_ref))
        Q_trig = float(-np.mean(trig * sin_ref))
        trig_phase = np.arctan2(Q_trig, I_trig)

        # Relative phase
        phase_deg = float(np.degrees(sig_phase - trig_phase))

        # Store diagnostics
        self.I_signal_ = I_sig
        self.Q_signal_ = Q_sig
        self.I_trigger_ = I_trig
        self.Q_trigger_ = Q_trig
        self.n_samples_used_ = n

        if self.plotting:
            self._plot_demodulation(
                signal[:n], t, amplitude, phase_deg,
                sig_phase, trig_phase, rotation_freq,
                I_sig, Q_sig, I_trig, Q_trig,
            )

        return float(amplitude), phase_deg

    @staticmethod
    def _plot_demodulation(
        signal: np.ndarray,
        t: np.ndarray,
        amplitude: float,
        phase_deg: float,
        sig_phase: float,
        trig_phase: float,
        rotation_freq: float,
        I_sig: float,
        Q_sig: float,
        I_trig: float,
        Q_trig: float,
        n_periods: int = 5,
    ) -> None:
        """Plot time-domain overlay and phasor diagram."""
        plt = _import_matplotlib()

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
        # Wrap phase for display: [0, 360) then shift >180 to negative
        display_phase = phase_deg % 360
        if display_phase > 180:
            display_phase -= 360

        ax1.plot(t[:n_show], reconstructed, color="red", linestyle="--",
                 label=f"IQ result: {amplitude:.4g} @ {display_phase:.1f}\u00b0")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Time domain \u2014 first {n_periods} periods")
        ax1.legend()

        # -- Right: phasor diagram --
        # Normalise both phasors to unit length (only phase matters here)
        sig_mag = np.sqrt(I_sig**2 + Q_sig**2)
        trig_mag = np.sqrt(I_trig**2 + Q_trig**2)

        sig_I = I_sig / max(sig_mag, 1e-12)
        sig_Q = Q_sig / max(sig_mag, 1e-12)
        trig_I = I_trig / max(trig_mag, 1e-12)
        trig_Q = Q_trig / max(trig_mag, 1e-12)

        ax2.annotate(
            "", xy=(sig_I, sig_Q), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=2),
        )
        ax2.annotate(
            "", xy=(trig_I, trig_Q), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )
        ax2.plot(0, 0, color="steelblue", label="Signal phasor")
        ax2.plot(0, 0, color="red", label="Trigger phasor")

        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-1.3, 1.3)
        ax2.set_aspect("equal")
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.axvline(0, color="gray", linewidth=0.5)
        ax2.set_xlabel("In-phase (I)")
        ax2.set_ylabel("Quadrature (Q)")
        ax2.set_title(f"Phasor diagram \u2014 "
                      f"relative phase = {display_phase:.1f}\u00b0\n"
                      f"(phasors normalized to unit length)")
        ax2.legend()

        plt.tight_layout()
        plt.show(block=True)
