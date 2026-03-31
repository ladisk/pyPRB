"""Synchronous averaging extraction method.

Extracts the fundamental (once-per-revolution) vibration component by
averaging revolution-locked signal segments.
"""

from __future__ import annotations

import numpy as np


class SynchronousAveraging:
    """Extracts the fundamental vibration component via synchronous averaging.

    The trigger signal (e.g. from a photoelectric probe) is used to
    segment the response signal into individual revolutions.  These
    segments are resampled to a common length and averaged, suppressing
    noise that is not synchronous with the rotation.  The fundamental
    component is then extracted from FFT bin 1 of the averaged waveform.

    Parameters
    ----------
    edge : str
        Trigger edge used to define revolution boundaries:
        ``'rising'`` (default) or ``'falling'``.
    period_tolerance : float
        Relative tolerance for rejecting segments whose length deviates
        from the expected rotation period.  Default is 0.1 (±10 %).
    plotting : bool
        If ``True``, show diagnostic plots (trigger crossings and
        synchronous average) during each call to :meth:`extract`.
        Requires matplotlib (install with ``pip install pyPRB[plot]``).

    Attributes
    ----------
    segments_ : list of np.ndarray
        Individual resampled revolution segments from the last call to
        :meth:`extract`.
    averaged_ : np.ndarray
        Synchronously averaged waveform from the last call to
        :meth:`extract`.

    Examples
    --------
    >>> from pyPRB.processing import SynchronousAveraging, VibrationExtractor
    >>> method = SynchronousAveraging(edge='rising', period_tolerance=0.3,
    ...                               plotting=True)
    >>> extractor = VibrationExtractor(sample_rate=10000, rotation_freq=25.0,
    ...                                method=method)
    """

    def __init__(
        self,
        edge: str = "rising",
        period_tolerance: float = 0.1,
        plotting: bool = False,
    ):
        if edge not in ("rising", "falling"):
            raise ValueError(f"edge must be 'rising' or 'falling', got '{edge}'.")
        if not 0 < period_tolerance < 1:
            raise ValueError(
                f"period_tolerance must be between 0 and 1, got {period_tolerance}."
            )
        self.edge = edge
        self.period_tolerance = period_tolerance
        self.plotting = plotting
        self.segments_: list[np.ndarray] = []
        self.averaged_: np.ndarray | None = None

    def extract(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
        sample_rate: float,
        rotation_freq: float,
    ) -> tuple[float, float]:
        """Extract the fundamental amplitude and phase.

        Parameters
        ----------
        signal : np.ndarray
            Response signal (acceleration, velocity, …).
        trigger : np.ndarray
            Trigger signal from a photoelectric probe.
        sample_rate : float
            Sampling rate in Hz.
        rotation_freq : float
            Rotation frequency in Hz.

        Returns
        -------
        amplitude : float
            Peak amplitude of the fundamental component.
        phase_deg : float
            Phase of the fundamental component in degrees.
        """
        crossings = self._find_trigger_crossings(trigger)
        if self.plotting:
            time = np.arange(len(trigger)) / sample_rate
            self._plot_trigger_crossings(time, trigger, crossings, rotation_freq)

        expected_period = sample_rate / rotation_freq
        self.averaged_, self.segments_ = self._synchronous_average(
            signal, crossings, expected_period,
        )
        amplitude, phase_deg = self._extract_fundamental(self.averaged_)

        if self.plotting:
            n_total = len(crossings) - 1
            n_rejected = n_total - len(self.segments_)
            self._plot_synchronous_average(
                self.segments_, self.averaged_, amplitude, phase_deg,
                expected_period=expected_period,
                n_total=n_total,
                n_rejected=n_rejected,
            )

        return amplitude, phase_deg

    def _find_trigger_crossings(self, trigger: np.ndarray) -> np.ndarray:
        """Return sample indices where the trigger signal crosses its mean.

        Parameters
        ----------
        trigger : np.ndarray
            Trigger signal (e.g. from a photoelectric probe).

        Returns
        -------
        np.ndarray
            Integer sample indices of detected crossings.

        Raises
        ------
        ValueError
            If no crossings are found.
        """
        threshold = np.mean(trigger)
        trigger_bool = trigger > threshold
        diff = np.diff(trigger_bool.astype(np.int8))

        if self.edge == "rising":
            crossings = np.where(diff > 0)[0]
        else:
            crossings = np.where(diff < 0)[0]

        if len(crossings) == 0:
            raise ValueError(
                f"No {self.edge} crossings found in the trigger signal. "
                "Check that the trigger channel contains a periodic pulse."
            )

        return crossings

    def _synchronous_average(
        self,
        signal: np.ndarray,
        crossings: np.ndarray,
        expected_period: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Average one-revolution segments of *signal* locked to *crossings*.

        Each segment between consecutive trigger crossings is resampled to a
        common length (median valid period) and then averaged.  Segments whose
        length deviates more than ``self.period_tolerance`` from
        *expected_period* are rejected.

        Parameters
        ----------
        signal : np.ndarray
            Response signal.
        crossings : np.ndarray
            Sample indices of trigger crossings.
        expected_period : float
            Expected period in samples (``sample_rate / rotation_freq``).

        Returns
        -------
        averaged : np.ndarray
            Averaged waveform of length equal to the median valid period.
        segments : list of np.ndarray
            Individual resampled segments.

        Raises
        ------
        ValueError
            If fewer than 2 valid segments remain after rejection.
        """
        periods = np.diff(crossings)

        lo = (1 - self.period_tolerance) * expected_period
        hi = (1 + self.period_tolerance) * expected_period
        valid_mask = (periods >= lo) & (periods <= hi)

        if np.sum(valid_mask) < 2:
            raise ValueError(
                f"Only {int(np.sum(valid_mask))} valid segment(s) found "
                f"(expected period ≈ {expected_period:.0f} samples, "
                f"tolerance ±{self.period_tolerance*100:.0f}%). "
                "Check rotation frequency and trigger signal."
            )

        median_period = np.median(periods[valid_mask])
        target_samples = int(np.round(median_period))

        segments = []
        for i in range(len(crossings) - 1):
            if not valid_mask[i]:
                continue
            start = crossings[i]
            end = crossings[i + 1]
            if start < 0 or end > len(signal):
                continue
            segment = signal[start:end]
            x_old = np.linspace(0, 1, len(segment))
            x_new = np.linspace(0, 1, target_samples)
            resampled = np.interp(x_new, x_old, segment)
            segments.append(resampled)

        averaged = np.mean(np.array(segments), axis=0)
        return averaged, segments

    @staticmethod
    def _extract_fundamental(averaged: np.ndarray) -> tuple[float, float]:
        """Extract the fundamental amplitude and phase from an averaged waveform.

        The averaged waveform spans exactly one revolution, so FFT bin 1
        corresponds to the fundamental (once-per-revolution) component.

        Parameters
        ----------
        averaged : np.ndarray
            One-revolution averaged waveform.

        Returns
        -------
        amplitude : float
            Peak amplitude of the fundamental component.
        phase_deg : float
            Phase of the fundamental component in degrees.
        """
        N = len(averaged)
        H1 = np.fft.rfft(averaged)[1]
        amplitude = float(2 * np.abs(H1) / N)
        phase_deg = float(np.degrees(np.angle(H1)))
        return amplitude, phase_deg

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _import_matplotlib():
        try:
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install pyPRB[plot]"
            ) from None

    def _plot_trigger_crossings(
        self,
        time: np.ndarray,
        trigger: np.ndarray,
        crossings: np.ndarray,
        rotation_freq: float,
        n_periods: int = 5,
    ) -> None:
        """Plot the trigger signal with detected crossings marked."""
        plt = self._import_matplotlib()

        t_end = n_periods / rotation_freq
        mask = time <= t_end

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time[mask], trigger[mask], color="steelblue", label="Trigger")
        ax.axhline(np.mean(trigger), color="gray", linestyle="--",
                    linewidth=0.8, label="Threshold")

        crossings_in_window = crossings[crossings < np.sum(mask)]
        ax.scatter(time[crossings_in_window], trigger[crossings_in_window],
                   color="red", zorder=5, s=30, label="Crossings")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Trigger crossings ({len(crossings)} total, "
                     f"first {n_periods} periods shown)")
        ax.legend()
        plt.tight_layout()
        plt.show(block=True)

    def _plot_synchronous_average(
        self,
        segments: list[np.ndarray],
        averaged: np.ndarray,
        amplitude: float,
        phase_deg: float,
        expected_period: float,
        n_total: int,
        n_rejected: int,
    ) -> None:
        """Plot individual segments, the synchronous average, and the
        fundamental cosine fit."""
        plt = self._import_matplotlib()

        target_samples = len(averaged)
        angle = np.linspace(0, 360, target_samples, endpoint=False)
        angle_rad = np.linspace(0, 2 * np.pi, target_samples, endpoint=False)
        cosine_fit = amplitude * np.cos(angle_rad + np.deg2rad(phase_deg))

        lo = (1 - self.period_tolerance) * expected_period
        hi = (1 + self.period_tolerance) * expected_period

        fig, ax = plt.subplots(figsize=(12, 6))
        for seg in segments:
            ax.plot(angle, seg, color="gray", alpha=0.1)
        ax.plot(angle, averaged, color="red", linewidth=2, label="Average")
        ax.plot(angle, cosine_fit, color="blue", linestyle="--",
                label=f"Fundamental: {amplitude:.4g} @ {phase_deg:.1f}°")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(
            f"Synchronous average \u2014 "
            f"{len(segments)}/{n_total} segments "
            f"({n_rejected} rejected, "
            f"tolerance \u00b1{self.period_tolerance*100:.0f}%)\n"
            f"Expected period: {expected_period:.0f} samples, "
            f"bounds: [{lo:.0f}, {hi:.0f}], "
            f"median period: {target_samples} samples"
        )
        ax.legend()
        plt.tight_layout()
        plt.show(block=True)
