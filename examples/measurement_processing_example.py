"""Example: Extract vibration vectors from time-domain signals.

This example demonstrates the full workflow using synthetic signals:
1. Create a VibrationExtractor
2. Detect the rotation frequency from the trigger signal
3. Extract vibration vectors for each measurement run
4. Pass them to StaticBalancing to compute the correction mass

Four extraction methods are compared side by side:
- IQDemodulation (lock-in amplification, cosine/sine correlation) — default
- LeastSquaresFit (least-squares sine fit, optimal under Gaussian noise)
- SynchronousAveraging (time-domain, trigger-crossing based)
- FFTExtraction (frequency-domain, phase subtraction)
"""

import numpy as np
from pyPRB import (
    VibrationExtractor,
    SynchronousAveraging,
    FFTExtraction,
    IQDemodulation,
    LeastSquaresFit,
    detect_rotation_freq,
    MassVector,
    StaticBalancing,
)

# -- Simulation parameters ---------------------------------------------------
f0 = 25.0           # Rotation frequency (Hz)
sample_rate = 10000  # Sampling rate (Hz)
duration = 5.0       # Measurement duration (s)

t = np.arange(0, duration, 1 / sample_rate)

# Trigger signal: square wave from a photoelectric probe
trigger = np.sign(np.sin(2 * np.pi * f0 * t))

# Run 0 (reference): vibration from original unbalance
V_0_amp, V_0_phase = 3.4, 116.0
signal_0 = V_0_amp * np.cos(2 * np.pi * f0 * t + np.deg2rad(V_0_phase))
signal_0 += 0.05 * np.random.default_rng(0).standard_normal(len(t))

# Run 1 (trial mass): vibration changes after adding trial mass
V_1_amp, V_1_phase = 1.8, 42.0
signal_1 = V_1_amp * np.cos(2 * np.pi * f0 * t + np.deg2rad(V_1_phase))
signal_1 += 0.05 * np.random.default_rng(1).standard_normal(len(t))

# -- Processing --------------------------------------------------------------

# Detect rotation frequency from trigger
detected_freq = detect_rotation_freq(trigger, sample_rate=sample_rate, max_freq=50.0,
                                     plot=True)
print(f"Detected rotation frequency: {detected_freq:.2f} Hz")

trial_mass = MassVector(2.0, 0.0)

# -- Method 1: IQ Demodulation (default) --------------------------------------
print("\n--- IQ Demodulation ---")

method_iq = IQDemodulation(remove_dc=True, trim_periods=True, plotting=True)
extractor_iq = VibrationExtractor(sample_rate=sample_rate, rotation_freq=detected_freq,
                                  method=method_iq)

V_0_iq = extractor_iq.get_vibration_vector(signal_0, trigger)
V_1_iq = extractor_iq.get_vibration_vector(signal_1, trigger)

print(f"V_0 = {V_0_iq}")
print(f"V_1 = {V_1_iq}")

balancer_iq = StaticBalancing(V_0_iq, V_1_iq, trial_mass=trial_mass)
print(balancer_iq)
comp_iq = balancer_iq.compute_compensation()
print(f"Result: {comp_iq}")

# -- Method 2: Least-Squares Fit ----------------------------------------------
print("\n--- Least-Squares Fit ---")

method_ls = LeastSquaresFit(fit_dc=True, plotting=True)
extractor_ls = VibrationExtractor(sample_rate=sample_rate, rotation_freq=detected_freq,
                                  method=method_ls)

V_0_ls = extractor_ls.get_vibration_vector(signal_0, trigger)
V_1_ls = extractor_ls.get_vibration_vector(signal_1, trigger)

print(f"V_0 = {V_0_ls}")
print(f"V_1 = {V_1_ls}")

balancer_ls = StaticBalancing(V_0_ls, V_1_ls, trial_mass=trial_mass)
print(balancer_ls)
comp_ls = balancer_ls.compute_compensation()
print(f"Result: {comp_ls}")

# -- Method 3: Synchronous Averaging ------------------------------------------
print("\n--- Synchronous Averaging ---")

method_sa = SynchronousAveraging(edge="rising", period_tolerance=0.1, plotting=True)
extractor_sa = VibrationExtractor(sample_rate=sample_rate, rotation_freq=detected_freq,
                                  method=method_sa)

V_0_sa = extractor_sa.get_vibration_vector(signal_0, trigger)
V_1_sa = extractor_sa.get_vibration_vector(signal_1, trigger)

print(f"V_0 = {V_0_sa}")
print(f"V_1 = {V_1_sa}")

balancer_sa = StaticBalancing(V_0_sa, V_1_sa, trial_mass=trial_mass)
print(balancer_sa)
comp_sa = balancer_sa.compute_compensation()
print(f"Result: {comp_sa}")

# -- Method 4: FFT Extraction -------------------------------------------------
print("\n--- FFT Extraction ---")

method_fft = FFTExtraction(nfft_mult=4, freq_tolerance=0.1, plotting=True)
extractor_fft = VibrationExtractor(sample_rate=sample_rate, rotation_freq=detected_freq,
                                   method=method_fft)

V_0_fft = extractor_fft.get_vibration_vector(signal_0, trigger)
V_1_fft = extractor_fft.get_vibration_vector(signal_1, trigger)

print(f"V_0 = {V_0_fft}")
print(f"V_1 = {V_1_fft}")

balancer_fft = StaticBalancing(V_0_fft, V_1_fft, trial_mass=trial_mass)
print(balancer_fft)
comp_fft = balancer_fft.compute_compensation()
print(f"Result: {comp_fft}")
