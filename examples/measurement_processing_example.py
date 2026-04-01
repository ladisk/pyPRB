"""Example: Extract vibration vectors from time-domain signals.

This example demonstrates the full workflow using synthetic signals:
1. Create a VibrationExtractor
2. Detect the rotation frequency from the trigger signal
3. Extract vibration vectors for each measurement run
4. Pass them to StaticBalancing to compute the correction mass

Two extraction methods are compared side by side:
- SynchronousAveraging (time-domain, trigger-crossing based)
- FFTExtraction (frequency-domain, phase subtraction)
"""

import numpy as np
from pyPRB import (
    VibrationExtractor,
    SynchronousAveraging,
    FFTExtraction,
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

# -- Method 1: Synchronous Averaging ----------------------------------------
print("\n--- Synchronous Averaging ---")

method_sa = SynchronousAveraging(edge="rising", period_tolerance=0.1, plotting=True)
extractor_sa = VibrationExtractor(sample_rate=sample_rate, rotation_freq=detected_freq,
                                  method=method_sa)

V_0_sa = extractor_sa.get_vibration_vector(signal_0, trigger)
V_1_sa = extractor_sa.get_vibration_vector(signal_1, trigger)

print(f"V_0 = {V_0_sa}")
print(f"V_1 = {V_1_sa}")

trial_mass = MassVector(2.0, 0.0)
balancer_sa = StaticBalancing(V_0_sa, V_1_sa, trial_mass=trial_mass)
print(balancer_sa)
comp_sa = balancer_sa.compute_compensation()
print(f"Result: {comp_sa}")

# -- Method 2: FFT Extraction -----------------------------------------------
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
