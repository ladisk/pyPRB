"""
How good is a reading?  The 1x vector with its scatter, the speed during
the record and the harmonic content.

A simulated record: the 1x vibration of 3.4 @ 116°, some 2x, noise, and a
speed that drifts slightly during the measurement.
"""
import numpy as np

from pyPRB import VibrationExtractor, rotation_freq_variation

sample_rate = 10000.0
duration = 10.0
t = np.arange(0, duration, 1 / sample_rate)
rng = np.random.default_rng(0)

# Rotation angle with a slow speed drift from 24.95 Hz to 25.05 Hz
angle = 2 * np.pi * (24.95 * t + 0.5 * (0.1 / duration) * t ** 2)
trigger = np.sign(np.sin(angle))
signal = (3.4 * np.sin(angle + np.deg2rad(116))
          + 0.6 * np.sin(2 * angle)
          + 0.3 * rng.standard_normal(len(t)))

# -- Speed during the record -------------------------------------------------
speed = rotation_freq_variation(trigger, sample_rate)
print(f"Speed: {speed.mean:.3f} Hz ({60 * speed.mean:.0f} r/min), "
      f"from {speed.minimum:.3f} Hz to {speed.maximum:.3f} Hz "
      f"({speed.relative_range:.2%} of the mean) "
      f"over {speed.n_revolutions} revolutions")

# -- The 1x vector with its scatter ------------------------------------------
extractor = VibrationExtractor(sample_rate=sample_rate,
                               rotation_freq=speed.mean)
scatter = extractor.get_vector_scatter(signal, trigger, n_parts=10)
print("\n1x vector of ten consecutive parts:")
for k, v in enumerate(scatter.vectors, 1):
    print(f"  part {k:2d}: {v.amplitude:.3f} @ {v.phase:.1f}°")
print(f"Mean vector: {scatter.mean.amplitude:.3f} @ {scatter.mean.phase:.1f}°")
print(f"Amplitude spread: {scatter.amplitude_spread:.3f}, "
      f"phase spread: {scatter.phase_spread:.1f}°, "
      f"radius of the cloud: {scatter.relative_deviation:.1%} of the mean")

# -- Harmonic content ---------------------------------------------------------
content = extractor.get_harmonic_content(signal)
for order, amplitude in zip(content.orders, content.amplitudes):
    print(f"{order}x: {amplitude:.3f}")
print(f"Overall level: {content.overall_rms:.3f} r.m.s.; "
      f"1x is {content.fraction_1x:.0%} of it")
# The harmonic content is evaluated over the whole record at one fixed
# frequency, so the speed drift lowers the 1x and 2x values a little
# compared with the vectors of the shorter parts above.
