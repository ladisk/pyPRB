"""
From the computed compensation to the weights that are actually fitted.

A fan with 12 blades needs 50 g at 35°.  Weights can only be clamped to the
blades, only 5 g weights are at hand, and the influence coefficient is
known from the trial run.
"""
from pyPRB import (InfluenceCoefficients, MassVector, VibrationVector,
                   change_radius, combine_masses, mass_error,
                   opposite_position, round_mass, split_mass_on_positions)

coefficients = InfluenceCoefficients([[0.08 * (0.6 - 0.8j)]])
reading = VibrationVector(4.0, 161.87)

ideal = coefficients.compute_compensation(reading)[0]
print(f"Ideal compensation: {ideal}")

# Split onto the two nearest blades (blade 0 at 0°)
on_blades = split_mass_on_positions(ideal, n_positions=12)
for blade, part in on_blades:
    print(f"  blade {blade}: {part}")

# Round to 5 g weights
fitted = []
for blade, part in on_blades:
    rounded = round_mass(part, step=5.0)
    print(f"  blade {blade}: fit {rounded.mass}, error {rounded.error}")
    if rounded.mass is not None:
        fitted.append(rounded.mass)

print(f"Fitted weights as one mass: {combine_masses(fitted)}")
print(f"Left on the rotor:          {mass_error(ideal, fitted)}")
after = coefficients.predict_vibration(reading, [fitted])[0]
print(f"Vibration to expect: {after.amplitude:.2f} "
      f"(before: {reading.amplitude:.2f})")

# Other practical variants of the ideal compensation
print(f"At a radius of 200 mm instead of 150 mm: "
      f"{change_radius(ideal, radius=150, new_radius=200)}")
print(f"Removing material instead: {opposite_position(ideal)}")
