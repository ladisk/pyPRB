"""
Choosing a trial mass and judging the trial run.

Both suggestions are rules of thumb; they are starting points.
Units: rotor mass in kg, speed in r/min, radius in mm, masses in g.
"""
from pyPRB import (VibrationVector, assess_trial_run, centrifugal_force,
                   permissible_unbalance, trial_mass_from_force,
                   trial_mass_from_tolerance)

rotor_mass = 150.0  # kg
speed = 1500.0      # r/min
radius = 400.0      # mm

by_force = trial_mass_from_force(rotor_mass, speed, radius)
print(f"Force rule (10 % of the rotor weight): {by_force:.1f} g, "
      f"centrifugal force {centrifugal_force(by_force * radius, speed):.0f} N")

U_per = permissible_unbalance(6.3, rotor_mass, speed)
by_tolerance = trial_mass_from_tolerance(U_per, radius)
print(f"Five times the permissible residual mass (G 6.3): "
      f"{by_tolerance:.1f} g")

# The trial run
V_0 = VibrationVector(3.4, 116)
for label, V_1 in (("first try", VibrationVector(3.7, 122)),
                   ("larger mass", VibrationVector(1.8, 42))):
    a = assess_trial_run(V_0, V_1)
    print(f"{label}: amplitude {a.amplitude_change:+.0%}, "
          f"phase {a.phase_change:+.0f}°, "
          f"effect {a.effect_ratio:.0%} of the initial reading -> "
          f"{'adequate' if a.adequate else 'not adequate'}")
