"""
Trial masses left on, stored influence coefficients and a trim correction.

The first job on a machine needs trial runs.  Here every trial mass stays
on the rotor.  The influence coefficients are then kept as plain data, and
a later job on the same machine needs only one reading.
"""
import json

from pyPRB import (InfluenceCoefficients, MassVector, MultiPlaneBalancing,
                   VibrationVector)

# -- First job: trial runs, each trial mass left on --------------------------
V_0 = [VibrationVector(7.2, 238), VibrationVector(13.5, 296)]
V_trial = [
    [VibrationVector(4.9, 114), VibrationVector(9.2, 347)],  # mass 1 on
    [VibrationVector(6.1, 30), VibrationVector(8.0, 340)],   # masses 1 and 2 on
]
trial_masses = [MassVector(2.5, 0.0), MassVector(2.5, 0.0)]

balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses, trial_left_on=True)
print(balancer)

print("\nMasses to add while the trial masses stay on:")
to_add = balancer.compute_compensation(keep_trial_masses=True)

# -- Keep the coefficients ----------------------------------------------------
coefficients = balancer.get_influence_coefficients()
print(f"\nCondition number: {coefficients.check_conditioning():.2f}")
stored = json.dumps(coefficients.to_dict())  # write this to a file

# -- Later job: no trial runs -------------------------------------------------
coefficients = InfluenceCoefficients.from_dict(json.loads(stored))
check = [VibrationVector(0.9, 75), VibrationVector(1.4, 310)]

print("\nTrim correction from one check reading:")
for plane, mass in enumerate(coefficients.compute_compensation(check), 1):
    print(f"  plane {plane}: {mass}")

print("\nEquivalent unbalance of the reading:")
for plane, mass in enumerate(coefficients.residual_unbalance(check), 1):
    print(f"  plane {plane}: {mass}")

print("\nVibration to expect with 0.3 g at 45° in plane 1 only:")
fitted = [MassVector(0.3, 45), None]
for point, v in enumerate(coefficients.predict_vibration(check, fitted), 1):
    print(f"  point {point}: {v}")
