"""
The same balancing job with angles counted with and against the rotation.

pyPRB counts angles from the trigger/datum mark in the direction of
rotation by default.  Many instruments count against the rotation and show
the phase as a lag.  Both describe the same place on the rotor.
"""
from pyPRB import (MassVector, StaticBalancing, VibrationVector,
                   convert_convention)

# Readings with the rotation (pyPRB default), B&K example 2
V_0 = VibrationVector(3.4, 116)
V_1 = VibrationVector(1.8, 42)
trial_mass = MassVector(2.0, 20.0)

print("Angles with the direction of rotation:")
with_rotation = StaticBalancing(V_0, V_1, trial_mass).compute_compensation()

# The same readings and trial position as an instrument would show them
against = "against_rotation"
V_0_a = convert_convention(V_0, "with_rotation", against)
V_1_a = convert_convention(V_1, "with_rotation", against)
trial_mass_a = convert_convention(trial_mass, "with_rotation", against)
print(f"\nInstrument readings: {V_0_a}, {V_1_a}; trial mass {trial_mass_a}")

print("Angles against the direction of rotation:")
balancer = StaticBalancing(V_0_a, V_1_a, trial_mass_a,
                           angle_convention=against)
against_rotation = balancer.compute_compensation()

back = convert_convention(against_rotation, against, "with_rotation")
print(f"\nThe same place counted with the rotation: {back.phase:.1f}°")
