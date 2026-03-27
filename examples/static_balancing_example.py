"""
This is a basic example of static balancing.
"""
from pyPRB import VibrationVector, MassVector, StaticBalancing

# Initial measurements
V_0 = VibrationVector(amplitude=3.4, phase=116)  # No trial mass
V_1 = VibrationVector(amplitude=1.8, phase=42)   # With trial mass

# Trial mass: 2.0 g mounted at 0° on the rotor
trial_mass = MassVector(2.0, 0.0)

# Create balancer
balancer = StaticBalancing(V_0, V_1, trial_mass=trial_mass)

# Display measurement table
print(balancer)

# Compute compensation
comp_mass = balancer.compute_compensation()
