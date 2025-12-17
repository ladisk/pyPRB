"""
This is a basic example of static balancing.
"""
from pyPRB import VibrationVector, StaticBalancing

# Initial measurements
V_0 = VibrationVector(amplitude=3.4, phase=116)  # No trial mass
V_1 = VibrationVector(amplitude=1.8, phase=42)   # With 2.0 g trial mass

# Create balancer
balancer = StaticBalancing(V_0, V_1, mass=2.0)

# Display measurement table
print(balancer)

# Compute compensation
compensation_mass, angle = balancer.compute_compensation()