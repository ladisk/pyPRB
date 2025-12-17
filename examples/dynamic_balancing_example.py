"""
This is a basic example of dynamic balancing.
"""
from pyPRB import VibrationVector, DynamicBalancing

# Initial measurements (no trial masses)
V_1_0 = VibrationVector(7.2, 238)   # Plane 1
V_2_0 = VibrationVector(13.5, 296)  # Plane 2

# With trial mass in Plane 1
V_1_1 = VibrationVector(4.9, 114)   # Plane 1
V_2_1 = VibrationVector(9.2, 347)   # Plane 2

# With trial mass in Plane 2
V_1_2 = VibrationVector(4.0, 79)    # Plane 1
V_2_2 = VibrationVector(12.0, 292)  # Plane 2

# Create balancer
balancer = DynamicBalancing(
    V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2,
    mass_1_1=2.5, mass_2_2=2.5
)

# Display measurement table
print(balancer)

# Compute compensations for both planes
m1, a1, m2, a2 = balancer.compute_compensation()