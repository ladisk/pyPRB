"""
This is a basic example of dynamic balancing.
"""
from pyPRB import VibrationVector, MassVector, DynamicBalancing

# Initial measurements (no trial masses)
V_1_0 = VibrationVector(7.2, 238)   # Plane 1
V_2_0 = VibrationVector(13.5, 296)  # Plane 2

# With trial mass in Plane 1
V_1_1 = VibrationVector(4.9, 114)   # Plane 1
V_2_1 = VibrationVector(9.2, 347)   # Plane 2

# With trial mass in Plane 2
V_1_2 = VibrationVector(4.0, 79)    # Plane 1
V_2_2 = VibrationVector(12.0, 292)  # Plane 2

# Trial masses: 2.5 g each, mounted at 0° on their respective planes
trial_mass_1 = MassVector(2.5, 0.0)
trial_mass_2 = MassVector(2.5, 0.0)

# Create balancer
balancer = DynamicBalancing(
    V_1_0, V_2_0, V_1_1, V_2_1, V_1_2, V_2_2,
    trial_mass_1=trial_mass_1, trial_mass_2=trial_mass_2
)

# Display measurement table
print(balancer)

# Compute compensations for both planes
comp_mass_1, comp_mass_2 = balancer.compute_compensation()
