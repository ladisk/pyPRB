"""
This is a basic example of multi-plane balancing (three correction planes,
three measurement points).
"""
from pyPRB import VibrationVector, MassVector, MultiPlaneBalancing

# Initial run (no trial masses): one vector per measurement point
V_0 = [VibrationVector(5.1, 40), VibrationVector(7.3, 115),
       VibrationVector(4.2, 260)]

# One trial run per correction plane, each measured at all points
V_trial = [
    [VibrationVector(3.0, 75), VibrationVector(6.8, 128),
     VibrationVector(4.5, 250)],  # trial mass in plane 1
    [VibrationVector(5.6, 22), VibrationVector(4.1, 150),
     VibrationVector(3.9, 242)],  # trial mass in plane 2
    [VibrationVector(5.0, 48), VibrationVector(7.9, 104),
     VibrationVector(2.2, 300)],  # trial mass in plane 3
]

# Trial masses: 2.0 g each, mounted at 0° on their respective planes
trial_masses = [MassVector(2.0, 0.0), MassVector(2.0, 0.0),
                MassVector(2.0, 0.0)]

# Create balancer
balancer = MultiPlaneBalancing(V_0, V_trial, trial_masses)

# Display measurement table
print(balancer)

# Compute compensations for all planes
compensation = balancer.compute_compensation()
