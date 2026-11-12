"""
Balance tolerance from a balance quality grade, and a check against it.

Units: grade in mm/s, rotor mass in kg, speed in r/min, unbalance in g·mm,
radius in mm, masses in g.
"""
from pyPRB import (MassVector, allocate_tolerance, check_residual,
                   permissible_mass, permissible_specific_unbalance,
                   permissible_unbalance)

grade = 6.3         # G 6.3
rotor_mass = 150.0  # kg
speed = 1500.0      # r/min
radius = 400.0      # mm, where correction masses are mounted

U_per = permissible_unbalance(grade, rotor_mass, speed)
print(f"Permissible residual unbalance: {U_per:.0f} g·mm")
print(f"  per kg of rotor mass: "
      f"{permissible_specific_unbalance(grade, speed):.1f} g·mm/kg")
print(f"  as a mass at {radius:.0f} mm: "
      f"{permissible_mass(grade, rotor_mass, speed, radius):.1f} g")

# Two planes, 300 mm and 500 mm from the centre of mass
U_1, U_2 = allocate_tolerance(U_per, distance_1=300.0, distance_2=500.0)
print(f"Plane 1: {U_1:.0f} g·mm, plane 2: {U_2:.0f} g·mm")

# Residual masses found in a check run
residuals = [MassVector(4.2, 130.0), MassVector(6.5, -20.0)]
for plane, (residual, permitted) in enumerate(zip(residuals, (U_1, U_2)), 1):
    check = check_residual(residual, permitted, radius=radius)
    verdict = "within" if check.within else "outside"
    print(f"Plane {plane}: {check.residual:.0f} g·mm of {permitted:.0f} g·mm "
          f"({check.ratio:.0%}) - {verdict} tolerance")
