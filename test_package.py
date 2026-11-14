import io
import sys

from pyPRB import DynamicBalancing, MassVector, StaticBalancing, VibrationVector

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test 1: Create a vibration vector
v = VibrationVector(10.0, 45.0)
print(f"Created: {v}")

# Test 2: Vector math
v1 = VibrationVector(10.5, 45)
v2 = VibrationVector(5.2, 135)
v3 = v1 + v2
print(v3)

# Example 1 - Static Balancing
V_0 = VibrationVector(15, 55)
V_1 = VibrationVector(18, 170)
balancer = StaticBalancing(V_0, V_1, trial_mass=MassVector(5.0, 0.0))
compensation = balancer.compute_compensation()  # 2.6 g at 37°

# Example 2 - Static Balancing
V_0 = VibrationVector(3.4, 116)
V_1 = VibrationVector(1.8, 42)
balancer = StaticBalancing(V_0, V_1, trial_mass=MassVector(2.0, 0.0))
compensation = balancer.compute_compensation()  # 2.03 g at -31°
print(balancer)

# Example 4 - Dynamic Balancing
V_1_0 = VibrationVector(amplitude=7.2, phase=238)
V_2_0 = VibrationVector(amplitude=13.5, phase=296)
V_1_1 = VibrationVector(amplitude=4.9, phase=114)
V_2_1 = VibrationVector(amplitude=9.2, phase=347)
V_1_2 = VibrationVector(amplitude=4, phase=79)
V_2_2 = VibrationVector(amplitude=12, phase=292)
mass_1_1 = 2.5
mass_2_2 = 2.5

balancer = DynamicBalancing(V_1_0=V_1_0,
                            V_2_0=V_2_0,
                            V_1_1=V_1_1,
                            V_2_1=V_2_1,
                            V_1_2=V_1_2,
                            V_2_2=V_2_2,
                            trial_mass_1=MassVector(mass_1_1, 0.0),
                            trial_mass_2=MassVector(mass_2_2, 0.0))
compensation_1, compensation_2 = balancer.compute_compensation()  # P1: 2.93 g at 50.4°, P2: 2.84 g, at -81.9°
print(balancer)


# Example 5 - Dynamic Balancing
V_1_0 = VibrationVector(amplitude=170, phase=112)
V_2_0 = VibrationVector(amplitude=53, phase=78)
V_1_1 = VibrationVector(amplitude=235, phase=94)
V_2_1 = VibrationVector(amplitude=58, phase=68)
V_1_2 = VibrationVector(amplitude=185, phase=115)
V_2_2 = VibrationVector(amplitude=77, phase=104)
mass_1_1 = 1.15
mass_2_2 = 1.15

balancer = DynamicBalancing(V_1_0=V_1_0, V_2_0=V_2_0,
                            V_1_1=V_1_1, V_2_1=V_2_1,
                            V_1_2=V_1_2, V_2_2=V_2_2,
                            trial_mass_1=MassVector(mass_1_1, 0.0),
                            trial_mass_2=MassVector(mass_2_2, 0.0))
compensation_1, compensation_2 = balancer.compute_compensation()  # P1: 1.98 g at 236.2°, P2: 1.07 g at 121.8°
print(balancer)



print("\nAll tests passed!")
