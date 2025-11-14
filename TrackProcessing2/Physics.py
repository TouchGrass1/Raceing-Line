from enum import Enum
import numpy as np

class PhysicsConsts(Enum):
    TURNING_RADIUS = 5 #M
    VELOCITY_MAX = 380 #KPH
    POWER = 750 #kW
    ACCEL_MIN = -24.5 #MS-2
    ACCEL_MAX = 10.7 #MS-2
    MASS_MIN = 878 #KG
    MASS_MAX = 988 #KG starting value 110kg of fuel
    FUEL_MAX = 110
    ACCEL_CORNER_MAX = 4 #gs not including vertical accel, only lateral
    MU_SOFT = 1.5
    MU_MED = 1.4
    MU_HARD = 1.3
    g = 9.81
    TYRE_LIFE_SOFT = 20
    TYRE_LIFE_MEDIUM = 40
    TYRE_LIFE_HARD = 60
    

class PhysicsFormulas:
    def downforceEquation(x): #x = velocity
        return 0.174686*(x**2) + 25.6869*x + 101.731
    def lateralForceEquation(mass, velocity, radius):
        return (mass*(velocity**2))/radius
    def maxLateralForceEquation(mu, mass, downforce):
        weight = mass*PhysicsConsts['g'].value
        lateral_force = weight * mu
        return lateral_force + downforce
    def maxLateralAccelerationEquation(force, mass):
        f2 = PhysicsConsts['ACCEL_CORNER_MAX'].value*PhysicsConsts['g'].value*mass
        if force > f2: return f2  
        return force
    def tyreWearEquation(num_laps, start_mu, TYRE_LIFE):
        return ((-start_mu * 2)/(10*TYRE_LIFE))*num_laps + start_mu
    def forceToVelocity(force, mass, radius):
        return ((force*radius)/mass)**0.5
    def accelerationPowerEquation(speed, mass):
        if speed == 0: return PhysicsConsts["ACCEL_MAX"].value
        return PhysicsConsts['POWER'].value/(speed*mass)
