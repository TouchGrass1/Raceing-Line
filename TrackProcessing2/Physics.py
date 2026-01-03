from enum import Enum
import numpy as np

class PhysicsConsts(Enum):
    TURNING_RADIUS = 5 #M
    VELOCITY_MAX = 380 #KPH
    POWER = 750000 #W
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
        if radius > 300: return 0 #largest radius of a turn is 250m
        else: return (mass*(velocity**2))/radius
    def maxLateralForceEquation(mu, mass, downforce):
        weight = mass*PhysicsConsts['g'].value
        lateral_force = (weight + downforce) * mu
        return lateral_force 
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
    def maxThrustEquation(density): #finds when thrust == drag
        vel = 0
        run = True 
        while run:
            vel +=1
            thrust = updateVar.updateThrust(vel)
            drag = updateVar.updateDrag(vel, density)
            if drag > thrust:
                run = False
                vel = vel - 1
                thrust = updateVar.updateThrust(vel)
        return thrust, vel 


class updateVar:
    def updateMass(noLap, maxNoLap): #end of every lap
        if maxNoLap == 0: return PhysicsConsts['MASS_MAX'].value
        return noLap*((PhysicsConsts['MASS_MIN'].value - PhysicsConsts['MASS_MAX'].value)/maxNoLap) + PhysicsConsts['MASS_MAX'].value

    def updateDownforce(vel):
        return 0.17468*(vel**2) + 25.68969*vel + 101.731 #constants are from desmos can change

    def updateTyreWear(tyreType, noLap): #end of every lap
        tType = f"MU_{tyreType.upper()}"
        maxLife = f"TYRE_LIFE_{tyreType.upper()}"
        return -noLap * (0.2 * PhysicsConsts[tType].value) / PhysicsConsts[maxLife].value + PhysicsConsts[tType].value

    def updateThrust(vel):
        if vel == 0: return PhysicsConsts['POWER'].value
        return PhysicsConsts['POWER'].value/vel

    def updatePressure(height, temp): #height (m) temp(deg) #only once
        pressureSeaLevel = 101325
        molarMass = 0.02897
        temp += 273.15
        R = 8.314
        if temp == 0: raise Exception("Sorry, can't work at absolute 0")
        return pressureSeaLevel* np.e ** ((-molarMass * PhysicsConsts['g'].value * height)/ (R * temp))

    def updateDensity(pressure, temp): #only once
        temp += 273.15
        R = 287.058 # specifc gas constant of dry air
        if temp == 0: raise Exception("Sorry, can't work at absolute 0")
        return (pressure) / (R * temp)
    
    def updateDrag(vel, density):
        drag_coeff = 0.9 # typically between 0.7 and 1.1
        A = 1.5156 #for2023 RED BULL
        return drag_coeff * A * density * 0.5 * np.square(vel)
    
    def updateResultantForce(thrust, FRes):
        return thrust - FRes
    
    def updateStaticForceFriction(mu, mass, downForce):
        return mu*(mass*PhysicsConsts['g'].value + downForce)


