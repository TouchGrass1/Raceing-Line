from enum import Enum
import numpy as np

class PhysicsConsts(Enum):
    TURNING_RADIUS = 5 #M
    MAX_RADIUS = 300  # meters
    SPEED_OF_SOUND = 343  # m/s at sea level
    ABSOLUTE_ZERO = -273.15  # deg C
    VELOCITY_MAX = 112 #m/s
    POWER = 750000 #W
    ACCEL_MIN = -24.5 #MS-2
    ACCEL_MAX = 10.7 #MS-2
    MASS_MIN = 878 #KG
    MASS_MAX = 988 #KG starting value 110kg of fuel
    FUEL_MAX = 110
    ACCEL_CORNER_MAX = 4 #gs not including vertical accel, only lateral
    MU_SOFT = 1.5
    MU_MEDIUM = 1.4
    MU_HARD = 1.3
    g = 9.81
    TYRE_LIFE_SOFT = 20
    TYRE_LIFE_MEDIUM = 40
    TYRE_LIFE_HARD = 60
    HEIGHT_MAX = 2200 #meters in mexico
    HEIGHT_MIN = -28 #meters compared to sea level in baku
    

class PhysicsFormulas:
    def downforceEquation(x): #x = velocity
        PhysicsValidator.validate_speed(x)
        return 0.174686*(x**2) + 25.6869*x + 101.731
    def lateralForceEquation(mass, velocity, radius):
        PhysicsValidator.validate_mass(mass)
        PhysicsValidator.validate_speed(velocity)
        PhysicsValidator.validate_radius(radius)
        return (mass*(velocity**2))/radius
    def maxLateralForceEquation(mu, mass, downForce):
        PhysicsValidator.validate_positive(mu)
        PhysicsValidator.validate_mass(mass)
        PhysicsValidator.validate_positive(downForce)
        weight = mass*PhysicsConsts['g'].value
        lateral_force = (weight + downForce) * mu
        return lateral_force 
    def maxLateralAccelerationEquation(force, mass):
        PhysicsValidator.validate_mass(mass)
        f2 = PhysicsConsts['ACCEL_CORNER_MAX'].value*PhysicsConsts['g'].value*mass
        if force > f2: return f2  
        return force
    def tyreWearEquation(num_laps, start_mu, TYRE_LIFE):
        PhysicsValidator.validate_positive(num_laps)
        return ((-start_mu * 2)/(10*TYRE_LIFE))*num_laps + start_mu
    def forceToVelocity(force, mass, radius):
        PhysicsValidator.validate_mass(mass)
        PhysicsValidator.validate_radius(radius)
        PhysicsValidator.validate_positive(force)
        return ((force*radius)/mass)**0.5
    def accelerationPowerEquation(speed, mass):
        PhysicsValidator.validate_mass(mass)
        PhysicsValidator.validate_speed(speed)
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
            if vel > PhysicsConsts['VELOCITY_MAX'].value:
                run = False
        return thrust, vel 


class updateVar:
    def updateMass(noLap, maxNoLap): #end of every lap
        if maxNoLap == 0: return PhysicsConsts['MASS_MAX'].value
        return noLap*((PhysicsConsts['MASS_MIN'].value - PhysicsConsts['MASS_MAX'].value)/maxNoLap) + PhysicsConsts['MASS_MAX'].value

    def updateDownforce(vel):
        PhysicsValidator.validate_speed(vel)
        return 0.17468*(vel**2) + 25.68969*vel

    def updateTyreWear(tyreType, noLap): #end of every lap
        PhysicsValidator.validate_positive(noLap)
        tType = f"MU_{tyreType.upper()}"
        maxLife = f"TYRE_LIFE_{tyreType.upper()}"
        mu = -noLap * (0.2 * PhysicsConsts[tType].value) / PhysicsConsts[maxLife].value + PhysicsConsts[tType].value
        PhysicsValidator.validate_positive(mu)
        return mu

    def updateThrust(vel):
        PhysicsValidator.validate_speed(vel)
        if vel == 0: return PhysicsConsts['POWER'].value
        return PhysicsConsts['POWER'].value/vel

    def updatePressure(height, temp): #height (m) temp(deg) #only once
        PhysicsValidator.validate_temperature(temp)
        PhysicsValidator.validate_height(height)
        pressureSeaLevel = 101325
        molarMass = 0.02897
        temp += 273.15
        R = 8.314
        return pressureSeaLevel* np.e ** ((-molarMass * PhysicsConsts['g'].value * height)/ (R * temp))

    def updateDensity(pressure, temp): #only once
        PhysicsValidator.validate_temperature(temp)
        temp += 273.15
        R = 287.058 # specifc gas constant of dry air
        return (pressure) / (R * temp)
    
    def updateDrag(vel, density):
        PhysicsValidator.validate_positive(density)
        drag_coeff = 0.9 # typically between 0.7 and 1.1
        A = 1.5156 #for2023 RED BULL
        return drag_coeff * A * density * 0.5 * np.square(vel)
    
    def updateResultantForce(thrust, FRes):
        return thrust - FRes
    
    def updateStaticForceFriction(mu, mass, downForce):
        PhysicsValidator.validate_positive(mu)
        PhysicsValidator.validate_mass(mass)
        PhysicsValidator.validate_positive(downForce)
        return mu*(mass*PhysicsConsts['g'].value + downForce)

class PhysicsValidator:
    def validate_temperature(temperature):
        if temperature < PhysicsConsts['ABSOLUTE_ZERO'].value:
            raise ValueError("Temperature cannot be below absolute zero.")
        return True
    
    def validate_radius(radius):
        if radius <= 0 or radius > PhysicsConsts['MAX_RADIUS'].value:
            raise ValueError(f"Radius must be between 0 and {PhysicsConsts['MAX_RADIUS'].value} meters.")
        return True
    
    def validate_speed(speed):
        if speed < 0 or speed > PhysicsConsts['VELOCITY_MAX'].value:
            raise ValueError(f"Current speed is: {speed}\nSpeed must be between 0 and {PhysicsConsts['VELOCITY_MAX'].value} m/s.")
        return True
    
    def validate_mass(mass):
        if mass < PhysicsConsts['MASS_MIN'].value or mass > PhysicsConsts['MASS_MAX'].value:
            raise ValueError(f"Mass must be between {PhysicsConsts['MASS_MIN'].value} KG and {PhysicsConsts['MASS_MAX'].value} KG.")
        return True
    
    def validate_positive(var): #for mu, lapnumber, density, pressure, mass, square root etc
        if var < 0:
            raise ValueError("This variable must be positive.")
        return True
    
    def validate_height(height):
        if height < PhysicsConsts['HEIGHT_MIN'].value or height > PhysicsConsts['HEIGHT_MAX'].value:
            raise ValueError(f"Height must be between {PhysicsConsts['HEIGHT_MIN'].value} meters and {PhysicsConsts['HEIGHT_MAX'].value} meters.")
        return True

print(updateVar.updateDensity(0, 0))