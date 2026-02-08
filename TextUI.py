from ComponentModule.components import *
from colours import colour_palette
import time
#from set_up_track import OrderOfOperations
import numpy as np

from TrackProcessing2.genetic_algorithm import repeat as ga
from TrackProcessing2.config import config, real_properties, default_variables, variable_options

def change_variables(variables):
        print("Which do you want to change?")
        for i, key in enumerate(default_variables.keys()):
            print(f"{i}. {key}")
        var_choice = int(input("Enter number: "))
        var_key = list(default_variables.keys())[var_choice]
        print("Select in the range:")
        for i, option in enumerate(variable_options[var_key]):
            print(f"{i}. {option}")
        
        if var_key in ['mass', 'lapNo', 'elevation']:
            var_value = int(input(f"Enter value for {var_key}: "))
        elif var_key in ['track', 'weather', 'tyre']:
            var_value = int(input("Enter number:"))
            var_value = variable_options[var_key][var_value]
        else:
            raise ValueError("Invalid variable key.")
        
        variables[var_key] = var_value

        done = input("Are you done changing variables? (y/n): ")
        while done.lower() != 'y':
            variables = change_variables(variables)
            done = input("Are you done changing variables? (y/n): ")
        return variables
        
        


def main():
    variables = default_variables.copy()

    print("Hello there young one")
    for key, value in variables.items():
        print(f"{key}: {value}")
    use_defaults = 'y'
    #use_defaults = input("Do you want to use default variables? (y/n): ")
    if use_defaults.lower() == 'y':
        pass
    else:
        variables = change_variables(variables)

    for key, value in variables.items():
        print(f"{key}: {value}")
    print("Running genetic algorithm...")

    start_time = time.time()
    racing_line, best_time, vels, mesh = ga(variables)
            


main()
    