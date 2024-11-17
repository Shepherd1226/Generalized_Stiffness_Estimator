# data_generator.py

import os
import numpy as np
from scipy.io import savemat  # Import savemat for saving MATLAB-readable files
from k_simulator import F_Pos_simulator
from rnd_generator import Generater
import plot_from_data

def main():
    # Set the random seed for reproducibility
    np.random.seed(3649601111) #3649601111 for data3,364374560 for data2,3334444560 for data1
    
    # Initialize generators and simulators
    F = Generater()
    FN = F.generate_func()
    
    ela = F_Pos_simulator()
    ela.generate()
    
    # Initialize lists to hold data
    curves = []
    F_line = []
    x_line = []
    
    # Simulation parameters
    num_steps = 2000
    num_F_values = 50
    F_values = np.linspace(0, 5, num=num_F_values)
    
    # Run the simulation loop
    for i in range(num_steps):
        curve = []
        for F_val in F_values:
            x = ela.pos_F_func(F_val)[0] * 0.01  # Convert to mm
            curve.append(x)
        curves.append(curve)
        
        current_time = 0.01 * i
        F_current = FN(current_time)
        F_line.append(F_current)
        
        x_current = ela.pos_F_func(F_current)[0] * 0.01
        x_line.append(x_current)
        
        ela.step()  # Advance the simulation
    
    # Convert lists to NumPy arrays
    curves = np.array(curves)            # Shape: (2000, 50)
    F_line = np.array(F_line)            # Shape: (2000,)
    x_line = np.array(x_line)            # Shape: (2000,)
    
    # Save the data to a compressed .npz file for Python use
    npz_output_file = 'simulation_data.npz'
    np.savez_compressed(npz_output_file, curves=curves, F_line=F_line, x_line=x_line, F_values=F_values)
    
    # Save the data to a .mat file for MATLAB use
    mat_output_file = 'simulation_data.mat'
    mat_data = {
        'curves': curves,
        'F_line': F_line,
        'x_line': x_line,
        'F_values': F_values
    }
    savemat(mat_output_file, mat_data)
    
    print(f"Data generation complete. Data saved to '{npz_output_file}' and '{mat_output_file}'.")

if __name__ == "__main__":
    main()
    plot_from_data.main()
