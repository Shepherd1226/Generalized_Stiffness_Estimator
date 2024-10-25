# plotter.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    # Load the data from the .npz file
    data_file = 'simulation_data.npz'
    if not os.path.exists(data_file):
        print(f"Data file '{data_file}' not found. Please run 'data_generator.py' first.")
        return
    
    data = np.load(data_file)
    curves = data['curves']         # Shape: (2000, 50)
    F_line = data['F_line']         # Shape: (2000,)
    x_line = data['x_line']         # Shape: (2000,)
    F_values = data['F_values']     # Shape: (50,)
    
    num_steps = curves.shape[0]
    
    # Create a meshgrid for plotting the surface
    X, Y = np.meshgrid(F_values, np.arange(num_steps))
    Z = curves
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='x (mm)')
    
    # Plot the trajectory line
    Y_line = np.arange(num_steps)
    ax.plot3D(F_line, Y_line, x_line, color='red', linewidth=2, label='Trajectory')
    
    # Set labels
    ax.set_xlabel('F (N)')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('x (mm)')
    ax.set_title('3D Simulation Results')
    
    ax.legend()
    
    # Adjust the viewing angle for better visualization (optional)
    ax.view_init(elev=30, azim=45)
    
    # Show the plot
    plt.tight_layout()
    # Save the figure
    plt.savefig('plot.png', dpi=150)
    print("Plot saved as 'plot.png'.")
    plt.show()

if __name__ == "__main__":

    main()
