import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径       

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from k_simulator import F_Pos_simulator
from rnd_generator import Generater

np.random.seed(166)

F=Generater()
FN=F.generate_func()

ela=F_Pos_simulator()
ela.generate()


# Initialize a list to hold all the 2D curves
curves = []
F_line= []
x_line= []
# Loop through 2000 steps
for i in range(2000):
    curve = []
    for F in np.linspace(0, 5, num=50):  # Adjust the num parameter as needed for resolution
        curve.append(ela.pos_F_func(F)[0]*0.01)  # Set the current F value
    F_line.append(FN(0.01*i))
    x_line.append(ela.pos_F_func(FN(0.01*i))[0]*0.01)
    ela.step()  # Perform a step and store the result
    curves.append(curve)

# Convert the list of curves to a numpy array for plotting
curves = np.array(curves)

# Create a meshgrid for the X (F values), Y (step number), and Z (curve values) axes
F_values = np.linspace(0, 5, num=50)
steps = np.arange(2000)
X, Y = np.meshgrid(F_values, steps)
Z = curves

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot the series of points as a line
# Convert steps to the correct shape for plotting
Y_line = np.arange(2000)
# Plot the line
ax.plot3D(F_line, Y_line, x_line, color='red')  # Use a distinct color for visibility

ax.set_xlabel('F(N)')
ax.set_ylabel('t')
ax.set_zlabel('x(mm)')

# Save the figure
plt.savefig('plot.png', dpi=300)

plt.show()
