# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:34:28 2024

@author: vpixx
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:34:28 2024

@author: vpixx
"""
import numpy as np
import matplotlib.pyplot as plt

# Data
Angle = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 
                  -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15])
Angle_radians = np.deg2rad(Angle)  # Convert to radians for polar plots

PD_right_X = np.array([-51, -51, -45, -35, -24, -10, 3, 16, 29, 40, 48, 52, 53, 
                       50, 44, 34, 24, 11, -3, -17, -29, -40, -47, -52])
PD_right_Y = np.array([3, 17, 29, 40, 50, 52, 53, 50, 44, 36, 24, 11, -3, 
                       -17, -29, -39, -48, -52, -53, -50, -43, -35, -24, -10])

PD_left_X = np.array([53, 51, 45, 35, 24, 10, -4, -17, -30, -40, -48, -53, 
                      -53, -50, -44, -35, -24, -10, 4, 17, 30, 40, 48, 52])
PD_left_Y = np.array([-3, -17, -30, -41, -48, -52, -53, -51, -45, -35, -24, -11, 
                      4, 17, 29, 40, 47, 53, 53, 51, 44, 35, 23, 10])

# Plot 1a: Left and Right PD values
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

# Left PD
axes[0].errorbar(Angle_radians, PD_left_X, yerr=0.5, label="PD_left_X", fmt='o', capsize=5, color='orange')
axes[0].errorbar(Angle_radians, PD_left_Y, yerr=0.5, label="PD_left_Y", fmt='s', capsize=5, color='green')
axes[0].set_title("Left PD Covered")
axes[0].legend()

# Right PD
axes[1].errorbar(Angle_radians, PD_right_X, yerr=0.5, label="PD_right_X", fmt='o', capsize=5, color='orange')
axes[1].errorbar(Angle_radians, PD_right_Y, yerr=0.5, label="PD_right_Y", fmt='s', capsize=5, color='green')
axes[1].set_title("Right PD Covered")
axes[1].legend()

plt.tight_layout()
plt.show()


#Plot 1b, same as above but not polar:
    
#Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Left plot: variables with "left"
axes[0].errorbar(Angle, PD_left_X, yerr=0.5, label="PD_left_X", fmt='o', capsize=5, color='orange')
axes[0].errorbar(Angle, PD_left_Y, yerr=0.5, label="PD_left_Y", fmt='s', capsize=5, color='green')
axes[0].set_title("Left PD covered")
axes[0].set_xlabel("Angle")
axes[0].set_ylabel("Signal (nAmp)")
axes[0].legend()
axes[0].grid(True)

# Right plot: variables with "right"
axes[1].errorbar(Angle, PD_right_X, yerr=0.5, label="PD_right_X", fmt='o', capsize=5, color='orange')
axes[1].errorbar(Angle, PD_right_Y, yerr=0.5, label="PD_right_Y", fmt='s', capsize=5, color='green')
axes[1].set_title("Right PD covered")
axes[1].set_xlabel("Angle")
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# Plot 2a: Left and Right deltas
Right_delta = np.abs(PD_right_X) - np.abs(PD_right_Y)
Left_delta = np.abs(PD_left_X) - np.abs(PD_right_Y)

# Create the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

# Plot Left Delta in blue
ax.plot(Angle_radians, Left_delta, 'o-', label="Left Delta", color='blue')

# Plot Right Delta in red
ax.plot(Angle_radians, Right_delta, 's-', label="Right Delta", color='red')

# Title, legend, and layout
ax.set_title("Left and Right PD X-Y Deltas", va='bottom')
ax.legend()

plt.tight_layout()
plt.show()



# Plot 2b: Left and Right deltas
Right_delta_abs = np.abs(np.abs(PD_right_X) - np.abs(PD_right_Y))
Left_delta_abs = np.abs(np.abs(PD_left_X) - np.abs(PD_right_Y))

# Create the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

# Plot Left Delta in blue
ax.plot(Angle_radians, Left_delta_abs, 'o-', label="Left Delta", color='blue')

# Plot Right Delta in red
ax.plot(Angle_radians, Right_delta_abs, 's-', label="Right Delta", color='red')

# Title, legend, and layout
ax.set_title("Left and Right PD X-Y Deltas", va='bottom')
ax.legend()

plt.tight_layout()
plt.show()






# Plot 3: Compute differences
X_difference = PD_left_X + PD_right_X
Y_difference = PD_left_Y + PD_right_Y

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the differences
ax.errorbar(Angle, X_difference, yerr=0.5, label="X Difference", fmt='o', capsize=5, color='blue')
ax.errorbar(Angle, Y_difference, yerr=0.5, label="Y Difference", fmt='s', capsize=5, color='green')

# Labels and legend
ax.set_title("Left-to-Right Differences")
ax.set_xlabel("Angle")
ax.set_ylabel("Difference in Signal (nAmp)")
ax.legend()
ax.grid(True)

# Show the plot
plt.tight_layout()
plt.show()