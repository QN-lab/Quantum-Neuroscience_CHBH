# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:45:02 2025

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt

# Data
coupler = np.array([21,31,41,52,63,70,82,93,104,121,136,154,174,191,212,232,255,275,294,313,328,352,374,393,404,430,450])
coupler_err = 2

fiber = np.array([11,16,22,26,33,36,41,47,53,64,71,80,92,99,110,124,135,145,155,166,173,184,191,206,215,230,241])
fiber_err = 2

cells = np.array([1.85,2.2,2.7,3.4,4.0,4.13,4.25,4.9,5.7,6.7,7.4,7.8,8.8,9.3,9.7,10.8,11.9,12.6,13.4,14.4,15.1,16.0,16.7,17.9,18.6,19.9,20.9])
cell_err = 0.5

pd = np.array([230.0,242.3,251.5,258.9,268.9,269.2,270.5,275.9,280.2,287.0,290.2,295.2,299.5,302.9,305.2,308.9,312.1,314.1,316.8,319.0,320.8,322.9,325.3,327.8,329.4,332.3,333.9])
pd_err = 1

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

# Plot each dataset in a separate subplot
axes[0].errorbar(coupler, fiber, yerr=fiber_err, fmt='o-', capsize=3, label='Fiber', color='b')
axes[0].set_ylabel("Power after fiber(uW)")
axes[0].grid(True)
axes[0].legend()

axes[1].errorbar(coupler, cells, yerr=cell_err, fmt='s-', capsize=3, label='Cells', color='g')
axes[1].set_ylabel("Power before cells (uW)")
axes[1].grid(True)
axes[1].legend()

axes[2].errorbar(coupler, pd, yerr=pd_err, fmt='d-', capsize=3, label='PD', color='r')
axes[2].set_ylabel("Photodiode voltage (mV)")
axes[2].grid(True)
axes[2].legend()

# Set x-axis label for the last subplot
axes[2].set_xlabel("Power delivered to coupler (uW)")

# Adjust layout
plt.tight_layout()
plt.show()

#%%
# Compute efficiency and propagate error
eff = fiber / coupler
eff_err = eff * np.sqrt((fiber_err / fiber) ** 2 + (coupler_err / coupler) ** 2)

# Convert to percentage
eff *= 100
eff_err *= 100

# Plot
plt.figure(figsize=(6, 4))
plt.errorbar(coupler, eff, yerr=eff_err, fmt='o-', capsize=3, color='b', label='Fiber/Coupler (%)')

print(f"Average efficiency: {np.mean(eff):.0f} pm {np.std(eff):.0f} Percent")

# Labels and formatting
plt.xlabel("Power delivered to coupler (uW)")
plt.ylabel("Efficiency (%)")
plt.grid(True)

# Show plot
plt.show()

#%%
from scipy.optimize import curve_fit

# Define the logarithmic model
def log_func(x, a, b):
    return a + b * np.log(x)

# Fit the model to the data
params, covariance = curve_fit(log_func, cells, pd)

# Extract best-fit parameters
a_fit, b_fit = params

# Create a smooth range for plotting the fit
cells_fit = np.linspace(min(cells), max(cells), 100)
pd_fit = log_func(cells_fit, a_fit, b_fit)

# Plot data and fit
plt.figure(figsize=(6, 4))
plt.scatter(cells, pd, label="Data", color='b')
plt.plot(cells_fit, pd_fit, label=f"Fit: PD = {a_fit:.2f} + {b_fit:.2f} * ln(P_before_cells)", color='r')
plt.xlabel("Power before Cells (uW)")
plt.ylabel("PD voltage (mV)")
plt.legend()
plt.grid(True)
plt.show()


#%%
from sympy import symbols, Eq, ln, latex

# Function to predict cells for a given PD value
def predict_cells(pd_value):
    return np.exp((pd_value - a_fit) / b_fit)

# Example usage
input_pd = 267.4  # Change this to test different values
predicted_cells = predict_cells(input_pd)
print(f"Predicted cells for PD = {input_pd}: {predicted_cells:.2f}uW")


