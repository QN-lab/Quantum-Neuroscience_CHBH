# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:21:56 2022

@author: kowalcau-admin
"""
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation


ard = serial.Serial('COM7',9600)
ard.close()
ard.open()

# while True:
#     data = ard.readline()
#     print(data.decode())

x_len = 200     # Number of points to display
ys = [0] * x_len
xs = list(range(0, 200))
     
     
# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = ax.plot(xs, ys)

# This function is called periodically from FuncAnimation
def animate(i, ys):

    # Read line from arduino
    data = ard.readline()
    
    # Add x and y to lists
    ys.append(float(data.decode()))

    # Limit y list to set number of items
    ys = ys[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)
    
    
    return line, 

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(ys,), interval=50, blit=True)
plt.show()