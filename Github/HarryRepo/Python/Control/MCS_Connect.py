# -*- coding: utf-8 -*-

import serial
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

MCS = serial.Serial(port='COM5',baudrate=115200,timeout=2)

dataout = MCS.readline
print(dataout)




MCS.close()


# t = 0.3*np.linspace(0,10,10000, endpoint=False)
# sqwav = signal.square(2*np.pi*5*t)
# plt.plot(t,sqwav)