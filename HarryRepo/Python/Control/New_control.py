# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:55:57 2023

@author: H
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import zhinst.core
import toptica.lasersdk
import sys
import time

#%%% Laser setup

from toptica.lasersdk.dlcpro.v1_6_3 import DLCpro, NetworkConnection

with DLCpro(NetworkConnection('172.29.169.14')) as dlcpro:
    print(dlcpro.system_label.get())
    dlcpro.system_label.set('Please do not touch!')






# Zhinst setup 
#%% DAQ
daq = zhinst.core.ziDAQServer('172.29.168.20', 8004, 6)

#DAQ
daq_module = daq.dataAcquisitionModule()
daq_module.set('triggernode', '/dev3994/demods/0/sample.R')
daq_module.set('preview', 1)
daq_module.set('device', 'dev3994')
daq_module.set('historylength', 100)
daq_module.set('count', 10)
daq_module.set('type', 6)
daq_module.set('edge', 1)
daq_module.set('bits', 1)
daq_module.set('bitmask', 1)
daq_module.set('eventcount/mode', 1)
daq_module.set('delay', 0)
daq_module.set('grid/mode', 4)
daq_module.set('grid/cols', 4096)
daq_module.set('duration', 4.89335467)
daq_module.set('bandwidth', 0)
daq_module.set('pulse/min', 0)
daq_module.set('pulse/max', 0.001)
daq_module.set('holdoff/time', 0)
daq_module.set('holdoff/count', 0)
daq_module.set('grid/rows', 1)
daq_module.set('grid/repetitions', 1)
daq_module.set('grid/rowrepetition', 0)
daq_module.set('grid/direction', 0)
daq_module.set('grid/waterfall', 0)
daq_module.set('grid/overwrite', 0)
daq_module.set('fft/window', 1)
daq_module.set('refreshrate', 5)
daq_module.set('awgcontrol', 0)
daq_module.set('historylength', 500)
daq_module.set('bandwidth', 0)
daq_module.set('hysteresis', 0.01)
daq_module.set('level', 0.1)
daq_module.set('triggernode', '/dev3994/demods/0/sample.TrigIn2')
daq_module.set('save/directory', 'C:\\Users\\vpixx\\Documents\\Zurich Instruments\\LabOne\\WebServer')
daq_module.set('clearhistory', 1)
daq_module.set('clearhistory', 1)
daq_module.set('bandwidth', 0)
daq_module.set('clearhistory', 1)
###############################################################################
#%% Spectrum
# daq_module = daq.dataAcquisitionModule()
# daq_module.set('grid/mode', 4)
# daq_module.set('type', 0)
# daq_module.set('preview', 1)
# daq_module.set('grid/rows', 10)
# daq_module.set('grid/rowrepetition', 1)
# daq_module.set('grid/waterfall', 1)
# daq_module.set('grid/overwrite', 1)
# daq_module.set('device', 'dev3994')
# daq_module.set('historylength', 10)
# daq_module.set('delay', 0)
# daq_module.set('spectrum/enable', 1)
# daq_module.set('grid/cols', 8192)
# daq_module.set('grid/mode', 4)
# daq_module.set('grid/cols', 1024)
# daq_module.set('grid/rows', 1)
# daq_module.set('grid/repetitions', 1)
# daq_module.set('grid/direction', 0)
# daq_module.set('grid/waterfall', 0)
# daq_module.set('grid/overwrite', 0)
# daq_module.set('fft/window', 1)
# daq_module.set('refreshrate', 5)
# daq_module.set('spectrum/overlapped', 1)
# daq_module.set('spectrum/frequencyspan', 837.053589)
# daq_module.set('historylength', 30)
# daq_module.set('clearhistory', 1)
# daq_module.set('save/directory', 'C:\\Users\\vpixx\\Documents\\Zurich Instruments\\LabOne\\WebServer')
###############################################################################
#%% PiD 

pid = daq.pidAdvisor()
pid.set('device', 'dev3994')
pid.set('index', 0)
pid.execute()
# To read the acquired data from the module, use a
# while loop like the one below. This will allow the
# data to be plotted while the measurement is ongoing.
# Note that any device nodes that enable the streaming
# of data to be acquired, must be set before the while loop.
# result = 0
# while not pid.finished():
#     time.sleep(1)
#     result = pid.read()
#     print(f"Progress {float(pid.progress()) * 100:.2f} %\r")
pid.set('response', 1)
# Starting module pidAdvisor on 2023/10/16 12:24:47
pid = daq.pidAdvisor()
pid.set('device', 'dev3994')
pid.set('index', 1)
pid.execute()
# To read the acquired data from the module, use a
# while loop like the one below. This will allow the
# data to be plotted while the measurement is ongoing.
# Note that any device nodes that enable the streaming
# of data to be acquired, must be set before the while loop.
# result = 0
# while not pid.finished():
#     time.sleep(1)
#     result = pid.read()
#     print(f"Progress {float(pid.progress()) * 100:.2f} %\r")
pid.set('response', 1)
# Starting module pidAdvisor on 2023/10/16 12:24:47
pid = daq.pidAdvisor()
pid.set('device', 'dev3994')
pid.set('index', 2)
pid.execute()
# To read the acquired data from the module, use a
# while loop like the one below. This will allow the
# data to be plotted while the measurement is ongoing.
# Note that any device nodes that enable the streaming
# of data to be acquired, must be set before the while loop.
# result = 0
# while not pid.finished():
#     time.sleep(1)
#     result = pid.read()
#     print(f"Progress {float(pid.progress()) * 100:.2f} %\r")
pid.set('response', 1)
# Starting module pidAdvisor on 2023/10/16 12:24:47
pid = daq.pidAdvisor()
pid.set('device', 'dev3994')
pid.set('index', 3)
pid.execute()
# To read the acquired data from the module, use a
# while loop like the one below. This will allow the
# data to be plotted while the measurement is ongoing.
# Note that any device nodes that enable the streaming
# of data to be acquired, must be set before the while loop.
# result = 0
# while not pid.finished():
#     time.sleep(1)
#     result = pid.read()
#     print(f"Progress {float(pid.progress()) * 100:.2f} %\r")
pid.set('response', 1)
pid.set('pid/targetbw', 1000)
pid.set('pid/mode', 3)
pid.set('pid/autobw', 1)
pid.set('demod/order', 4)
pid.set('demod/timeconstant', 1.38458257e-05)
pid.set('demod/harmonic', 1)
pid.set('advancedmode', 0)
pid.set('display/freqstart', 2.82370225)
pid.set('display/freqstop', 282370.225)
pid.set('display/timestart', 0)
pid.set('display/timestop', 0.00248335896)
pid.set('tf/input', 1)
pid.set('tf/output', 1)
pid.set('tf/closedloop', 1)
pid.set('pid/autolimit', 1)
pid.set('dut/source', 4)
pid.set('dut/delay', 0)
pid.set('dut/gain', 1)
pid.set('dut/bw', 1000)
pid.set('dut/fcenter', 1591.68414)
pid.set('dut/damping', 0.707)
pid.set('dut/q', 10)
pid.set('pid/rate', 8371)
pid.set('tuner/mode', 3)
pid.set('tuner/averagetime', 0.05)
pid.set('pid/targetbw', 10000)
pid.set('pid/mode', 3)
pid.set('pid/autobw', 1)
pid.set('demod/order', 4)
pid.set('demod/timeconstant', 0.0001)
pid.set('demod/harmonic', 1)
pid.set('advancedmode', 0)
pid.set('display/freqstart', 50)
pid.set('display/freqstop', 5000000)
pid.set('display/timestart', 0)
pid.set('display/timestop', 0.000138309545)
pid.set('tf/input', 1)
pid.set('tf/output', 1)
pid.set('tf/closedloop', 1)
pid.set('pid/autolimit', 1)
pid.set('dut/source', 0)
pid.set('dut/delay', 0)
pid.set('dut/gain', 1)
pid.set('dut/bw', 1000)
pid.set('dut/fcenter', 100000)
pid.set('dut/damping', 0.707)
pid.set('dut/q', 10)
pid.set('pid/rate', 14060000)
pid.set('tuner/mode', 3)
pid.set('tuner/averagetime', 0.05)
pid.set('pid/targetbw', 10000)
pid.set('pid/mode', 3)
pid.set('pid/autobw', 1)
pid.set('demod/order', 4)
pid.set('demod/timeconstant', 0.0001)
pid.set('demod/harmonic', 1)
pid.set('advancedmode', 0)
pid.set('display/freqstart', 1.81375221)
pid.set('display/freqstop', 181375.221)
pid.set('display/timestart', 0)
pid.set('display/timestop', 0.00179096237)
pid.set('tf/input', 1)
pid.set('tf/output', 1)
pid.set('tf/closedloop', 1)
pid.set('pid/autolimit', 1)
pid.set('dut/source', 0)
pid.set('dut/delay', 0)
pid.set('dut/gain', 1)
pid.set('dut/bw', 1000)
pid.set('dut/fcenter', 100000)
pid.set('dut/damping', 0.707)
pid.set('dut/q', 10)
pid.set('pid/rate', 14060000)
pid.set('tuner/mode', 3)
pid.set('tuner/averagetime', 0.05)
pid.set('pid/targetbw', 10000)
pid.set('pid/mode', 3)
pid.set('pid/autobw', 1)
pid.set('demod/order', 4)
pid.set('demod/timeconstant', 0.0001)
pid.set('demod/harmonic', 1)
pid.set('advancedmode', 0)
pid.set('display/freqstart', 1.81375221)
pid.set('display/freqstop', 181375.221)
pid.set('display/timestart', 0)
pid.set('display/timestop', 0.00179096237)
pid.set('tf/input', 1)
pid.set('tf/output', 1)
pid.set('tf/closedloop', 1)
pid.set('pid/autolimit', 1)
pid.set('dut/source', 0)
pid.set('dut/delay', 0)
pid.set('dut/gain', 1)
pid.set('dut/bw', 1000)
pid.set('dut/fcenter', 100000)
pid.set('dut/damping', 0.707)
pid.set('dut/q', 10)
pid.set('pid/rate', 14060000)
pid.set('tuner/mode', 3)
pid.set('tuner/averagetime', 0.05)
daq.setInt('/dev3994/pids/2/outputchannel', 0)
daq.setInt('/dev3994/pids/3/outputchannel', 0)
pid.set('pid/p', -25)
pid.set('pid/i', -350)
pid.set('pid/p', 1)
pid.set('pid/i', 100000)
pid.set('pid/p', 1)
pid.set('pid/i', 100000)
pid.set('pid/p', 1)
pid.set('pid/i', 100000)
pid.set('dut/fcenter', 1476.01546)
###############################################################################
#%% Sweeper
sweeper = daq.sweep()
sweeper.set('device', 'dev3994')
sweeper.set('historylength', 100)
sweeper.set('start', 1313.58828)
sweeper.set('stop', 2010.24103)
sweeper.set('scan', 0)
sweeper.set('xmapping', 0)
sweeper.set('samplecount', 150)
sweeper.set('loopcount', 1)
sweeper.set('gridnode', '/dev3994/oscs/0/freq')
sweeper.set('bandwidth', 1000)
sweeper.set('order', 4)
sweeper.set('settling/inaccuracy', 0.0001)
sweeper.set('settling/time', 0)
sweeper.set('averaging/tc', 0)
sweeper.set('averaging/sample', 1)
sweeper.set('averaging/time', 0)
sweeper.set('filtermode', 1)
sweeper.set('maxbandwidth', 1250000)
sweeper.set('bandwidthoverlap', 0)
sweeper.set('omegasuppression', 40)
sweeper.set('phaseunwrap', 0)
sweeper.set('sincfilter', 0)
sweeper.set('awgcontrol', 0)
sweeper.set('historylength', 100)
sweeper.set('bandwidthcontrol', 0)
sweeper.set('save/directory', 'C:\\Users\\vpixx\\Documents\\Zurich Instruments\\LabOne\\WebServer')
###############################################################################