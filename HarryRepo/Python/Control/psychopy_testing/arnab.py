# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard
# Run 'Before Experiment' code from VPIXX_init
from pypixxlib import _libdpx as dp
from pypixxlib import responsepixx as rp
import csv
import numpy as np

# Initialize the PsychoPy window
win = visual.Window([800, 600], fullscr=False, monitor="testMonitor", units="deg")

# Initialize DataPixx
print("Opening DataPixx device...")
dp.DPxOpen()
isReady = dp.DPxIsReady()
if not isReady:
    raise ConnectionError('VPixx Hardware not detected! Check your connection and try again.')

dp.DPxStopAllScheds()
dp.DPxWriteRegCache()  


print("Setting all TTL digital outputs to low...")



stimulus = visual.Circle(win, radius=0.5, fillColor='red', lineColor='red')

n_stimuli = 10  # Number of stimuli
stimulus_duration = 1  # Duration of each stimulus in seconds

# Start the experiment
print('\nExperiment starting...')
timeOn = dp.DPxGetTime()

for trial in range(n_stimuli):
    print(f"Starting trial {trial+1}/{n_stimuli}...")
   
    # Send TTL trigger at the start of each trial
    doutValue = int('000000000000000011010000', 2)
    bitMask = 0xffffff
    dp.DPxSetDoutValue(doutValue, bitMask)
    dp.DPxWriteRegCache()
   
    doutValue = int('000000000000000000000000', 2)
    bitMask = 0xffffff
    dp.DPxSetDoutValue(doutValue, bitMask)
    dp.DPxWriteRegCache()
    # Draw and present stimulus
    stimulus.draw()
    win.flip()
   
    core.wait(stimulus_duration)

# End of experiment
print('\nExperiment completed. Press any key to exit.')
event.waitKeys()

# Cleanup
print("Closing DataPixx device...")
dp.DPxWriteRegCache()
dp.DPxClose()
win.close()
core.quit()



	
