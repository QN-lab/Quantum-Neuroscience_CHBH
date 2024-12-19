from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout

import psychopy.iohub as io
from psychopy.hardware import keyboard
# Run 'Before Experiment' code from VPIXX_init
from pypixxlib import _libdpx as dp
from pypixxlib import responsepixx as rp
import csv
import numpy as np

# Initialize DataPixx
print("Opening DataPixx device...")
dp.DPxOpen()
isReady = dp.DPxIsReady()
if not isReady:
    raise ConnectionError('VPixx Hardware not detected! Check your connection and try again.')

dp.DPxStopAllScheds()
dp.DPxWriteRegCache()

# Parameters
n_blocks = 1  # Number of blocks
n_trials_per_block = 15  # Trials per block
fixation_duration = 2  # Duration of fixation cross (in seconds)
circle_duration = 1  # Duration of red circle (in seconds)

print('Expected time:' + str(n_blocks*n_trials_per_block*(fixation_duration+circle_duration)/60) + 'minutes')

# Create a window
win = visual.Window(size=(800, 600), color=(0, 0, 0), units="pix")

# Define stimuli
fixation = visual.TextStim(win, text="+", height=40, color="white")
circle = visual.Circle(win, radius=50, fillColor="red", lineColor="red")

continuous_task_text = visual.TextStim(win, text=(
    "Welcome to the experiment!\n\n"
    "We will start with a quick maximal tension test.\n\n"
    "For the next 10 seconds, you will see a fixation cross.\n\n"
    "Please tense your index and thumb together as hard as you can, within reason.\n\n"
    "Press any key to start."
), height=30, color="white")

intro_text = visual.TextStim(win, text=(
    "Thanks! Now for the main experiment\n\n"
    "Fixate on the cross.\n\n"
    "When the circle appears, LIGHTLY tense your index and thumb together until the circle dissapears.\n\n"
    "Press any key to continue."
), height=30, color="white")

block_text = visual.TextStim(win, text="", height=30, color="white")

end_text = visual.TextStim(win, text=(
    "Thank you for participating!\n\n"
    "Press any key to exit."
), height=30, color="white")

# Function to display text and wait for key press
def show_text(win, text_stim):
    text_stim.draw()
    win.flip()
    event.waitKeys()

#TTL function
def deliver_TTL1():
    # Send TTL trigger
    doutValue = int('000000000000000011010000', 2)
    bitMask = 0xffffff
    dp.DPxSetDoutValue(doutValue, bitMask)
    dp.DPxWriteRegCache()
   
    doutValue = int('000000000000000000000000', 2)
    bitMask = 0xffffff
    dp.DPxSetDoutValue(doutValue, bitMask)
    dp.DPxWriteRegCache()
    return
    
def deliver_TTL2():
    # Send TTL trigger
    doutValue = int('000000000000000001000000', 2)
    bitMask = 0xffffff
    dp.DPxSetDoutValue(doutValue, bitMask)
    dp.DPxWriteRegCache()
   
    doutValue = int('000000000000000000000000', 2)
    bitMask = 0xffffff
    dp.DPxSetDoutValue(doutValue, bitMask)
    dp.DPxWriteRegCache()
    return


# Continuous task block
show_text(win, continuous_task_text)  # Display continuous task instructions
fixation.draw()  # Display fixation cross for 10 seconds
win.flip()
core.wait(10)

# Show introductory text (moved here)
show_text(win, intro_text)

# Main experiment blocks
for block_num in range(1, n_blocks + 1):
    # Display block start text
    block_text.text = f"Block {block_num} of {n_blocks}\n\n Press any key to continue."
    show_text(win, block_text)
    
    
    # Loop through trials
    for trial_num in range(1, n_trials_per_block + 1):
        
        trigger_sent = False
        deliver_TTL1()
        print("BANG")  # Debug output to simulate trigger
        trigger_sent = True
        
        # Fixation cross
        fixation.draw()
        win.flip()
        core.wait(fixation_duration)
       
        trigger_sent= False
        deliver_TTL2()
        print("BANG")  # Debug output to simulate trigger
        trigger_sent = True
        
        # Red circle
        circle.draw()
        win.flip()
        core.wait(circle_duration)

# Display ending message
show_text(win, end_text)

# Close the window
win.close()
core.quit()
