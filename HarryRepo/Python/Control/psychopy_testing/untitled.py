from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout

import psychopy.iohub as io
from psychopy.hardware import keyboard
# Run 'Before Experiment' code from VPIXX_init
from pypixxlib import _libdpx as dp
from pypixxlib import responsepixx as rp
import csv
import numpy as np

# Set up the window
win = visual.Window(size=(800, 600), color=(1, 1, 1), units='pix')

# Initialize DataPixx
print("Opening DataPixx device...")
dp.DPxOpen()
isReady = dp.DPxIsReady()
if not isReady:
    raise ConnectionError('VPixx Hardware not detected! Check your connection and try again.')

dp.DPxStopAllScheds()
dp.DPxWriteRegCache()  

print("Setting all TTL digital outputs to low...")

################################################################################
############################## Create Slides####################################
intro_text = """
Welcome to the Experiment!

You will see a dot on the screen.
Fixate your eyes on this dot.

When the dot turns red, the trial is over. 
Please press any key to continue to the next trial.

Please try to remain still and blink only during the 'red' dot period.
If you feel uncomfortable at any point, press the 'p' button to pause the experiment'

There will be N blocks, each with M trials.

Press any key to begin.
"""
intro_slide = visual.TextStim(win, text=intro_text, color='black', height=30, wrapWidth=700)

# Create pause message
pause_text = """
Experiment Paused.

Press 'r' to resume.
Press 'q' to quit the experiment.
"""
pause_message = visual.TextStim(win, text=pause_text, color='black', height=30, wrapWidth=700)

#Block end text
def block_text_f(b_val,total_b_number):
    block_text= f"""
    Block {b_val} of {total_b_number} done
    
    Relax and press any button to continue
    """
    block_message = visual.TextStim(win, text=block_text, color='black', height=30, wrapWidth=700)
    block_message.draw()
    win.flip()
    return



# Create outro message
outro_text = """
Experiment Over! 

Thanks for your help on this study!
Relax and I will come and grab you. 

"""
outro_message = visual.TextStim(win, text=outro_text, color='black', height=30, wrapWidth=700)

################################################################################
############################## CRun Experiment #################################
# Display the intro slide
intro_slide.draw()
win.flip()

# Wait for the participant to press any key to start the experiment
event.waitKeys()

# Create the dot stimulus
dot = visual.Circle(win, radius=10, fillColor='black', lineColor='black')

# Set the duration for displaying the dot
dot_duration = 3.0  # 3000ms
trigger_time = 1.5  # 1500ms

#TTL function
def deliver_TTL():
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


# Number of trials
n_blocks = 3
n_trials = 4

# Trial loop
for block in range(n_blocks):
    for trial in range(n_trials):
        print(f"Starting trial {trial + 1}")
    
        # Clock to keep track of time
        clock = core.Clock()
        trigger_sent = False

        # Start the trial
        clock.reset()
        while clock.getTime() < dot_duration:
            # Draw the dot
            dot.draw()
            win.flip()
            
            # Check for pause key
            keys = event.getKeys()
            if 'p' in keys:
                # Display the pause message
                pause_message.draw()
                win.flip()
                
                # Wait for 'r' to resume or 'q' to quit
                while True:
                    pause_keys = event.waitKeys(keyList=['r', 'q'])
                    if 'r' in pause_keys:
                        print("Resuming experiment...")
                        break
                    elif 'q' in pause_keys:
                        print("Quitting experiment...")
                        win.close()
                        core.quit()

            # Send the trigger at 1500ms
            if not trigger_sent and clock.getTime() >= trigger_time:
                # Send trigger
                deliver_TTL()
                print("BANG")  # Debug output to simulate trigger
                trigger_sent = True

        # After 3000ms, change the dot to red
        dot.fillColor = 'red'
        dot.lineColor = 'red'
        dot.draw()
        win.flip()

        # Wait for participant input to continue to the next trial
        keys = event.waitKeys()
        print(f"Response recorded for trial {trial + 1}")
        
        # Reset the dot color for the next trial
        dot.fillColor = 'black'
        dot.lineColor = 'black'
    block_text_f(block+1,n_blocks)
    keys = event.waitKeys()


#End Screen
outro_message.draw()
win.flip()

# Cleanup
keys = event.waitKeys()
win.close()
core.quit()
