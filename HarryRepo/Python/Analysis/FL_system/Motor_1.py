import mne
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import array
from datetime import datetime


directory_path = 'Y:/Harry_TMS/Motor'
files = os.listdir(directory_path)
file_name= '20241211_161104_sub-Pilot2_file-Motor1_raw.fif'
data_path=os.path.join(directory_path,file_name)

# Check if the specific file exists
file_exists = os.path.isfile(data_path)