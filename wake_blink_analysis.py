"""
Input script for analysing wake blink eye movements from EEG&EOG data
author: @agpr141
date completed: 02.06.22
"""

# import modules
import os
from blink_automatic_detection_funcs import *

# 1. set path for 'Wake Eyes Open' .edf file & enter participant details
pptid = '1119'
night = '2'
group = 'HC'
os.chdir('Z:/Data/HC/1119/H1/EEG/Night 2/EDF/Eyes Open')  # change directory to folder of interest
weo = '1119^weo.edf'  # define your .edf file
if not os.path.exists('EOG-complete'):
    os.makedirs('EOG-complete')

# analyse
eyeblink_stats = blink_analyse(weo, pptid, night, group)
