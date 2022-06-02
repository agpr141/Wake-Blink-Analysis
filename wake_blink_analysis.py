"""
Input script for analysing wake blink eye movements from EEG&EOG data
author: @agpr141
date completed: 02.06.22
"""

# import modules
import os
from blink_automatic_detection_funcs import *

# 1. set path for 'Wake Eyes Open' .edf file & enter participant details
pptid = '1102'
night = '2'
group = 'HC'
os.chdir('Z:/Data/HC/1102/H1/EEG/Night 2/EDF/Eyes Open')  # change directory to folder of interest
weo = 'N2weo.edf'  # define your .edf file

# analyse
eyeblink_stats = blink_analyse(weo, pptid, night, group)
