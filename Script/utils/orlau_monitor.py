#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter
"""

# Reset the vars at run if using ipython-based IDE if in main
if __name__ == "__main__":
    try:
        from IPython import get_ipython
        get_ipython().run_line_magic('reset', '-sf')
    except: pass

import sys, os, copy, time
CURR_DIR = os.path.dirname(os.path.realpath(__file__))     # get current directory variable
os.chdir(os.path.abspath(os.path.dirname(__file__)))       # change current working directory
sys.path.append(CURR_DIR)                                  # load from the utils directory

from datetime import datetime
from time import sleep
import numpy as np
from collections import OrderedDict

from pprint import pprint
import beautifultable

import multiprocessing
from multiprocessing import Process, Pipe, freeze_support, Manager

from orlau_utils import showTitle, printColor, show

def funcMonitor(sharedConfig, sharedData):
        
    while not sharedConfig['emg_shutdown']:
    
        table = beautifultable.BeautifulTable(detect_numerics=False, maxwidth=120)
        
        refreshRate = 2
        sleep(refreshRate)

        printColor(f"\nMonitor: (refreshRate {refreshRate}s)", 'cyan')
        
        behind_iters   = sharedConfig['emg_iter']        - sharedConfig['rms_iter']        # how many iterations behind is the filter from the raw delsys
        behind_lengths = len(sharedData['emg_delt_raw']) - len(sharedData['emg_delt_rms']) # how many batches of data behind is the filter from the raw delsys

        streamTimeElapsed = 0
        if sharedConfig['streamTimeStart']:
            streamTimeElapsed = round( (time.time() - sharedConfig['streamTimeStart']), 2)


        this_dic = {
            'iterEmg'           : sharedConfig['emg_iter'],
            'iterFilter'        : sharedConfig['filter_iter'],
            'iterRms'           : sharedConfig['rms_iter'],
                        
            'len emg_delt_raw'  : len(sharedData['emg_delt_raw']),
            'len emg_delt_filt' : len(sharedData['emg_delt_filt']),
            'len emg_delt_rms'  : len(sharedData['emg_delt_rms']),
            
            'time_streaming'    : streamTimeElapsed, # len(sharedDict['emg_delt_raw'])/sharedConfig['rms_analogFreq'],
            'iter behind'       : behind_iters
            }
        
        #pprint(this_dic)

        i=1
        for key, value in this_dic.items():
            table.rows.append([key, value])
            i+=1

        if not sharedConfig['emg_shutdown']: print(table)
        
        """
        #print(f"shape emg_delt_raw  {sharedData['emg_delt_raw'].shape}")
        #print(f"shape emg_delt_filt {sharedData['emg_delt_filt'].shape}")
        #print(f"shape emg_delt_rms  {sharedData['emg_delt_rms'].shape}")
        
        print(f"iterEmg    = {sharedConfig['emg_iter']}")
        print(f"iterRms    = {sharedConfig['rms_iter']}")
        print(f"iterFilter = {sharedConfig['filter_iter']}")
        
        print(f"len emg_delt_raw = {len(sharedDict['emg_delt_raw'])}")
        print(f"len emg_delt_filt = {len(sharedDict['emg_delt_filt'])}")
        print(f"len emg_delt_rms = {len(sharedDict['emg_delt_rms'])}")

        print(f"iters behind {behind_iters}")
        """

if __name__ == "__main__":
    
    print("orlau_monitor loaded as main")
