#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

import multiprocessing
from multiprocessing import Process, Pipe, freeze_support, Manager

from orlau_utils import showTitle, printColor, show
import orlau_emg_utils as delsys

from termcolor import colored
import colorama
colorama.init()

def funcStreamImu(sharedConfig, sharedData, sharedQueue1, sharedQueue2, verbose=False, debug=False):
    
    # Parameters
    samples_per_read = sharedConfig['imu_samples_per_read']
    
    ###
    # Try connecting in a loop until success
    ###

    showTitle("Connecting to IMU", 'cyan')
    
    connectedIMU = False
    devIMU       = None
  
    i=1
    while not connectedIMU:
    
        if not sharedConfig['imu_shutdown']:
            
            print(f"status of imu_shutdown is {sharedConfig['imu_shutdown']}")

            printColor("connecting to IMU try {}".format(i), color='white')       
            try:
                devIMU       = delsys.TrignoAccel(channel_range=(0, 150), samples_per_read=samples_per_read, host='127.0.0.1', timeout=10)
                connectedIMU = True

            except:
                printColor('Unable to connect to TCU (IMU), make sure it is running', color='red')
                i+=1
            time.sleep(3)

    if verbose: printColor('Connected to IMU', color='green')
    devIMU.become_master()
    devIMU.start()

    # Notify the other processes that we have succeded
    sharedConfig['imu_connected'] = True
    print("Connected to IMU")
    
    ###
    # Get the data in a loop
    ###
    
    # open files on disk
    #fileImu_delt = open(sharedConfig['dataSaveFolder']+'/data_imu_delt.txt', 'ab')
    #fileImu_bic  = open(sharedConfig['dataSaveFolder']+'/data_imu_bic.txt',  'ab')
    fileImu_all  = open(sharedConfig['dataSaveFolder']+'/data_imu_all.txt',  'ab')

    iImu = 0
    while not sharedConfig['imu_shutdown']:
   
        if verbose: print("# IMU: getting IMU from delsys")

        # we want to keep sync with the batch (iter) number of the emg socket
        # while we don't have a sync, check if emg has started and save iter number of imu+emg in a tuple to do sync post-hoc
        if not sharedConfig['imu_iter_sync_emg']:
            tmp_emg_iter = sharedConfig['emg_iter']
            if tmp_emg_iter > 0:
                sharedConfig['imu_iter_sync_emg'] = (tmp_emg_iter, iImu)

        ###
        # Get data from Delsys
        ###
        
        # blocking mechanism, so we read() until our local buffer has gotten enough frames from delsys to constitute a full batch
        this_data_frame    = devIMU.read()
        #this_imu_delt = this_data_frame[27:31]
        #this_imu_bic  = this_data_frame[63:67]

        if iImu==0: # as we receive the first batch, get the time to provide feedback on how long we have been streaming
            sharedConfig['streamTimeStart'] = time.time()

        ###
        # dump in csv
        ###
        
        #np.savetxt(fileImu_delt, this_imu_delt, delimiter=',')
        #np.savetxt(fileImu_bic,  this_imu_bic,  delimiter=',')
        np.savetxt(fileImu_all, this_data_frame.T, delimiter=',')
        
        ###
        # Put in shared dict
        ###
        
        # as we have no preview, no need for it

        ###
        # only then (after get;dump;resize), increment iterNumber to let the filter process know it can process the next batch
        ###
        
        iImu+=1
        sharedConfig['imu_iter'] = iImu
            
    print("imu thread: Stopping IMU gracefully")
    
    fileImu_all.close()
    devIMU.stop()
    devIMU.__del__()

if __name__ == "__main__":

    print("orlau_live_imu loaded as main")

