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

def funcStreamEmg(sharedConfig, sharedData, sharedQueue1, sharedQueue2, verbose=False, debug=False):
    
    # Parameters
    samples_per_read = sharedConfig['samples_per_read']
    
    ###########################################################################
    # dummy mode
    ###########################################################################

    if sharedConfig['emg_dummyMode']:

        ###
        # just generate a 300 values sinwave
        ###

        def genWave():
            cycles        = 1   # how many sine cycles
            resolution    = 300 # how many datapoints to generate
            length        = np.pi * 3 * cycles
            wave          = np.sin(np.arange(0, length, length / resolution))
            wave          = wave/2000 # want it lower to fit the graph nicely
    
            # make first muscle the sinwave followed by a flat line
            #this_emg_delt = np.concatenate( (wave, np.zeros(150)) )
            # have a 25% offset for the sinwave on second muscle to differenciate on plot
            #this_emg_bic  = np.concatenate( (np.zeros(75), wave) )
            #this_emg_bic= np.concatenate( (this_emg_bic, np.zeros(75)) )
            
            return wave, wave
        
        # open files on disk
        fileRaw_delt = open(sharedConfig['dataSaveFolder']+'/data_raw_delt.txt', 'ab')
        fileRaw_bic  = open(sharedConfig['dataSaveFolder']+'/data_raw_bic.txt',  'ab')

        fakeIndex     = 20 # first batch detected at index 20
        fakeFrequency = 80 # every 80 values
        
        if verbose: print(f"fakeIndex init is {fakeIndex}")
        if verbose: print(f"fakeFrequency is {fakeFrequency}")

        ###
        # Loop with a delay
        ###
        
        iEmg = 0
        while not sharedConfig['emg_shutdown']:

            # can induce artificial delay
            sleep(0.5)

            ###
            # get the data
            ###

            # we dummy always have a fresh sin wave in this_emg_delt and this_emg_bic
            this_emg_delt, this_emg_bic = genWave()
            this_emg_bic = np.array([0.0006]*300)

            if iEmg==0: # as we receive the first batch, get the time to provide feedback on how long we have been streaming
                sharedConfig['streamTimeStart'] = time.time()
                if verbose: print(f"First batch, setting EMG start time at {time.time()}")

            # can fake a stimulation artifact at a specific value (on 2 values to have a nicer line +-)
            stimValueDummy = 0.0011

            # just one or two random spikes per batch to visualize them
            """
            #this_emg_delt[150]  = +stimValueDummy
            #this_emg_delt[151]  = -stimValueDummy
            """

            # if it's every 80 values (so varies from batch to batch)
            # init first batch at 20
            listIndexes = []
            for i in range(5):
                listIndexes.append(fakeIndex + fakeFrequency*i)
            listIndexes2 = [x for x in listIndexes if x < 300]
            printColor(f"\n# emg iter {iEmg}: peaks on indexes {listIndexes2}")

            for this_index in listIndexes2:
                this_emg_delt[this_index] = stimValueDummy
            #print(f"so next index should be at {(listIndexes[-1]+80)-300}")
            fakeIndex = (listIndexes2[-1]+fakeFrequency)-300

            # add a peak at 0 to delimitate frames
            #this_emg_delt[0] = 1
            #print(f"this batch of len {len(this_emg_delt)} has {len(listIndexes2)} peaks")

            ###
            # dump in csv
            ###
            
            np.savetxt(fileRaw_delt, this_emg_delt, delimiter=',')
            np.savetxt(fileRaw_bic,  this_emg_bic,  delimiter=',')
          
            ###
            # Put in shared dict
            ###
            
            # add new batch to current array
            sharedData['emg_delt_raw'] = np.concatenate( (sharedData['emg_delt_raw'],  this_emg_delt),  axis=0 )
            sharedData['emg_bic_raw']  = np.concatenate( (sharedData['emg_bic_raw'],   this_emg_bic),   axis=0 )
            
            # remove the first part to keep size consistent (if reached max limit)
            if len(sharedData['emg_delt_raw']) > sharedConfig['maxArraySize']:
                # we remove length of frame from beginning of array
                sharedData['emg_delt_raw'] = sharedData['emg_delt_raw'][samples_per_read:]
                sharedData['emg_bic_raw']  = sharedData['emg_bic_raw'][samples_per_read:]

            ###
            # only then (after get;dump;resize), increment interNumber to let the other processes know they can process the next batch
            ###
            
            iEmg+=1
            sharedConfig['emg_iter'] = iEmg
    
            if verbose: print(f"iter {iEmg}, q1 = {sharedQueue1.qsize()}, q2 = {sharedQueue2.qsize()}")
            
 
        fileRaw_delt.close()
        fileRaw_bic.close()
        print("emg dummy: goodbye")
        return

    ###########################################################################
    # Normal mode (not dummy)
    ###########################################################################

    ###
    # Try connecting in a loop until success
    ###

    showTitle("Connecting to EMG", 'cyan')
    
    connectedEMG = False
    devEMG       = None
  
    i=1
    while not connectedEMG:
    
        if not sharedConfig['emg_shutdown']:
            
            print(f"status of emg_shutdown is {sharedConfig['emg_shutdown']}")

            printColor("connecting to EMG try {}".format(i), color='white')       
            try:
                devEMG       = delsys.TrignoEMG(channel_range=(0, 16), samples_per_read=samples_per_read, host='127.0.0.1', timeout=10)
                connectedEMG = True

            except:
                printColor('Unable to connect to TCU, make sure it is running', color='red')
                i+=1
            time.sleep(3)

    if verbose: printColor('Connected to EMG', color='green')
    devEMG.become_master()
    devEMG.start()

    # Notify the other processes that we have succeded
    sharedConfig['emg_connected'] = True
    print("Connected to EMG")
    
    ###
    # Get the data in a loop
    ###
    
    # open files on disk
    fileRaw_delt = open(sharedConfig['dataSaveFolder']+'/data_raw_delt.txt', 'ab')
    fileRaw_bic  = open(sharedConfig['dataSaveFolder']+'/data_raw_bic.txt',  'ab')    

    iEmg = 0
    while not sharedConfig['emg_shutdown']:
   
        if verbose: print("# emg: getting EMG from delsys")

        ###
        # Get data from Delsys
        ###
        
        # blocking mechanism, so we read() until our local buffer has gotten enough frames from delsys to constitute a full batch
        data_frame    = devEMG.read()
        this_emg_delt = data_frame[0]
        this_emg_bic  = data_frame[1]

        if iEmg==0: # as we receive the first batch, get the time to provide feedback on how long we have been streaming
            sharedConfig['streamTimeStart'] = time.time()

        ###
        # dump in csv
        ###
        
        np.savetxt(fileRaw_delt, this_emg_delt, delimiter=',')
        np.savetxt(fileRaw_bic,  this_emg_bic,  delimiter=',')
        
        ###
        # Put in shared dict
        ###
        
        # add new batch to current array
        sharedData['emg_delt_raw'] = np.concatenate( (sharedData['emg_delt_raw'],  this_emg_delt),  axis=0 )
        sharedData['emg_bic_raw']  = np.concatenate( (sharedData['emg_bic_raw'],   this_emg_bic),   axis=0 )
        
        # remove the first part to keep size consistent (if reached max limit)
        if len(sharedData['emg_delt_raw']) > sharedConfig['maxArraySize']:
            # we remove length of frame from beginning of array
            sharedData['emg_delt_raw'] = sharedData['emg_delt_raw'][samples_per_read:]
            sharedData['emg_bic_raw']  = sharedData['emg_bic_raw'][samples_per_read:]

        ###
        # Put in queue: discarded
        ###
        #sharedQueue1.put(this_emg_delt)
        #sharedQueue2.put(this_emg_bic)
        #if verbose: print(f"iter {iEmg}, q1 = {sharedQueue1.qsize()}, q2 = {sharedQueue2.qsize()}")

        ###
        # only then (after get;dump;resize), increment iterNumber to let the filter process know it can process the next batch
        ###
        
        iEmg+=1
        sharedConfig['emg_iter'] = iEmg
            
    print("emg thread: Stopping EMG gracefully")
    
    fileRaw_delt.close()
    fileRaw_bic.close()
    devEMG.stop()
    devEMG.__del__()

if __name__ == "__main__":

    print("orlau_live_emg loaded as main")
