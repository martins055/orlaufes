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

import multiprocessing
from multiprocessing import Process, Pipe, freeze_support, Manager

from orlau_utils import showTitle, printColor, show

def funcFilter(sharedConfig, sharedData, sharedQueue1, sharedQueue2, verbose=False, debug=False):
    
    # using sharedDict and not queues

    # open files on disk
    fileFilter_delt = open(sharedConfig['dataSaveFolder']+'/data_filter_delt.txt', 'ab')
    fileFilter_bic  = open(sharedConfig['dataSaveFolder']+'/data_filter_bic.txt',  'ab')

    samples_per_read = sharedConfig['samples_per_read']
    
    iFilter = 0
    while not sharedConfig['emg_shutdown']:
              
        #induce delay
        #sleep(0.4)

        if debug: print(f"\n# filt: current raw is {len(sharedData['emg_delt_raw'])}, current filt is {len(sharedData['emg_delt_filt'])}")

        # Pass if we don't have any new data from raw (could change to blocking mechanism)
        if not sharedConfig['emg_iter'] > iFilter:
            if debug: print(f"# filt: no new data from raw, skipping")
            pass

        # if the array has been populated, process the latest batch of data
        if sharedConfig['emg_iter'] > iFilter :

            if verbose: printColor(f"\nFilter iter {iFilter}", 'green')
            
            # create local vars just in case a flag changes through the function, we want to finish the loop
            this_emg_filter_do = sharedConfig['emg_filter_do']
            this_stim_do       = sharedConfig['stim_do']
            
            ###
            # get the latest filtered data
            ###
            
            # latest batch of raw data
            this_emg_delt_raw = sharedData['emg_delt_raw'][-samples_per_read:]
            this_emg_bic_raw  = sharedData['emg_bic_raw'][-samples_per_read:]
            
            # getting a copy of the array of the past filtered data
            # we can't edit a nested value in the manager's shared dict directly, so will need to replace it
            # not a deep copy, and a numpy array, so should not impact performance
            copy_delt_filt =  sharedData['emg_delt_filt']
            copy_bic_filt  =  sharedData['emg_bic_filt']
            this_emg_delt_filt = None
            this_emg_bic_filt  = None

            ###
            # and do the filtering (if asked)
            ###

            # if not asked, just pass the raw data through (for debugging/assessment only)
            # now directly in RMS that chooses who to take from
            #if not this_emg_filter_do:
            #    this_emg_delt_filt = sharedData['emg_delt_raw'][-samples_per_read:]
            #    this_emg_bic_filt  = sharedData['emg_bic_raw'][-samples_per_read:]
            #    pass

            ##
            # Option 1 : simpliest, for variable frequency noise

            # we take what has been filtered so far, and we add the new batch of unfiltered data, and we will loop through it to filter
            # this is because if one of the first frames (within the filter_delWindow) is above the threshold, we need the previous good value which only exists in the previous batch of data           
            copy_delt_filt =  np.concatenate( (copy_delt_filt, this_emg_delt_raw),  axis=0 )
            copy_bic_filt  =  np.concatenate( (copy_bic_filt,  this_emg_bic_raw),   axis=0 )

            # Delt
            if verbose: print("# emg: applying filter to delt")
            # Loop through the latest batch of data DELT
            ifilter1 = -samples_per_read
            
            # we loop through the last batch of data
            for index in range(ifilter1,0):
                
                this_value = copy_delt_filt[index]

                # if we have a value that is above the threshold            
                if abs(this_value) >= sharedConfig['stim_thresh']:
                    
                    if verbose: print("FILT : found threshold value in DELT")
                    
                    latestGoodKnownValue = copy_delt_filt[index - sharedConfig['filter_delWindow'] - 1]
                    
                    # we change the values before
                    copy_delt_filt[index - sharedConfig['filter_delWindow'] : index] = latestGoodKnownValue

                    # we change this value
                    copy_delt_filt[index] = latestGoodKnownValue
                    
                    # we change the values after (special case if there's less than the window remaining in the array)
                    if index < -sharedConfig['filter_delWindow']:
                        copy_delt_filt[index : index + sharedConfig['filter_delWindow']] = latestGoodKnownValue
                    elif index >= -sharedConfig['filter_delWindow']:
                        copy_delt_filt[index : ] = latestGoodKnownValue


            # Bic
            if verbose: print("# emg: applying filter to bic")
            # Loop through the latest batch of data BIC
            ifilter2 = -samples_per_read
            
            # we loop through the last batch of data
            for index in range(ifilter2,0):
                
                this_value = copy_bic_filt[index]

                # if we have a value that is above the threshold            
                if abs(this_value) >= sharedConfig['stim_thresh']:
                    
                    if verbose: print("FILT : found threshold value in DELT")
                                        
                    # we change the values before
                    copy_bic_filt[index - sharedConfig['filter_delWindow'] : index] = 0

                    # we change this value
                    copy_bic_filt[index] = 0
                    
                    # we change the values after (special case if there's less than the window remaining in the array)
                    if index < -sharedConfig['filter_delWindow']:
                        copy_bic_filt[index : index + sharedConfig['filter_delWindow']] = 0
                    elif index >= -sharedConfig['filter_delWindow']:
                        copy_bic_filt[index : ] = 0


            # Replace the shared array by the one we have just processed
            this_emg_delt_filt = copy_delt_filt[-samples_per_read:]
            this_emg_bic_filt  = copy_bic_filt[-samples_per_read:]
            
            ##
            # Debug/testing: divide by two
            #this_emg_delt_filt = sharedData['emg_delt_raw'][-samples_per_read:] /2
            #this_emg_bic_filt  = sharedData['emg_bic_raw'][-samples_per_read:]  /2
            
            ###
            # dump filtered data in csv
            ###

            np.savetxt(fileFilter_delt, this_emg_delt_filt, delimiter=',')
            np.savetxt(fileFilter_bic,  this_emg_bic_filt,  delimiter=',')

            ###
            # Put in shared dict
            ###

            # add to our shared dict
            sharedData['emg_delt_filt'] = np.concatenate( (sharedData['emg_delt_filt'],  this_emg_delt_filt),  axis=0 )
            sharedData['emg_bic_filt']  = np.concatenate( (sharedData['emg_bic_filt'],   this_emg_bic_filt),   axis=0 )
            if verbose: print(f"now filt is {len(sharedData['emg_delt_filt'])}")
            
            # remove the first part to keep size consistent (if reached max limit)
            if len(sharedData['emg_delt_filt']) > sharedConfig['maxArraySize']:
                # we remove length of frame from beginning of array
                sharedData['emg_delt_filt'] = sharedData['emg_delt_filt'][samples_per_read:]
                sharedData['emg_bic_filt']  = sharedData['emg_bic_filt'][samples_per_read:]

            ###
            # only then (after get;dump;resize), increment iterNumber to let rms processe know it can process the next batch
            ###

            iFilter+=1
            sharedConfig['filter_iter'] = iFilter
            if verbose: print(f"# filt: new data, done iter {iFilter}")    

    # Graceful exit
    fileFilter_delt.close()
    fileFilter_bic.close()

if __name__ == "__main__":
    
    print("orlau_filter loaded as main")
