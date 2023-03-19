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

from orlau_utils import showTitle, printColor, show, convert_to_rms, time_normalise
from orlau_controller import controller

def funcRms(sharedConfig, sharedData, sharedQueue1, sharedQueue2, verbose=False, debug=False):
    
    ###########################################################################
    # Normal mode (not dummy)
    ###########################################################################

    # open files on disk
    fileRms_delt         = open(sharedConfig['dataSaveFolder']+'/data_rms_delt.txt',             'ab')
    fileRms_bic          = open(sharedConfig['dataSaveFolder']+'/data_rms_bic.txt',              'ab')

    fileActiv_delt       = open(sharedConfig['dataSaveFolder']+'/data_activ_delt_history.txt',   'ab')
    fileActiv_bic        = open(sharedConfig['dataSaveFolder']+'/data_activ_bic_history.txt',    'ab')

    fileController_value = open(sharedConfig['dataSaveFolder']+'/data_controller_value.txt',     'ab')
    fileController_pulse = open(sharedConfig['dataSaveFolder']+'/data_controller_intensity.txt', 'ab')

    samples_per_read = sharedConfig['samples_per_read']

    ###
    # option1 : using sharedDict and not queues
    ###

    # Loop with a delay

    iRms = 0
    while not sharedConfig['emg_shutdown']:
               
        # induce delay
        #sleep(0.5)
                        
        #if not len_emg_raw > len_emg_filt:
        if not sharedConfig['filter_iter'] > iRms:
            #if verbose: print(f"# rms: no new data from filter, skipping")
            pass

        if sharedConfig['filter_iter'] > iRms :

            if verbose: print(f"\niRms {iRms}")

            # the array has been populated, so process the latest batch of data
            
            if verbose: print(f"\n# rms: current filtered is {len(sharedData['emg_delt_filt'])}, current rms is {len(sharedData['emg_delt_rms'])} so we process")
            
            # create local vars just in case a flag changes through the function, we want to finish the loop
            this_emg_filter_do        = sharedConfig['emg_filter_do']
            this_stim_do              = sharedConfig['stim_do']
            this_controller_on        = sharedConfig['controller_on']
            this_pulse_intensity_man  = sharedConfig['pulse_intensity_man']
            this_pulse_intensity_auto = sharedConfig['pulse_intensity_auto']
            
            ###
            # get the latest raw data and do the filtering
            ###

            this_emg_delt_filt = None
            this_emg_bic_filt  = None
            
            # If we want to bypass the filtering and display the RMS value of the RAW directly (for debug purposes, but accessible from GUI)
            if this_emg_filter_do:
                this_emg_delt_filt = sharedData['emg_delt_filt'][-samples_per_read:]
                this_emg_bic_filt  = sharedData['emg_bic_filt'][-samples_per_read:]
            else:
                this_emg_delt_filt = sharedData['emg_delt_raw'][-samples_per_read:]
                this_emg_bic_filt  = sharedData['emg_bic_raw'][-samples_per_read:]

            if verbose: print(f"\nRMS got 2 new filt arrays of length {len(this_emg_delt_filt)} and {len(this_emg_bic_filt)}")

            # get the parameters as can be changed from the GUI
            analogFreq = sharedConfig['rms_analogFreq']
            power      = sharedConfig['rms_power']
            high       = sharedConfig['rms_high']
            low        = sharedConfig['rms_low']
            window     = sharedConfig['rms_window']

            # perform RMS on this batch
            this_emg_delt_rms = convert_to_rms(this_emg_delt_filt, analogFrequency=analogFreq, power=power, highFreq=high, lowFreq=low, window=window)
            this_emg_bic_rms  = convert_to_rms(this_emg_bic_filt,  analogFrequency=analogFreq, power=power, highFreq=high, lowFreq=low, window=window)
            #this_emg_delt_rms = sharedData['emg_delt_filt'][-samples_per_read:] /2
            #this_emg_bic_rms  = sharedData['emg_bic_filt'][-samples_per_read:]  /2
            
            if verbose: print(f"after RMS, their length is {len(this_emg_delt_rms)} and {len(this_emg_bic_rms)}")
            
            # test: show where each batch starts/finished: add a "1" value at beginning of this array
            #this_emg_delt_rms[0] = 1
            #this_emg_bic_rms[0]  = 1

            ###
            # Feedback on if muscles are active
            ###

            ##
            # Prepare Muscle activity detection: choose the threshold to use
            
            # we have two different thresholds for when STIM is active or not
            # i.e. we only take the withStim threshold when:
                # we have been asked to do stim (this_stim_do = True), and:
                    # either: the stimulator is AUTO   (this_controller_on = True)  and its pulse is > 0 (this_pulse_intensity_auto > 0) (set in the previous batch based on threshold_noStim)
                    # or    : the stimulator is MANUAL (this_controller_on = False) and its pulse is > 0 (this_pulse_intensity_man  > 0)
                
            # to simplify the nested ifs, start with noStim threshold, and replace if we match a condition
            this_thres_delt = sharedConfig['emg_thresh_noStim'][0]
            this_thres_bic  = sharedConfig['emg_thresh_noStim'][1]
            
            chosenThreshNoStim   = True
            chosenThreshWithStim = False
            
            if this_stim_do:
                
                if this_controller_on: # auto mode
            
                    if this_pulse_intensity_auto > 0:
                        
                        this_thres_delt = sharedConfig['emg_thresh_withStim'][0]
                        this_thres_bic  = sharedConfig['emg_thresh_withStim'][1] 
                        #if verbose: print("mode auto, this_pulse_intensity_auto > 0 : taking thresh withStim")
                        #if verbose: print(sharedConfig['emg_thresh_withStim'])
                        chosenThreshNoStim   = False
                        chosenThreshWithStim = True
            
                elif not this_controller_on: # manual mode
                
                    if this_pulse_intensity_man > 0:
                    
                        this_thres_delt = sharedConfig['emg_thresh_withStim'][0]
                        this_thres_bic  = sharedConfig['emg_thresh_withStim'][1] 
                        
                        #if verbose: print("mode man, this_pulse_intensity_man > 0 : taking thresh withStim")
                        #if verbose: print(sharedConfig['emg_thresh_withStim'])
                        chosenThreshNoStim   = False
                        chosenThreshWithStim = True
            
            if chosenThreshNoStim:
                if verbose: print(f"we've chosen chosenThreshNoStim ")
                pass
            elif chosenThreshWithStim:
                if verbose: print(f"we've chosen chosenThreshWithStim ")
                pass

            # then, decide if its average is below or above the threshold
            
            if verbose: print("is delt active in this batch?")
            #print(f"{np.mean(this_emg_delt_rms)} >\n{this_thres_delt} ?")
            sharedConfig['active_delt']   = np.mean(this_emg_delt_rms) > this_thres_delt
            sharedConfig['active_bic']    = np.mean(this_emg_bic_rms)  > this_thres_bic
            
            # save muscle activity history timeseries
            if sharedConfig['active_delt']:
                #sharedData['activ_delt_hist'] = np.concatenate( (sharedData['activ_delt_hist'], np.ones(samples_per_read)),  axis=0 )
                np.savetxt(fileActiv_delt, np.ones(samples_per_read), delimiter=',')
            else:
                #sharedData['activ_delt_hist'] = np.concatenate( (sharedData['activ_delt_hist'], np.zeros(samples_per_read)), axis=0 )
                np.savetxt(fileActiv_delt, np.zeros(samples_per_read), delimiter=',')
            if sharedConfig['active_bic']:
                #sharedData['activ_bic_hist']  = np.concatenate( (sharedData['activ_bic_hist'],  np.ones(samples_per_read)),  axis=0 )
                np.savetxt(fileActiv_bic, np.ones(samples_per_read), delimiter=',')
            else:
                #sharedData['activ_bic_hist']  = np.concatenate( (sharedData['activ_bic_hist'],  np.zeros(samples_per_read)), axis=0 )
                np.savetxt(fileActiv_bic, np.zeros(samples_per_read), delimiter=',')
    
            # provide feedback in the console
            if verbose: print(f"DEL {sharedConfig['active_delt']} ; BIC {sharedConfig['active_delt']}")

            ###
            # Send to controller to get new stimulation value (will be updated in the GUI automatically)
            ###
            
            if verbose: print("# emg: asking controller")
            # this is redundant as we have already determined the activity of the muscles, but keeping the controller function tidy and separate for now
            # we send the mean of RMS of each muscle, their threshold of contraction, and the latest controller value
            new_stim_value = controller(np.mean(this_emg_delt_rms), this_thres_delt, np.mean(this_emg_bic_rms), this_thres_bic, sharedConfig['controller_value'], stim_lambda=0.5, verbose=False)
            if verbose: print(f"Controller: setting value from {sharedConfig['controller_value']} to {np.around(new_stim_value,4)}")
            # the ratio (0 -> 1) given by the stimulator is:
            sharedConfig['controller_value'] = np.around(new_stim_value,4)
            # to get the new pulse_intensity value we need to multiply by the max_allowed_intensity from calibration
            sharedConfig['pulse_intensity_auto'] = sharedConfig['controller_value'] * sharedConfig['pulse_intensity_man']
            if verbose: print(f"set pulse_intensity_auto to {sharedConfig['pulse_intensity_auto']}")
            
            if verbose: print(f"new stim value = {new_stim_value}")
            if verbose: print(f"set pulse_intensity_auto to {sharedConfig['pulse_intensity_auto']}")

            ###
            # dump controller values
            ###

            np.savetxt(fileController_value, [new_stim_value]                       * len(this_emg_delt_rms), delimiter=',')
            np.savetxt(fileController_pulse,       [sharedConfig['pulse_intensity_auto']] * len(this_emg_delt_rms), delimiter=',')

            ###
            # dump emg rms data
            ###
            
            np.savetxt(fileRms_delt, this_emg_delt_rms, delimiter=',')
            np.savetxt(fileRms_bic,  this_emg_bic_rms,  delimiter=',')

            ###
            # Put in shared dict
            ###

            if verbose: print("adding to shared dict")
            if verbose: print(f"current size: {len(sharedData['emg_delt_rms'])} and {len(sharedData['emg_bic_rms'])}")

            # we have to make sure that they are the same size as RMS changes the number of points         
            this_emg_delt_rms_tn = time_normalise(this_emg_delt_rms, length=samples_per_read)
            this_emg_bic_rms_tn  = time_normalise(this_emg_bic_rms,  length=samples_per_read)
            if verbose: print(f"after time_normalise, their length is {len(this_emg_delt_rms_tn)} and {len(this_emg_bic_rms_tn)}")
            
            # add to our shared dict
            sharedData['emg_delt_rms'] = np.concatenate( (sharedData['emg_delt_rms'],  this_emg_delt_rms_tn),  axis=0 )
            sharedData['emg_bic_rms']  = np.concatenate( (sharedData['emg_bic_rms'],   this_emg_bic_rms_tn),   axis=0 )
            if verbose: print(f"now rms is {len(sharedData['emg_delt_rms'])} because we just added {len(this_emg_delt_rms_tn)} values")
            if verbose: print(f"after concat, size: {len(sharedData['emg_delt_rms'])} and {len(sharedData['emg_bic_rms'])}")
            
            # remove the first part to keep size consistent (if reached max limit)
            if len(sharedData['emg_delt_rms']) > sharedConfig['maxArraySize']:
                if verbose: print(f"current rms len ({len(sharedData['emg_delt_rms'])}) is > to max array size ({sharedConfig['maxArraySize']})")
                # we remove length of frame from beginning of array
                sharedData['emg_delt_rms'] = sharedData['emg_delt_rms'][samples_per_read:] # sharedData['emg_delt_rms'][len(this_emg_delt_rms):]
                sharedData['emg_bic_rms']  = sharedData['emg_bic_rms'][samples_per_read:]   # sharedData['emg_bic_rms'][len(this_emg_bic_rms):]
                
            if verbose: print(f"after shifting, size: {len(sharedData['emg_delt_rms'])} and {len(sharedData['emg_bic_rms'])}")

            ###
            # only then (after get;dump;resize), increment iterNumber
            ###

            iRms+=1
            sharedConfig['rms_iter'] = iRms
            if verbose: print(f"# rms: new data, done iter {iRms}")

    # Graceful exit
    fileRms_delt.close()
    fileRms_bic.close()
    fileActiv_delt.close()
    fileActiv_bic.close()
    fileController_value.close()
    fileController_pulse.close()

if __name__ == "__main__":
    
    print("orlau_filter loaded as main")
