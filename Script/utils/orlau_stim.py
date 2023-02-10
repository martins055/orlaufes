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
sys.path += [CURR_DIR, CURR_DIR,CURR_DIR+'/utils']         # add relative folders to path, to load our modules easily without installing them

from time import sleep
from sciencemode3 import sciencemode # stimulator
from orlau_utils import showTitle, printColor, show, convert_to_rms

def funcStim2(sharedConfig, sharedData, verbose=False, debug=False):

    if verbose: showTitle('STIM: Connecting to stimulator\n', color='blue')

    ###
    # Find the stimulator
    ###
    
    ack         = sciencemode.ffi.new("Smpt_ack*")
    device      = sciencemode.ffi.new("Smpt_device*")
    version_ack = sciencemode.ffi.new("Smpt_get_version_ack*")
    
    com_connected = False
    com_port_number = 1
    while not com_connected:
        com = sciencemode.ffi.new("char[]", bytes("COM{0}".format(com_port_number), "utf-8"))
        ret = sciencemode.smpt_check_serial_port(com)
        if debug: print(f"checking port {com}")
        if ret==False:
            com_port_number += 1
        else:
            com_connected = True
            if verbose: print(f"#STIM: Found stimulator on COM port {com_port_number}")
    
    ###
    # Connect to the stimulator
    ###

    ret = sciencemode.smpt_open_serial_port(device,com)
    if debug: print("smpt_open_serial_port: {}", ret)

    ###
    # Initiate mid-level mode
    ###

    ml_init = sciencemode.ffi.new("Smpt_ml_init*")
    ml_init.packet_number = sciencemode.smpt_packet_number_generator_next(device)
    ret = sciencemode.smpt_send_ml_init(device, ml_init)
    if debug: print("smpt_send_ml_init: {}", ret)

    # let the other processes we succedded
    sharedConfig['stim_connected'] = True
    if verbose: printColor('#STIM: Connected to stimulator\n', color='green')

    ###
    # Stimulate
    ###
    
    # initiate 0 pulse
    ml_update = sciencemode.ffi.new("Smpt_ml_update*")
    ml_update.packet_number = sciencemode.smpt_packet_number_generator_next(device)
    channelNumber = sharedConfig['stim_channelNumbers']
    ml_update.enable_channel[channelNumber]                   = True
    ml_update.channel_config[channelNumber].period            = sharedConfig['pulse_period']
    ml_update.channel_config[channelNumber].number_of_points  = 1
    ml_update.channel_config[channelNumber].points[0].time    = sharedConfig['pulse_width']
    ml_update.channel_config[channelNumber].points[0].current = 0
    
    if debug: print("set 0 pulse")
    
    # While we don't ask for shutdown,
    iStim = 0
    while not sharedConfig['stim_shutdown']:

        # delay between updates (?10hz)
        sleep(0)

        iStim+=1
        
        if debug: print(f"iStim {iStim}")
        
        # only if we asked the stimulator to send pulses (otherwise it stays at 0)
        if not sharedConfig['stim_do']:
            ml_update.channel_config[channelNumber].points[0].current = 0
        
        if sharedConfig['stim_do']:

            new_pulse_intensity = 0            

            # the intensity value is different if we are in AUTO mode (controller) or MANUAL (set in GUI )

            # if AUTO: get controller's current value
            if sharedConfig['controller_on']:
                new_pulse_intensity = sharedConfig['pulse_intensity_auto']
                if debug: print(f"automode, new_pulse_intensity is {sharedConfig['controller_value']} * {sharedConfig['stim_max_intensity']} = {new_pulse_intensity}")

            # if MANUAL: get value from GUI
            else:
                new_pulse_intensity = sharedConfig['pulse_intensity_man']
                if debug: print(f"manual, new_pulse_intensity is {new_pulse_intensity}")

            # double check that it's not > max or set to max
            if new_pulse_intensity > sharedConfig['stim_max_intensity']:
                new_pulse_intensity = sharedConfig['stim_max_intensity']
                if debug: print(f"intensity > max ({sharedConfig['stim_max_intensity']}), keeping {new_pulse_intensity}")

            # update settings on the stim
            ml_update.channel_config[channelNumber].period            = sharedConfig['pulse_period']
            ml_update.channel_config[channelNumber].points[0].time    = sharedConfig['pulse_width']
            ml_update.channel_config[channelNumber].points[0].current = new_pulse_intensity

        #######################################################################
        # Keep alive (required every < 2 seconds)
        #######################################################################
        ret = sciencemode.smpt_send_ml_update(device, ml_update)

    ###########################################################################
    # Disconnect
    ###########################################################################
    
    if verbose: showTitle("graceful exit of the stimulator")
    ret = sciencemode.smpt_close_serial_port(device)
    if debug: print(f"smpt_close_serial_port: {ret}")
    
