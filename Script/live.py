# -*- coding: utf-8 -*-
"""

@author: Martin

Main script that start the processes including the live plot

- 6 independent processes:
    - connect and get emg from TCU to get raw data
    - connect and use the stimulator
    - filter raw data
    - rms filtered data
    - plot live
    - console monitor

"""

if __name__ == "__main__":
    try:
        from IPython import get_ipython
        get_ipython().run_line_magic('reset', '-sf')
    except: pass

import sys, os, copy, time, pickle, csv
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CURR_DIR, CURR_DIR+'/utils/']
import numpy as np
from datetime import datetime
from time import sleep
import scipy.io as sio
from collections import OrderedDict
import multiprocessing
from multiprocessing import Process, Pipe, freeze_support, Manager

from orlau_utils import showTitle, printColor, show

if __name__ == '__main__':
    
    from orlau_plot            import functionLivePlot
    from orlau_emg             import funcStreamEmg
    from orlau_rms             import funcRms
    from orlau_filter          import funcFilter        # Option 1 : simpliest, for variable frequency noise
    from orlau_filter_mask     import funcFilterMask    # Option 2 : mask faster processing but requires constant frequency noise
    from orlau_monitor         import funcMonitor
    from orlau_stim            import funcStim2

    ###
    # Config
    ###

    config = OrderedDict({
        
        # Session
        'participant_name'      : 'participantA',
        'session_name'          : 'test1',              # custom name for this session/participant.
        'session_date'          : datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss"),
        'session_notes'         : None,
        'dataSaveFolder'        : None,
        'recordButton'          : [],                   # list of lists containing the current iteration of raw/filt/rms emg every time we press it, so we can work out when it was pressed during the whole recording (as we save all data from start/stop of the GUI)

        # EMG and filtering
        'emg_muscles_names'     : ['deltoid','biceps'], # list of the muscles that have an EMG/IMU: one device per muscle
        'emg_thresh_noStim'     : [0.00032,0.00032],                # threshold at which a muscle is considered active (when there is no stimulation running, in which case it is higher)
        'emg_thresh_withStim'   : [0.00032,0.00032],                # threshold at which a muscle is considered active (when stimulation is ongoing)
        'emg_connected'         : False,                # flag indicating if we have successfully connected to the delsys base station and are streaming live
        'emg_shutdown'          : False,                # flag indicating if we should do a graceful exit of the emg process
        'emg_dummyMode'         : False,                # for offline testing live-stream
        'streamTimeStart'       : None,                 # keep track of when we received the first batch of data to display streaming time (as can't calculate from length of arrays that get dumped to disk) 
        'samples_per_read'      : 300,                  # size of batch of data received from the basestation
        'maxArraySize'          : 6000,                 # max length of data arrays to keep in memory before dumping to disk (used by all arrays)
        'emg_iter'              : 0,                    # keep track of the number of batches to synchronise the processes with blocking mechanisms

        # EMG filtering
        'emg_filter_do'         : True,                 # for debug: only change for testing! True by default. if False, the "filt" process will pass the raw data to rms. if True, uses 'stim_thresh' and 'filter_delWindow' to filter the data
        'stim_thresh'           : 0.0009,               # the threshold at which we will decide to start filtering EMG when the stim is making our signal noisy
        'filter_delWindow'      : 10,                   # number of data to filter out before and after each value that was above the stimulation trigger value 'stim_thresh'. Try a value between 10 and 20.
        'filter_iter'           : 0,                    # keep track of the number of batches to synchronise the processes with blocking mechanisms

        # EMG rms
        'rms_analogFreq'        : 2000,                 # RMS parameters for emg filtering (delsys is 2000Hz)
        'rms_power'             : 2,
        'rms_high'              : 10,
        'rms_low'               : 400,
        'rms_window'            : 150,
        'rms_iter'              : 0,                    # keep track of the number of batches to synchronise the processes with blocking mechanisms

        # Controller
        'controller_on'         : False,                # if True, the intensity is set by the feedback of the controller. If False, we are in "manual" mode and the pulse intensity is set in the GUI. The controller is called by the EMG process thus this parameter here
        'controller_value'      : 0,                    # starts at 0, will be changed by the emg process that calls the controller

        # Stim
        'stim_musclesName'      : 'triceps',            # list of the muscles stimulated
        'stim_channelNumbers'   : 1,                    # lisf of the color of the cables of the stimulator (red = 0, blue = 1, grey = 2)
        'stim_max_intensitiy'   : 15,                   # the hardcoded max intensity chosen at calibration for each muscle (milliAmp)
        'pulse_width'           : 300,                  # the pulse length, or 'time' in microseconds (us)
        'pulse_period'          : 40,                   # the period in milliSeconds, so if period is 25ms it is 1/25 = 40Hz ; 1/10 = 100Hz
        'pulse_intensity_man'   : 0,                    # current or intensity in milliAmps: start at 0 - the one we set manually in the GUI
        'pulse_intensity_auto'  : 0,                    # current or intensity in milliAmps: start at 0 - the one that is calculated from the controller's value
        'stim_connected'        : False,                # flag indicating if we have successfully connected to the stimulator
        'stim_shutdown'         : False,                # flag indicating if we should do a graceful exit of the stimulator process
        'stim_dummyMode'        : False,                # for offline testing live-stream
        'stim_do'               : False,                # flag indicating if the stimulator should send a pulse (whether it skips it or sends a 0 pulse)

        # GUI only
        'gui_preview_type'      : 'raw',                # raw / rms (takes from a different array)
        'gui_zoom'              : 0.0014,               # zoom value for the plot
        'gui_refresh_delay'     : 200,                  # refresh rate of the GUI (ms)
        'gui_downsample'        : 0,                    # if want to downsample the preview (does not impact the actual data, just the live plot!)
        'active_delt'           : False,                # the emg rms process/controller will decide which muscle is active, and update this flag
        'active_bic'            : False,
        'recordBtn_timeFrames'  : 0,                    # saved the frame current frame number when pressing the record button: it is generic so can also serve to indicate various events (if following a protocol) that we can flag when saving the data
        'buttonRecord_status'   : False,                # feedback on whether we are set to save the current stream as recorded or not
        })

    data = OrderedDict({
        
        'emg_delt_raw'        : np.array([]), # we constantly save the raw data from the basestation so that is our raw data
        'emg_bic_raw'         : np.array([]),
        
        'emg_delt_filt'       : np.array([]), # the raw data is then going through the filter and is saved here
        'emg_bic_filt'        : np.array([]),
        
        'emg_delt_rms'        : np.array([]), # after the filtering, rms is performed
        'emg_bic_rms'         : np.array([]),
        
        'controller_val_hist' : np.array([]), # history of the values of the controller
        'pulse_val_hist'      : np.array([]), # history of the values of the pulse intensity, might be useful when we look at the calibrations?
        'activ_delt_hist'     : np.array([]), # history of the activity of the muscles ? useful for analysing results easily
        'activ_bic_hist'      : np.array([]),

        })

    # Prepare folders for data recording
    config['dataSaveFolder'] = CURR_DIR + "/Data/" + config['participant_name'] + '_' + config['session_name'] + "_" + config['session_date']
    # Make sure output folder exists
    if not os.path.exists(config['dataSaveFolder']):
        print(f"creating output folder")
        os.makedirs(config['dataSaveFolder'])

    # Make our config a shared object (create unique keys instead of nesting a dict: otherwise proxy won't be triggered and change not propagated to manager)
    sharedConfig = multiprocessing.Manager().dict()  # dict wit results
    for key in config:
        sharedConfig[key] = config[key]

    # Then we create a different sharedData for the data (so that we can export config / data easily and separately!)
    sharedData   = multiprocessing.Manager().dict()  # dict wit results
    for key in data:
        sharedData[key] = data[key]

    ###
    # Create the processes
    ###

    # set FIFO queues as alternative mechanism also
    sharedQueue1         = multiprocessing.Manager().Queue()
    sharedQueue2         = multiprocessing.Manager().Queue()
    sharedQueue1rms      = multiprocessing.Manager().Queue()
    sharedQueue2rms      = multiprocessing.Manager().Queue()
    sharedQueue1Filtered = multiprocessing.Manager().Queue()
    sharedQueue2Filtered = multiprocessing.Manager().Queue()

    # Dummy mode for offline testing
    sharedConfig['emg_dummyMode']    = False

    # 1) get raw emg from delsys place in shared obj
    t1 = multiprocessing.Process(target=funcStreamEmg,    args=(sharedConfig, sharedData, sharedQueue1, sharedQueue2), kwargs={"verbose":False,"debug":False})
    # 2) filter from raw (whether we perform any filtering or not, it's going through it!)
    t2 = multiprocessing.Process(target=funcFilter,       args=(sharedConfig, sharedData, sharedQueue1, sharedQueue2), kwargs={"verbose":True,"debug":False})  
    # 3) rms on filtered data from t2 to t3
    t3 = multiprocessing.Process(target=funcRms,          args=(sharedConfig, sharedData, sharedQueue1, sharedQueue2), kwargs={"verbose":False,"debug":False})
    # 4) live plot from raw/filtered/rms
    t4 = multiprocessing.Process(target=functionLivePlot, args=(sharedConfig, sharedData, sharedQueue1, sharedQueue2), kwargs={"verbose":False,"debug":False})
    # 5) stimulator
    t5 = multiprocessing.Process(target=funcStim2,        args=(sharedConfig, sharedData),                             kwargs={"verbose":False,"debug":False})
    # 6) console monitor for summary of processes
    t6 = multiprocessing.Process(target=funcMonitor,      args=(sharedConfig, sharedData))

    # Start the processes
    t1.start() # emg delsys
    t2.start() # filter
    t3.start() # rms
    t4.start() # plot
    t5.start() # stim
    t6.start() # monitor

    # Join the GUI. Blocking mechanism: will move on and terminate the other processes once it is closed only
    t4.join() # wait for GUI to close
   
    print("\nClosed GUI, processing graceful exit")

    # Graceful exit by setting flags so that each process has time to finish and wrap up current iteration
    sharedConfig['emg_shutdown']  = True
    sharedConfig['stim_shutdown'] = True

    ###
    # Tidy up recorded values
    ###
    """
    # need to put all in separate columns, and add time also
    # need to make separate ones for each "recorded" times (in between the values of recordBtn_timeFrames)

    # pickles
    finalConfig = {}
    finalData   = {}
    # deproxify
    for key in sharedConfig:
        finalConfig[key] = sharedConfig[key]
    for key in sharedData:
        finalData[key] = sharedData[key]
    filehandler = open(CURR_DIR+'/Data/'+outputFolder+'/config.pickle', 'wb')
    pickle.dump(finalConfig, filehandler)
    filehandler.close()
    filehandler = open(CURR_DIR+'/Data/'+outputFolder+'/data.pickle', 'wb')
    pickle.dump(finalData, filehandler)
    filehandler.close()

    # csv
    for fileName in [key for key in sharedData]:
        print(fileName)
        print(f"saving {fileName} : len {len(sharedData[fileName])}")
        np.savetxt(CURR_DIR+'/Data/'+outputFolder+'/'+fileName+'.csv', sharedData[fileName], delimiter=",")
    
    # matlab
    sio.savemat(CURR_DIR+'/Data/'+outputFolder+'/config.mat', config)
    sio.savemat(CURR_DIR+'/Data/'+outputFolder+'/data.mat', data)
    """

    sleep(3) # give some time to the processes to finish before the main process exits (or it will kill spawned ones), esp. the stimulator to send the disconnection signal to the DLL

    print("(main live) Goodbye")

