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

import sys, os, copy, time, pickle, csv
CURR_DIR = os.path.dirname(os.path.realpath(__file__))     # get current directory variable
os.chdir(os.path.abspath(os.path.dirname(__file__)))       # change current working directory
sys.path += [CURR_DIR, CURR_DIR+'/utils/']                 # add relative folders to path, to load our modules easily without installing them
import numpy as np
from datetime import datetime
from time import sleep
from collections import OrderedDict
import multiprocessing
from multiprocessing import Process, Pipe, freeze_support, Manager
from orlau_utils import showTitle, printColor, show, convert_to_rms, time_normalise

def downSampleData(data, times=1):

    # Other option: time normalise to reduce number of points : warning, computationally expensive during a live plot!
    # prefer "downSamplePreview = True"
    #    previewData_DELT = time_normalise(previewData_DELT, length=previewData_size)
    #    previewData_BIC  = time_normalise(previewData_BIC,  length=previewData_size)

    # simply remove 1 every other value for quick smooth plotting that takes less computation (no processing just removing some of the data for preview)
    # do this as many times as requested (default=1)
    # if we also process x, we cover the same "range" on the graph (final x has the same value)
    for i in range(times):
        data = data[list(range(0,len(data),2))]
    return data

def functionLivePlot(sharedConfig, sharedData, sharedQueue1, sharedQueue2, verbose=False, debug=False):

    from PyQt5.QtWidgets import QApplication, QWidget,QDesktopWidget, QGroupBox, QLabel, QPushButton, QMessageBox, QShortcut, QGridLayout, QHBoxLayout, QVBoxLayout, QFormLayout, QLineEdit, QScrollArea
    from PyQt5.QtGui import QKeySequence, QIcon, QFont
    from PyQt5.QtCore import Qt # Qt.AlignCenter
    from scipy import signal as scipySignal

    ###########################################################################
    # Main
    ###########################################################################

    app = QApplication(sys.argv)
    def slot_aboutToQuit():
        print(f'stopping stimulator straight away on closing window')
        # make sure to stop stimulator now, otherwise it will continue for the 2 seconds of the midlevel mode, and might take a while between the moment the window is closing and the process is actually terminated
        sharedConfig['stim_do']       = False
    app.aboutToQuit.connect(slot_aboutToQuit) # myExitHandler is a callable

    # Create a main widget
    w = QWidget()
    w.setWindowTitle("FES streaming")

    # Layout general
    outerLayout = QGridLayout(w)

    # set the top and bottom parts
    topLayout = QHBoxLayout() # horizontal layout
    outerLayout.addLayout(topLayout, 0,0) # add this to the main layout
    # add bottom text as outerLayout 1,0
    bottomLayout = QGridLayout() # another grid for alignment of buttons
    outerLayout.addLayout(bottomLayout, 1,0) # add this to the main layout

    # now in the top part, use the 2 columns
    topLeftLayout  = QFormLayout() # feedback of muscle activity and stim value
    topRightLayout = QFormLayout() # live plot
    topLayout.addLayout(topLeftLayout,  30) # % of the width
    topLayout.addLayout(topRightLayout, 70)
    
    # populate the left column (settings)

    ###########################################################################
    # Biofeedback
    ###########################################################################

    # create box
    groupboxFeedback = QGroupBox("Biofeedback")
    topLeftLayout.addWidget(groupboxFeedback)
    boxFeedback = QGridLayout()
    groupboxFeedback.setLayout(boxFeedback)
    
    global delt_activ_label
    delt_activ_label1 = QLabel(f"DELT active")
    boxFeedback.addWidget(delt_activ_label1,0,0)
    delt_activ_label2 = QLabel(f"{sharedConfig['active_delt']}")
    delt_activ_label2.setStyleSheet("QLabel{color: red;font-size:20px;font-family:'Orbitron'}")
    boxFeedback.addWidget(delt_activ_label2,0,1)
    
    global bic_activ_label
    bic_activ_label1 = QLabel(f"BIC active")
    boxFeedback.addWidget(bic_activ_label1,1,0)
    bic_activ_label2 = QLabel(f"{sharedConfig['active_bic']}")
    bic_activ_label2.setStyleSheet("QLabel{color: red;font-size:20px;font-family:'Orbitron'}")
    boxFeedback.addWidget(bic_activ_label2,1,1)

    global controller_value_label
    controller_value_label1 = QLabel(f"Controller")
    boxFeedback.addWidget(controller_value_label1,0,2)
    controller_value_label2 = QLabel(f"{sharedConfig['controller_value']}")
    controller_value_label2.setStyleSheet("QLabel{color: black;font-size:20px;font-family:'Orbitron'}")
    boxFeedback.addWidget(controller_value_label2,0,3)
    
    global pulse_intensity_label
    pulse_intensity_label1 = QLabel(f"Pulse intensity")
    boxFeedback.addWidget(pulse_intensity_label1,1,2)
    pulse_intensity_label2 = QLabel()
    if sharedConfig['controller_on']:
        # if auto, the current pulse_intensity is max_allowed_intensity * sharedData['controller_value']
        new_pulse_intensity = sharedConfig['controller_value'] * sharedConfig['stim_max_intensitiy']
        pulse_intensity_label2.setText(f"{new_pulse_intensity:.3f}")
    else:
        # if mode manual, it is set manually in the editbox
        pulse_intensity_label2.setText(f"{sharedConfig['pulse_intensity_man']}")
    pulse_intensity_label2.setStyleSheet("QLabel{color: black;font-size:20px;font-family:'Orbitron'}")
    boxFeedback.addWidget(pulse_intensity_label2,1,3)
    
    ###
    # GUI options
    ###
    
    # Create a new groupbox
    groupboxGuiOptions = QGroupBox("GUI options")
    topLeftLayout.addWidget(groupboxGuiOptions)
    boxGuiOptions = QGridLayout()
    groupboxGuiOptions.setLayout(boxGuiOptions)
    
    #Zoom
    #heightGraph = 0.0012 # 0.0011 ; 0.014 ; 120 # max range of EMG data is 11 mV # zoom
    paramZoom = QLabel(f"Zoom") # add a text
    paramZoom.setStyleSheet("QLabel {background-color: ;}")
    btn_paramZoom_plus  = QPushButton("+")
    btn_paramZoom_minus = QPushButton("-")
    btn_paramZoom_set   = QPushButton("SET")
    editZoom = QLineEdit(str(sharedConfig['gui_zoom']))
    boxGuiOptions.addWidget(paramZoom,0,0,1,1)
    boxGuiOptions.addWidget(editZoom,0,1,1,2)
    boxGuiOptions.addWidget(btn_paramZoom_plus,0,3,1,1)
    boxGuiOptions.addWidget(btn_paramZoom_minus,0,4,1,1)
    boxGuiOptions.addWidget(btn_paramZoom_set,0,5,1,1)
    def slot_zoom_set():
        print(f'zoom: applying new value {editZoom.text()}')
        this_zoom = float(editZoom.text())
        dynamic_ax.set_ylim(bottom = -this_zoom, top = +this_zoom)
    btn_paramZoom_set.clicked.connect(slot_zoom_set)
    def slot_zoom_plus():
        this_zoom = float(editZoom.text())
        new_zoom = round( (this_zoom - 0.0002), 5)
        print(f'zoom: set value from {this_zoom} to {new_zoom}')
        editZoom.setText(str(new_zoom))
        slot_zoom_set()
    btn_paramZoom_plus.clicked.connect(slot_zoom_plus)
    def slot_zoom_minus():
        this_zoom = float(editZoom.text())
        new_zoom = round( (this_zoom + 0.0002), 5)
        print(f'zoom: set value from {this_zoom} to {new_zoom}')
        editZoom.setText(str(new_zoom))
        slot_zoom_set()
    btn_paramZoom_minus.clicked.connect(slot_zoom_minus)

    #Refresh rate
    paramRefreshRate = QLabel(f"Refresh (ms)")
    paramRefreshRate.setStyleSheet("QLabel {background-color: ;}")
    btn_paramRefreshRate_plus  = QPushButton("+")
    btn_paramRefreshRate_minus = QPushButton("-")
    btn_paramRefreshRate_set   = QPushButton("SET")
    editRefreshRate = QLineEdit(str(sharedConfig['gui_refresh_delay']))
    boxGuiOptions.addWidget(paramRefreshRate,1,0,1,1)
    boxGuiOptions.addWidget(editRefreshRate,1,1,1,2)
    boxGuiOptions.addWidget(btn_paramRefreshRate_plus,1,3,1,1)
    boxGuiOptions.addWidget(btn_paramRefreshRate_minus,1,4,1,1)
    boxGuiOptions.addWidget(btn_paramRefreshRate_set,1,5,1,1)
    def slot_refreshRate_set():
        this_val = int(editRefreshRate.text())
        sharedConfig['gui_refresh_delay'] = this_val
        print(f"refresh rate is now {sharedConfig['gui_refresh_delay']}")
    btn_paramRefreshRate_set.clicked.connect(slot_refreshRate_set)
    def slot_refreshRate_plus():
        this_val = int(editRefreshRate.text())
        #new_val  = round( (this_val + 0.1), 5)
        new_val = this_val + 5
        print(f'refreshRate: set sleep value from {this_val} to {new_val}')
        editRefreshRate.setText(str(new_val))
        slot_refreshRate_set()
    btn_paramRefreshRate_plus.clicked.connect(slot_refreshRate_plus)
    def slot_refreshRate_minus():
        this_val = int(editRefreshRate.text())
        #new_val  = round( (this_val - 0.1), 5)
        new_val = this_val - 5
        print(f'refreshRate: set sleep value from {this_val} to {new_val}')
        editRefreshRate.setText(str(new_val))
        slot_refreshRate_set()
    btn_paramRefreshRate_minus.clicked.connect(slot_refreshRate_minus)

    #Downsample
    paramDownsample = QLabel(f"Downsample")
    paramDownsample.setStyleSheet("QLabel {background-color: ;}")
    btn_paramDownsample_plus  = QPushButton("+")
    btn_paramDownsample_minus = QPushButton("-")
    btn_paramDownsample_set   = QPushButton("SET")
    editparamDownsample = QLineEdit(str(sharedConfig['gui_downsample']))
    boxGuiOptions.addWidget(paramDownsample,2,0,1,1)
    boxGuiOptions.addWidget(editparamDownsample,2,1,1,2)
    boxGuiOptions.addWidget(btn_paramDownsample_plus,2,3,1,1)
    boxGuiOptions.addWidget(btn_paramDownsample_minus,2,4,1,1)
    boxGuiOptions.addWidget(btn_paramDownsample_set,2,5,1,1)
    def slot_downSample_set():
        print(f'downsample: applying new value {editparamDownsample.text()}')
        this_val = int(editparamDownsample.text())
        if this_val < 0:
            print("downsample: corrected to 0")
            this_val = 0
            editparamDownsample.setText(str(this_val))
        sharedConfig['gui_downsample'] = this_val
    btn_paramDownsample_set.clicked.connect(slot_downSample_set)
    def slot_downSample_plus():
        this_val = int(editparamDownsample.text())
        new_val  = this_val + 1
        print(f'downsample: set sleep value from {this_val} to {new_val}')
        editparamDownsample.setText(str(new_val))
        slot_downSample_set()
    btn_paramDownsample_plus.clicked.connect(slot_downSample_plus)
    def slot_downSample_minus():
        this_val = int(editparamDownsample.text())
        new_val  = this_val - 1
        print(f'downsample: set sleep value from {this_val} to {new_val}')
        editparamDownsample.setText(str(new_val))
        slot_downSample_set()
    btn_paramDownsample_minus.clicked.connect(slot_downSample_minus)

    # Display RMS / RAW
    paramPreviewType1 = QLabel(f"Preview type")
    paramPreviewType2 = QLabel(f"{sharedConfig['gui_preview_type']}")
    btn_paramPreviewType_RAW  = QPushButton("RAW")
    btn_paramPreviewType_FILT = QPushButton("FILT")
    btn_paramPreviewType_RMS  = QPushButton("RMS")
    boxGuiOptions.addWidget(paramPreviewType1,        4,0)
    boxGuiOptions.addWidget(paramPreviewType2,        4,1,1,2)
    boxGuiOptions.addWidget(btn_paramPreviewType_RAW, 4,3)
    boxGuiOptions.addWidget(btn_paramPreviewType_FILT,4,4)
    boxGuiOptions.addWidget(btn_paramPreviewType_RMS, 4,5)
    def slot_PreviewType_RAW():
        sharedConfig['gui_preview_type'] = 'raw'
        paramPreviewType2.setText(f"{sharedConfig['gui_preview_type']}")
        print(f"previewType set to {sharedConfig['gui_preview_type']}")
    btn_paramPreviewType_RAW.clicked.connect(slot_PreviewType_RAW)
    def slot_PreviewType_FILT():
        sharedConfig['gui_preview_type'] = 'filt'
        paramPreviewType2.setText(f"{sharedConfig['gui_preview_type']}")
        print(f"previewType set to {sharedConfig['gui_preview_type']}")
    btn_paramPreviewType_FILT.clicked.connect(slot_PreviewType_FILT)
    def slot_PreviewType_RMS():
        sharedConfig['gui_preview_type'] = 'rms'
        paramPreviewType2.setText(f"{sharedConfig['gui_preview_type']}")
        print(f"previewType set to {sharedConfig['gui_preview_type']}")
    btn_paramPreviewType_RMS.clicked.connect(slot_PreviewType_RMS)

    ###
    # RMS/EMG options
    ###
    
    # Create a new groupbox
    groupboxRMSOptions = QGroupBox("RMS options")
    topLeftLayout.addWidget(groupboxRMSOptions)
    boxRMSOptions = QGridLayout()
    groupboxRMSOptions.setLayout(boxRMSOptions)
    
    # RMS analog frequency
    paramRMSanalogFreq = QLabel(f"Freq")
    paramRMSanalogFreq.setStyleSheet("QLabel {background-color: ;}")
    btn_paramRMSanalogFreq_plus  = QPushButton("+")
    btn_paramRMSanalogFreq_minus = QPushButton("-")
    btn_paramRMSanalogFreq_set   = QPushButton("SET")
    editParamRMSanalogFreq = QLineEdit(str(sharedConfig['rms_analogFreq']))
    boxRMSOptions.addWidget(paramRMSanalogFreq,0,0)
    boxRMSOptions.addWidget(editParamRMSanalogFreq,0,1)
    boxRMSOptions.addWidget(btn_paramRMSanalogFreq_plus,0,3)
    boxRMSOptions.addWidget(btn_paramRMSanalogFreq_minus,0,4)
    boxRMSOptions.addWidget(btn_paramRMSanalogFreq_set,0,5)
    def slot_RMSanalogFreq_set():
        # get the value as an int from the edit
        new_val = int(editParamRMSanalogFreq.text())
        # as it is a shared between processes dict, can't edit directly and need to:
        sharedConfig['rms_analogFreq'] # take the dict out of sharedData
        print(f'RMSanalogFreq: applying new value {new_val}')
    btn_paramRMSanalogFreq_set.clicked.connect(slot_RMSanalogFreq_set)
    def slot_paramRMSanalogFreq_plus():
        curr_val = int(editParamRMSanalogFreq.text()) # get the value as an int from the edit
        new_val  = curr_val + 1 # increment
        # replace it in the editbox
        editParamRMSanalogFreq.setText(str(new_val))
        # call set to save it
        slot_RMSanalogFreq_set()
        print(f'RMSanalogFreq: incrementing to {new_val}')
    btn_paramRMSanalogFreq_plus.clicked.connect(slot_paramRMSanalogFreq_plus)
    def slot_paramRMSanalogFreq_minus():
        curr_val = int(editParamRMSanalogFreq.text()) # get the value as an int from the edit
        new_val  = curr_val - 1 # increment
        # replace it in the editbox
        editParamRMSanalogFreq.setText(str(new_val))
        # call set to save it
        slot_RMSanalogFreq_set()
        print(f'RMSanalogFreq: decrementing to {new_val}')
    btn_paramRMSanalogFreq_minus.clicked.connect(slot_paramRMSanalogFreq_minus)

    # RMS power
    paramRMSpower = QLabel(f"Power")
    paramRMSpower.setStyleSheet("QLabel {background-color: ;}")
    btn_paramRMSpower_plus  = QPushButton("+")
    btn_paramRMSpower_minus = QPushButton("-")
    btn_paramRMSpower_set   = QPushButton("SET")
    editParamRMSpower = QLineEdit(str(sharedConfig['rms_power']))
    boxRMSOptions.addWidget(paramRMSpower,1,0)
    boxRMSOptions.addWidget(editParamRMSpower,1,1)
    boxRMSOptions.addWidget(btn_paramRMSpower_plus,1,3)
    boxRMSOptions.addWidget(btn_paramRMSpower_minus,1,4)
    boxRMSOptions.addWidget(btn_paramRMSpower_set,1,5)
    def slot_RMSpower_set():
        new_val = int(editParamRMSpower.text())
        sharedConfig['rms_power'] = new_val
        print(f'RMSpower: applying new value {new_val}')
    btn_paramRMSpower_set.clicked.connect(slot_RMSpower_set)
    def slot_paramRMSpower_plus():
        curr_val = int(editParamRMSpower.text())
        new_val  = curr_val + 1
        editParamRMSpower.setText(str(new_val))
        slot_RMSpower_set()
        print(f'RMSpower: incrementing to {new_val}')
    btn_paramRMSpower_plus.clicked.connect(slot_paramRMSpower_plus)
    def slot_paramRMSpower_minus():
        curr_val = int(editParamRMSpower.text())
        new_val  = curr_val - 1
        editParamRMSpower.setText(str(new_val))
        slot_RMSpower_set()
        print(f'RMSpower: decrementing to {new_val}')
    btn_paramRMSpower_minus.clicked.connect(slot_paramRMSpower_minus)
    
    # RMS high
    paramRMShigh = QLabel(f"High")
    paramRMShigh.setStyleSheet("QLabel {background-color: ;}")
    btn_paramRMShigh_plus  = QPushButton("+")
    btn_paramRMShigh_minus = QPushButton("-")
    btn_paramRMShigh_set   = QPushButton("SET")
    editParamRMShigh = QLineEdit(str(sharedConfig['rms_high']))
    boxRMSOptions.addWidget(paramRMShigh,2,0)
    boxRMSOptions.addWidget(editParamRMShigh,2,1)
    boxRMSOptions.addWidget(btn_paramRMShigh_plus,2,3)
    boxRMSOptions.addWidget(btn_paramRMShigh_minus,2,4)
    boxRMSOptions.addWidget(btn_paramRMShigh_set,2,5)
    def slot_RMShigh_set():
        new_val = int(editParamRMShigh.text())
        sharedConfig['rms_high'] = new_val
    btn_paramRMShigh_set.clicked.connect(slot_RMShigh_set)
    def slot_paramRMShigh_plus():
        curr_val = int(editParamRMShigh.text())
        new_val  = curr_val + 1
        editParamRMShigh.setText(str(new_val))
        slot_RMShigh_set()
        print(f'RMShigh: incrementing to {new_val}')
    btn_paramRMShigh_plus.clicked.connect(slot_paramRMShigh_plus)
    def slot_paramRMShigh_minus():
        curr_val = int(editParamRMShigh.text())
        new_val  = curr_val - 1
        editParamRMShigh.setText(str(new_val))
        slot_RMShigh_set()
        print(f'RMShigh: decrementing to {new_val}')
    btn_paramRMShigh_minus.clicked.connect(slot_paramRMShigh_minus)
    
    # RMS low
    paramRMSlow = QLabel(f"Low")
    paramRMSlow.setStyleSheet("QLabel {background-color: ;}")
    btn_paramRMSlow_plus  = QPushButton("+")
    btn_paramRMSlow_minus = QPushButton("-")
    btn_paramRMSlow_set   = QPushButton("SET")
    editParamRMSlow = QLineEdit(str(sharedConfig['rms_low']))
    boxRMSOptions.addWidget(paramRMSlow,3,0)
    boxRMSOptions.addWidget(editParamRMSlow,3,1)
    boxRMSOptions.addWidget(btn_paramRMSlow_plus,3,3)
    boxRMSOptions.addWidget(btn_paramRMSlow_minus,3,4)
    boxRMSOptions.addWidget(btn_paramRMSlow_set,3,5)
    def slot_RMSlow_set():
        new_val = int(editParamRMSlow.text())
        sharedConfig['rms_low'] = new_val
        print(f'RMSlow: applying new value {new_val}')
    btn_paramRMSlow_set.clicked.connect(slot_RMSlow_set)
    def slot_paramRMSlow_plus():
        curr_val = int(editParamRMSlow.text())
        new_val  = curr_val + 1
        editParamRMSlow.setText(str(new_val))
        slot_RMSlow_set()
        print(f'RMSlow: incrementing to {new_val}')
    btn_paramRMSlow_plus.clicked.connect(slot_paramRMSlow_plus)
    def slot_paramRMSlow_minus():
        curr_val = int(editParamRMSlow.text())
        new_val  = curr_val - 1
        editParamRMSlow.setText(str(new_val))
        slot_RMSlow_set()
        print(f'RMSlow: decrementing to {new_val}')
    btn_paramRMSlow_minus.clicked.connect(slot_paramRMSlow_minus)
    
    # RMS window
    paramRMSwindow = QLabel(f"Window")
    btn_paramRMSwindow_plus  = QPushButton("+")
    btn_paramRMSwindow_minus = QPushButton("-")
    btn_paramRMSwindow_set   = QPushButton("SET")
    editParamRMSwindow = QLineEdit(str(sharedConfig['rms_window']))
    boxRMSOptions.addWidget(paramRMSwindow,4,0)
    boxRMSOptions.addWidget(editParamRMSwindow,4,1)
    boxRMSOptions.addWidget(btn_paramRMSwindow_plus,4,3)
    boxRMSOptions.addWidget(btn_paramRMSwindow_minus,4,4)
    boxRMSOptions.addWidget(btn_paramRMSwindow_set,4,5)
    def slot_RMSwindow_set():
        new_val = int(editParamRMSwindow.text())
        sharedConfig['rms_window'] = new_val
        print(f'RMSwindow: applying new value {new_val}')
    btn_paramRMSwindow_set.clicked.connect(slot_RMSwindow_set)
    def slot_paramRMSwindow_plus():
        curr_val = int(editParamRMSwindow.text())
        new_val  = curr_val + 1
        editParamRMSwindow.setText(str(new_val))
        slot_RMSwindow_set()
        print(f'RMSwindow: incrementing to {new_val}')
    btn_paramRMSwindow_plus.clicked.connect(slot_paramRMSwindow_plus)
    def slot_paramRMSwindow_minus():
        curr_val = int(editParamRMSwindow.text())
        new_val  = curr_val - 1
        editParamRMSwindow.setText(str(new_val))
        slot_RMSwindow_set()
        print(f'RMSwindow: decrementing to {new_val}')
    btn_paramRMSwindow_minus.clicked.connect(slot_paramRMSwindow_minus)
        
    ###
    # now a scrolling area for the rest of the settings
    ###
    
    scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
    widget = QWidget()                 # Widget that contains the collection of Vertical Box
    vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of labels and buttons
    widget.setLayout(vbox)
    widget.setContentsMargins(0,0,0,0)
    #Scroll Area Properties
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    #add it to the main layout
    topLeftLayout.addWidget(scroll)

    # now add the rest in this scrollbox
    #for i in range(1,50):
    #   object = QLabel("TextLabel")
    #   vbox.addWidget(object)

    ###
    # Info patient/session
    ###
    
    # Create a new groupbox
    groupboxInfo = QGroupBox("Info session")
    vbox.addWidget(groupboxInfo)
    boxInfo = QGridLayout()
    boxInfo.setContentsMargins(0,0,0,0) # top left right bottom
    groupboxInfo.setLayout(boxInfo)
    
    # participant_name
    label_infoName   = QLabel(f"Name")    
    edit_infoName    = QLineEdit(str(sharedConfig['participant_name']))
    btn_infoName_set = QPushButton("SET")
    boxInfo.addWidget(label_infoName,0,0,1,1)
    boxInfo.addWidget(edit_infoName,0,1,1,4)
    boxInfo.addWidget(btn_infoName_set,0,5,1,1)
    def slot_infoName_set():
        new_val = edit_infoName.text()
        sharedConfig['participant_name'] = new_val
        print(f'infoName: applying new value {new_val}')
    btn_infoName_set.clicked.connect(slot_infoName_set)

    # session_name
    label_infoSession   = QLabel(f"Session")    
    edit_infoSession    = QLineEdit(str(sharedConfig['session_name']))
    btn_infoSession_set = QPushButton("SET")
    boxInfo.addWidget(label_infoSession,1,0)
    boxInfo.addWidget(edit_infoSession,1,1,1,4)
    boxInfo.addWidget(btn_infoSession_set,1,5)
    def slot_infoSession_set():
        new_val = edit_infoSession.text()
        sharedConfig['session_name'] = new_val
        print(f'infoSession: applying new value {new_val}')
    btn_infoSession_set.clicked.connect(slot_infoSession_set)

    #session_notes
    label_infoNotes   = QLabel(f"Notes")    
    edit_infoNotes    = QLineEdit(str(sharedConfig['session_notes']))
    btn_infoNotes_set = QPushButton("SET")
    boxInfo.addWidget(label_infoNotes,2,0)
    boxInfo.addWidget(edit_infoNotes,2,1,1,4)
    boxInfo.addWidget(btn_infoNotes_set,2,5)
    def slot_infoNotes_set():
        new_val = edit_infoNotes.text()
        sharedConfig['session_notes'] = new_val
        print(f'infoNotes: applying new value {new_val}')
    btn_infoNotes_set.clicked.connect(slot_infoNotes_set)

    ###
    # Calibration EMG thresholds without stim
    ###

    # Create a new groupbox
    groupboxCalibNoEmg = QGroupBox("1) EMG calibration without stimulator")
    vbox.addWidget(groupboxCalibNoEmg)
    boxCalibNoEmg = QGridLayout()
    groupboxCalibNoEmg.setLayout(boxCalibNoEmg)

    #Muscle threshold without stimulation : Delt
    paramMuscle0Thres = QLabel(f"Delt Thres")
    paramMuscle0Thres.setStyleSheet("QLabel {background-color: ;}")
    btn_paramMuscle0Thres_plus  = QPushButton("+")
    btn_paramMuscle0Thres_minus = QPushButton("-")
    btn_paramMuscle0Thres_set   = QPushButton("SET")
    editMuscle0Thres = QLineEdit(f"{sharedConfig['emg_thresh_noStim'][0]:.7f}")
    boxCalibNoEmg.addWidget(paramMuscle0Thres,0,0)
    boxCalibNoEmg.addWidget(editMuscle0Thres,0,1)
    boxCalibNoEmg.addWidget(btn_paramMuscle0Thres_plus,0,2)
    boxCalibNoEmg.addWidget(btn_paramMuscle0Thres_minus,0,3)
    boxCalibNoEmg.addWidget(btn_paramMuscle0Thres_set,0,4)
    def slot_muscle0Thres_set():
        new_val = float(editMuscle0Thres.text())
        # can't edit directly nested elements from a manager object (like the dicts)
        curr_values = sharedConfig['emg_thresh_noStim']
        curr_values[0] = new_val
        print(f"new val from gui is {new_val}")
        sharedConfig['emg_thresh_noStim'] = curr_values
        print(f"now in shared dict is its")
        print(sharedConfig['emg_thresh_noStim'])
        # update on graph
        print(f"set {sharedConfig['emg_thresh_noStim'][0]} in shared dict, applying to line3")
        try:
            xaxis = np.array(range(0,previewData_size))
            line3.set_data(xaxis, sharedConfig['emg_thresh_noStim'][0])
            line3.figure.canvas.draw()
        except:
            print("failed")
        print(f"new value is now { sharedConfig['emg_thresh_noStim'][0]}")
    btn_paramMuscle0Thres_set.clicked.connect(slot_muscle0Thres_set)
    def slot_muscle0Thres_plus():
        this_muscle0Thres = float(editMuscle0Thres.text())
        new_muscle0Thres = round( (this_muscle0Thres + 0.00001), 5)
        print(f'muscle0Thres: set value from {this_muscle0Thres} to {new_muscle0Thres}')
        editMuscle0Thres.setText(str(f"{new_muscle0Thres:.7f}"))
        slot_muscle0Thres_set()
    btn_paramMuscle0Thres_plus.clicked.connect(slot_muscle0Thres_plus)
    def slot_muscle0Thres_minus():
        this_muscle0Thres = float(editMuscle0Thres.text())
        new_muscle0Thres = round( (this_muscle0Thres - 0.00001), 5)
        print(f'muscle0Thres: set value from {this_muscle0Thres} to {new_muscle0Thres}')
        editMuscle0Thres.setText(str(f"{new_muscle0Thres:.7f}"))
        slot_muscle0Thres_set()
    btn_paramMuscle0Thres_minus.clicked.connect(slot_muscle0Thres_minus)
    
    #Muscle threshold B : Bic
    paramMuscle1Thres = QLabel(f"Bic Thres")
    btn_paramMuscle1Thres_plus  = QPushButton("+")
    btn_paramMuscle1Thres_minus = QPushButton("-")
    btn_paramMuscle1Thres_set   = QPushButton("SET")
    editMuscle1Thres = QLineEdit(f"{sharedConfig['emg_thresh_noStim'][1]:.7f}")
    boxCalibNoEmg.addWidget(paramMuscle1Thres,1,0)
    boxCalibNoEmg.addWidget(editMuscle1Thres,1,1)
    boxCalibNoEmg.addWidget(btn_paramMuscle1Thres_plus,1,2)
    boxCalibNoEmg.addWidget(btn_paramMuscle1Thres_minus,1,3)
    boxCalibNoEmg.addWidget(btn_paramMuscle1Thres_set,1,4)
    
    def slot_muscle1Thres_set():
        new_val = float(editMuscle1Thres.text())
        # can't edit directly nested elements from a manager object (like the dicts)
        curr_values = sharedConfig['emg_thresh_noStim']
        curr_values[1] = new_val
        print(f"new val from gui is {new_val}")
        sharedConfig['emg_thresh_noStim'] = curr_values
        print(f"now in shared dict is its")
        print(sharedConfig['emg_thresh_noStim'])
        # update on graph
        print(f"set {sharedConfig['emg_thresh_noStim'][0]} in shared dict, applying to line5")
        try:
            xaxis = np.array(range(0,previewData_size))
            line5.set_data(xaxis, sharedConfig['emg_thresh_noStim'][1])
            line5.figure.canvas.draw()
        except:
            print("failed")
        print(f"new value is now { sharedConfig['emg_thresh_noStim'][1]}")
    btn_paramMuscle1Thres_set.clicked.connect(slot_muscle1Thres_set)
    def slot_muscle1Thres_plus():
        this_muscle1Thres = float(editMuscle1Thres.text())
        new_muscle1Thres = round( (this_muscle1Thres + 0.00001), 5)
        print(f'stimThres: set value from {this_muscle1Thres} to {new_muscle1Thres}')
        editMuscle1Thres.setText(str(f"{new_muscle1Thres:.7f}"))
        slot_muscle1Thres_set()
    btn_paramMuscle1Thres_plus.clicked.connect(slot_muscle1Thres_plus)
    def slot_muscle1Thres_minus():
        this_muscle1Thres = float(editMuscle1Thres.text())
        new_muscle1Thres = round( (this_muscle1Thres - 0.00001), 5)
        print(f'muscle1Thres: set value from {this_muscle1Thres} to {new_muscle1Thres}')
        editMuscle1Thres.setText(str(f"{new_muscle1Thres:.7f}"))
        slot_muscle1Thres_set()
    btn_paramMuscle1Thres_minus.clicked.connect(slot_muscle1Thres_minus)

    ###
    # Calibration EMG thresholds WITH stim
    ###

    # Create a new groupbox
    groupboxCalibWithEmg = QGroupBox("2) EMG calibration WITH stimulator")
    vbox.addWidget(groupboxCalibWithEmg)
    boxCalibWithEmg = QGridLayout()
    groupboxCalibWithEmg.setLayout(boxCalibWithEmg)
    
    #Stim connected yes/no
    # updated in the matplotlib loop canvas update to check when var changes
    paramStimConnected1 = QLabel(f"Stim connected")
    paramStimConnected2 = QLabel(f"{sharedConfig['stim_connected']}")
    paramStimConnected1.setAlignment(Qt.AlignCenter)
    paramStimConnected2.setAlignment(Qt.AlignCenter)
    paramStimConnected2.setStyleSheet("QLabel{color: red;font-size:12px;font-family:'Orbitron'}")
    boxCalibWithEmg.addWidget(paramStimConnected1,0,0)
    boxCalibWithEmg.addWidget(paramStimConnected2,0,1)
    if sharedConfig['stim_connected']:
        paramStimConnected1.setStyleSheet("QLabel{color: green;font-size:12px;font-family:'Orbitron'}")
        paramStimConnected2.setStyleSheet("QLabel{color: green;font-size:12px;font-family:'Orbitron'}")
    
    #Stim sends pulses ONOFF
    paramStimOnOff1 = QLabel(f"Stim active")
    paramStimOnOff2 = QLabel(f"{sharedConfig['stim_do']}")
    paramStimOnOff2.setStyleSheet("QLabel{color: red;font-size:12px;font-family:'Orbitron'}")
    paramStimOnOff1.setAlignment(Qt.AlignCenter)
    paramStimOnOff2.setAlignment(Qt.AlignCenter)
    btn_paramStimOnOff_on  = QPushButton("ON")
    btn_paramStimOnOff_off = QPushButton("OFF")
    boxCalibWithEmg.addWidget(paramStimOnOff1,1,0)
    boxCalibWithEmg.addWidget(paramStimOnOff2,1,1)
    boxCalibWithEmg.addWidget(btn_paramStimOnOff_on,1,2)
    boxCalibWithEmg.addWidget(btn_paramStimOnOff_off,1,3)
    def slot_paramStimOnOff_on():
        sharedConfig['stim_do'] = True
        paramStimOnOff2.setText(f"{sharedConfig['stim_do']}")
        paramStimOnOff2.setStyleSheet("QLabel{color: green;font-size:12px;font-family:'Orbitron'}")
    btn_paramStimOnOff_on.clicked.connect(slot_paramStimOnOff_on)
    def slot_paramStimOnOff_off():
        sharedConfig['stim_do'] = False
        paramStimOnOff2.setText(f"{sharedConfig['stim_do']}")
        paramStimOnOff2.setStyleSheet("QLabel{color: red;font-size:12px;font-family:'Orbitron'}")
    btn_paramStimOnOff_off.clicked.connect(slot_paramStimOnOff_off)

    # Filter the EMG ON/OFF switch
    filterOn = sharedConfig['emg_filter_do']
    paramFilterOnOff1 = QLabel(f"EMG filters stim")
    paramFilterOnOff2 = QLabel(f"{filterOn}")
    paramFilterOnOff1.setAlignment(Qt.AlignCenter)
    paramFilterOnOff2.setAlignment(Qt.AlignCenter)
    if filterOn:
        paramFilterOnOff2.setStyleSheet("QLabel{color: green;font-size:12px;font-family:'Orbitron'}")
    else:
        paramFilterOnOff2.setStyleSheet("QLabel{color: red;font-size:12px;font-family:'Orbitron'}")
    #paramStimOnOff.setAlignment(Qt.AlignCenter)
    btn_paramFilterOnOff_on  = QPushButton("ON")
    btn_paramFilterOnOff_off = QPushButton("OFF")
    boxCalibWithEmg.addWidget(paramFilterOnOff1,2,0)
    boxCalibWithEmg.addWidget(paramFilterOnOff2,2,1)
    boxCalibWithEmg.addWidget(btn_paramFilterOnOff_on,2,2)
    boxCalibWithEmg.addWidget(btn_paramFilterOnOff_off,2,3)
    def slot_FilterOnOff_on():
        sharedConfig['emg_filter_do'] = True
        paramFilterOnOff2.setText(f"{sharedConfig['emg_filter_do']}")
        paramFilterOnOff2.setStyleSheet("QLabel{color: green;font-size:12px;font-family:'Orbitron'}")
    btn_paramFilterOnOff_on.clicked.connect(slot_FilterOnOff_on)
    def slot_FilterOnOff_off():
        sharedConfig['emg_filter_do'] = False
        paramFilterOnOff2.setText(f"{sharedConfig['emg_filter_do']}")
        paramFilterOnOff2.setStyleSheet("QLabel{color: red;font-size:12px;font-family:'Orbitron'}")
    btn_paramFilterOnOff_off.clicked.connect(slot_FilterOnOff_off)

    #Stim threshold
    paramStimThres = QLabel(f"Stim threshold")
    paramStimThres.setStyleSheet("QLabel {background-color: ;}")
    btn_stimThres_plus  = QPushButton("+")
    btn_stimThres_minus = QPushButton("-")
    btn_stimThres_set   = QPushButton("SET")
    editStimThres = QLineEdit(str( sharedConfig['stim_thresh'] ))
    boxCalibWithEmg.addWidget(paramStimThres,3,0)
    boxCalibWithEmg.addWidget(editStimThres,3,1)
    boxCalibWithEmg.addWidget(btn_stimThres_plus,3,2)
    boxCalibWithEmg.addWidget(btn_stimThres_minus,3,3)
    boxCalibWithEmg.addWidget(btn_stimThres_set,3,4)
    def slot_stimThres_set():
        print(f'stimThres: applying new value to {editStimThres.text()}')
        sharedConfig['stim_thresh'] = float(editStimThres.text())
        print(f"setting {sharedConfig['stim_thresh']} on line 7")
        try:
            xaxis = np.array(range(0,previewData_size))
            line7.set_data(xaxis, sharedConfig['stim_thresh'])
            line7.figure.canvas.draw()
            line8.set_data(xaxis, -sharedConfig['stim_thresh'])
            line8.figure.canvas.draw()
        except:
            print("failed")
        print(f"new value is now { sharedConfig['stim_thresh']}")
    btn_stimThres_set.clicked.connect(slot_stimThres_set)
    def slot_stimThres_plus():
        this_stimThres = float(editStimThres.text())
        new_stimThres = round( (this_stimThres + 0.0001), 5)
        print(f'stimThres: set value from {this_stimThres} to {new_stimThres}')
        editStimThres.setText(str(new_stimThres))
        slot_stimThres_set()
    btn_stimThres_plus.clicked.connect(slot_stimThres_plus)
    def slot_stimThres_minus():
        this_stimThres = float(editStimThres.text())
        new_stimThres = round( (this_stimThres - 0.0001), 5)
        print(f'stimThres: set value from {this_stimThres} to {new_stimThres}')
        editStimThres.setText(str(new_stimThres))
        slot_stimThres_set()
    btn_stimThres_minus.clicked.connect(slot_stimThres_minus)


    # Stim Filter setting Window
    paramStimWindow = QLabel(f"Filter window") # add a text
    paramStimWindow.setStyleSheet("QLabel {background-color: ;}")
    btn_paramStimWindow_plus  = QPushButton("+")
    btn_paramStimWindow_minus = QPushButton("-")
    btn_paramStimWindow_set   = QPushButton("SET")
    editParamStimWindow = QLineEdit(f"{sharedConfig['filter_delWindow']}")
    boxCalibWithEmg.addWidget(paramStimWindow,4,0)
    boxCalibWithEmg.addWidget(editParamStimWindow,4,1)
    boxCalibWithEmg.addWidget(btn_paramStimWindow_plus,4,2)
    boxCalibWithEmg.addWidget(btn_paramStimWindow_minus,4,3)
    boxCalibWithEmg.addWidget(btn_paramStimWindow_set,4,4)
    def slot_StimWindow_set():
        print(editParamStimWindow.text())
        sharedConfig['filter_delWindow'] = int(editParamStimWindow.text())
        print(f"set {sharedConfig['filter_delWindow']} in shared dict")
    btn_paramStimWindow_set.clicked.connect(slot_StimWindow_set)
    def slot_StimWindow_plus():
        this_stimWindow = int(editParamStimWindow.text())
        new_stimWindow  = (this_stimWindow + 1)
        print(f'stimWindow: set value from {this_stimWindow} to {new_stimWindow}')
        editParamStimWindow.setText(str(new_stimWindow))
        slot_StimWindow_set()
    btn_paramStimWindow_plus.clicked.connect(slot_StimWindow_plus)
    def slot_StimWindow_minus():
        this_stimWindow = int(editParamStimWindow.text())
        new_stimWindow  = (this_stimWindow - 1)
        print(f'stimWindow: set value from {this_stimWindow} to {new_stimWindow}')
        editParamStimWindow.setText(str(new_stimWindow))
        slot_StimWindow_set()
    btn_paramStimWindow_minus.clicked.connect(slot_StimWindow_minus)

    # Controller: defines if the pulses sent are defined by the controller or set manually
    controllerOn         = sharedConfig['controller_on']
    paramSetController1 = QLabel("Controller")
    paramSetController2 = QLabel()
    paramSetController1.setStyleSheet("QLabel{color: black;font-size:12px;font-family:'Orbitron'}")
    paramSetController2.setStyleSheet("QLabel{color: black;font-size:12px;font-family:'Orbitron'}")
    if controllerOn:
        paramSetController2.setText(f"auto")
    else:
        paramSetController2.setText(f"manual")
    paramStimOnOff1.setAlignment(Qt.AlignCenter)
    btn_paramSetController_auto   = QPushButton("AUTO")
    btn_paramSetController_manual = QPushButton("MANUAL")
    boxCalibWithEmg.addWidget(paramSetController1,5,0)
    boxCalibWithEmg.addWidget(paramSetController2,5,1)
    boxCalibWithEmg.addWidget(btn_paramSetController_auto,5,2)
    boxCalibWithEmg.addWidget(btn_paramSetController_manual,5,3)
    def slot_SetController_auto():
        sharedConfig['controller_on'] = True
        paramSetController2.setText(f"auto")
    btn_paramSetController_auto.clicked.connect(slot_SetController_auto)
    def slot_SetController_manual():
        sharedConfig['controller_on'] = False
        paramSetController2.setText(f"manual")
        # (re)set the current manual stim value to 0 just in case?
        #sharedData['pulse_intensity_man'] = 0
        #print(f"set pulse_intensity_man value to 0 : {sharedConfig['pulse_intensity_man']}")
        # it is updated by the canvas loop
    btn_paramSetController_manual.clicked.connect(slot_SetController_manual)

    #pulse_intensity
    param1 = QLabel(f"Intensity (<{sharedConfig['stim_max_intensitiy']})")
    param1.setStyleSheet("QLabel {background-color: ;}")
    btn_param1_plus  = QPushButton("+")
    btn_param1_minus = QPushButton("-")
    btn_param1_set   = QPushButton("SET")
    edit1 = QLineEdit(str( sharedConfig['pulse_intensity_man'] ))
    boxCalibWithEmg.addWidget(param1,6,0)
    boxCalibWithEmg.addWidget(edit1,6,1)
    boxCalibWithEmg.addWidget(btn_param1_plus,6,2)
    boxCalibWithEmg.addWidget(btn_param1_minus,6,3)
    boxCalibWithEmg.addWidget(btn_param1_set,6,4)
    def slot_btn_param1_set():
        print(f'param1: pulse_intensity_man: applying new value {edit1.text()}')
        sharedConfig['pulse_intensity_man'] = int(edit1.text())
    btn_param1_set.clicked.connect(slot_btn_param1_set)
    def slot_btn_param1_plus():
        this_val = int(edit1.text())
        new_val  = this_val + 1
        # make sure we don't go over the limit
        if new_val > sharedConfig['stim_max_intensitiy']:
            new_val = sharedConfig['stim_max_intensitiy']
        edit1.setText(str(new_val))
        slot_btn_param1_set()
    btn_param1_plus.clicked.connect(slot_btn_param1_plus)
    def slot_btn_param1_minus():
        this_val = int(edit1.text())
        new_val  = this_val - 1
        # make sure lowest value is 0
        if new_val < 0:
            new_val = 0
        edit1.setText(str(new_val))
        slot_btn_param1_set()
    btn_param1_minus.clicked.connect(slot_btn_param1_minus)

    #pulse_width (us)
    param2 = QLabel(f"Width (µs)")
    param2.setStyleSheet("QLabel {background-color: ;}")
    btn_param2_set   = QPushButton("SET")
    btn_param2_plus  = QPushButton("+")
    btn_param2_minus = QPushButton("-")   
    edit2 = QLineEdit(str( sharedConfig['pulse_width'] ))
    boxCalibWithEmg.addWidget(param2,7,0)
    boxCalibWithEmg.addWidget(edit2,7,1)
    boxCalibWithEmg.addWidget(btn_param2_plus,7,2)
    boxCalibWithEmg.addWidget(btn_param2_minus,7,3)
    boxCalibWithEmg.addWidget(btn_param2_set,7,4)   
    def slot_btn_param2_set():
        print(f'param2: pulse_width: applying new value {edit2.text()}')
        sharedConfig['pulse_width'] = int(edit2.text())
    btn_param2_set.clicked.connect(slot_btn_param2_set)
    def slot_btn_param2_plus():
        this_val = int(edit2.text())
        new_val  = this_val + 1
        edit2.setText(str(new_val))
        slot_btn_param2_set()
    btn_param2_plus.clicked.connect(slot_btn_param2_plus)
    def slot_btn_param2_minus():
        this_val = int(edit2.text())
        new_val  = this_val - 1
        edit2.setText(str(new_val))
        slot_btn_param2_set()
    btn_param2_minus.clicked.connect(slot_btn_param2_minus)

    #pulse_period
    param3 = QLabel(f"Period (ms)")
    param3.setStyleSheet("QLabel {background-color: ;}")
    btn_param3_plus  = QPushButton("+")
    btn_param3_minus = QPushButton("-")
    btn_param3_set   = QPushButton("SET")
    edit3 = QLineEdit(str( sharedConfig['pulse_period'] ))
    boxCalibWithEmg.addWidget(param3,8,0)
    boxCalibWithEmg.addWidget(edit3,8,1)
    boxCalibWithEmg.addWidget(btn_param3_plus,8,2)
    boxCalibWithEmg.addWidget(btn_param3_minus,8,3)
    boxCalibWithEmg.addWidget(btn_param3_set,8,4)
    def slot_btn_param3_set():
        print(f'param2: pulse_period: applying new value {edit3.text()}')
        sharedConfig['pulse_period'] = float(edit3.text())
    btn_param3_set.clicked.connect(slot_btn_param3_set)
    def slot_btn_param3_plus():
        this_val = int(edit3.text())
        new_val  = this_val + 1
        edit3.setText(str(new_val))
        slot_btn_param3_set()
    btn_param3_plus.clicked.connect(slot_btn_param3_plus)
    def slot_btn_param3_minus():
        this_val = int(edit3.text())
        new_val  = this_val - 1
        edit3.setText(str(new_val))
        slot_btn_param3_set()
    btn_param3_minus.clicked.connect(slot_btn_param3_minus)

    #Muscle threshold With stimulation : Delt
    paramMuscle0ThresWithStim = QLabel(f"Delt thres stim") # add a text
    btn_paramMuscle0ThresWithStim_plus  = QPushButton("+")
    btn_paramMuscle0ThresWithStim_minus = QPushButton("-")
    btn_paramMuscle0ThresWithStim_set   = QPushButton("SET")
    editMuscle0ThresWithStim = QLineEdit(f"{sharedConfig['emg_thresh_withStim'][0]:.7f}")
    boxCalibWithEmg.addWidget(paramMuscle0ThresWithStim,9,0)
    boxCalibWithEmg.addWidget(editMuscle0ThresWithStim,9,1)
    boxCalibWithEmg.addWidget(btn_paramMuscle0ThresWithStim_plus,9,2)
    boxCalibWithEmg.addWidget(btn_paramMuscle0ThresWithStim_minus,9,3)
    boxCalibWithEmg.addWidget(btn_paramMuscle0ThresWithStim_set,9,4)
    def slot_muscle0ThresWithStim_set():
        new_val = float(editMuscle0ThresWithStim.text())
        # can't edit directly nested elements from a manager object (like the dicts)
        curr_values = sharedConfig['emg_thresh_withStim']
        curr_values[0] = new_val
        print(f"new val from gui is {new_val}")
        sharedConfig['emg_thresh_withStim'] = curr_values
        print(f"now in shared dict is its")
        print(sharedConfig['emg_thresh_withStim'])
        # update on graph
        print(f"set {sharedConfig['emg_thresh_withStim'][0]} in shared dict, applying to line3")
        try:
            xaxis = np.array(range(0,previewData_size))
            line9.set_data(xaxis, sharedConfig['emg_thresh_withStim'][0])
            line9.figure.canvas.draw()
        except:
            print("failed")
        print(f"new value is now { sharedConfig['emg_thresh_withStim'][0]}")
    btn_paramMuscle0ThresWithStim_set.clicked.connect(slot_muscle0ThresWithStim_set)
    def slot_muscle0ThresWithStim_plus():
        this_muscle0ThresWithStim = float(editMuscle0ThresWithStim.text())
        new_muscle0ThresWithStim = round( (this_muscle0ThresWithStim + 0.00001), 5)
        print(f'muscle0ThresWithStim: set value from {this_muscle0ThresWithStim} to {new_muscle0ThresWithStim}')
        editMuscle0ThresWithStim.setText(str(f"{new_muscle0ThresWithStim:.7f}"))
        slot_muscle0ThresWithStim_set()
    btn_paramMuscle0ThresWithStim_plus.clicked.connect(slot_muscle0ThresWithStim_plus)
    def slot_muscle0ThresWithStim_minus():
        this_muscle0ThresWithStim = float(editMuscle0ThresWithStim.text())
        new_muscle0ThresWithStim = round( (this_muscle0ThresWithStim - 0.00001), 5)
        print(f'muscle0ThresWithStim: set value from {this_muscle0ThresWithStim} to {new_muscle0ThresWithStim}')
        editMuscle0ThresWithStim.setText(str(f"{new_muscle0ThresWithStim:.7f}"))
        slot_muscle0ThresWithStim_set()
    btn_paramMuscle0ThresWithStim_minus.clicked.connect(slot_muscle0ThresWithStim_minus)
    
    #Muscle threshold B : Bic
    paramMuscle1ThresWithStim = QLabel(f"Bic thres stim")
    btn_paramMuscle1ThresWithStim_plus  = QPushButton("+")
    btn_paramMuscle1ThresWithStim_minus = QPushButton("-")
    btn_paramMuscle1ThresWithStim_set   = QPushButton("SET")
    editMuscle1ThresWithStim = QLineEdit(f"{sharedConfig['emg_thresh_withStim'][1]:.7f}")
    boxCalibWithEmg.addWidget(paramMuscle1ThresWithStim,10,0)
    boxCalibWithEmg.addWidget(editMuscle1ThresWithStim,10,1)
    boxCalibWithEmg.addWidget(btn_paramMuscle1ThresWithStim_plus,10,2)
    boxCalibWithEmg.addWidget(btn_paramMuscle1ThresWithStim_minus,10,3)
    boxCalibWithEmg.addWidget(btn_paramMuscle1ThresWithStim_set,10,4)
    def slot_muscle1ThresWithStim_set():
        new_val = float(editMuscle1ThresWithStim.text())
        # can't edit directly nested elements from a manager object (like the dicts)
        curr_values = sharedConfig['emg_thresh_withStim']
        curr_values[1] = new_val
        print(f"new val from gui is {new_val}")
        sharedConfig['emg_thresh_withStim'] = curr_values
        print(f"now in shared dict is its")
        print(sharedConfig['emg_thresh_withStim'])
        # update on graph
        print(f"set {sharedConfig['emg_thresh_withStim'][0]} in shared dict, applying to line5")
        try:
            xaxis = np.array(range(0,previewData_size))
            line10.set_data(xaxis, sharedConfig['emg_thresh_withStim'][1])
            line10.figure.canvas.draw()
        except:
            print("failed")
        print(f"new value is now { sharedConfig['emg_thresh_withStim'][1]}")
    btn_paramMuscle1ThresWithStim_set.clicked.connect(slot_muscle1ThresWithStim_set)
    def slot_muscle1ThresWithStim_plus():
        this_muscle1ThresWithStim = float(editMuscle1ThresWithStim.text())
        new_muscle1ThresWithStim = round( (this_muscle1ThresWithStim + 0.00001), 5)
        print(f'stimThresWithStim: set value from {this_muscle1ThresWithStim} to {new_muscle1ThresWithStim}')
        editMuscle1ThresWithStim.setText(str(f"{new_muscle1ThresWithStim:.7f}"))
        slot_muscle1ThresWithStim_set()
    btn_paramMuscle1ThresWithStim_plus.clicked.connect(slot_muscle1ThresWithStim_plus)
    def slot_muscle1ThresWithStim_minus():
        this_muscle1ThresWithStim = float(editMuscle1ThresWithStim.text())
        new_muscle1ThresWithStim = round( (this_muscle1ThresWithStim - 0.00001), 5)
        print(f'muscle1ThresWithStim: set value from {this_muscle1ThresWithStim} to {new_muscle1ThresWithStim}')
        editMuscle1ThresWithStim.setText(str(f"{new_muscle1ThresWithStim:.7f}"))
        slot_muscle1ThresWithStim_set()
    btn_paramMuscle1ThresWithStim_minus.clicked.connect(slot_muscle1ThresWithStim_minus)

    ###
    # Record
    ###

    # Create a new groupbox
    groupboxStart = QGroupBox("3) Start")
    vbox.addWidget(groupboxStart)
    boxStart = QGridLayout()
    groupboxStart.setLayout(boxStart)
    
    #Stim connected yes/no
    # updated in the matplotlib loop canvas update to check when var changes
    paramStartLabel = QLabel(f"Everything is ready, you can save the settings\
                             \nDon't forget to set pulse to ON, emg filter to ON, and controller to AUTO...\
                             \nand you can press 'RECORD'")
    paramStartLabel.setStyleSheet("QLabel{color: blue;font-size:12px;font-family:'Orbitron'}")
    boxStart.addWidget(paramStartLabel,0,0)

    # Record button
    buttonRecord = QPushButton("START recording")
    boxStart.addWidget(buttonRecord,1,0)

    def slot_RecordButton():
        
        # get the current iteration of each process
        current_iters = [sharedConfig['emg_iter'], sharedConfig['filter_iter'], sharedConfig['rms_iter']]
        # get current list
        this_list = sharedConfig['recordButton']
        # append current states
        this_list.append(current_iters)
        # put in shared dict
        sharedConfig['recordButton'] = this_list   
        
        # just tidy up the color of the button to provide feedback on if recording or not
        global buttonRecord_status
        if sharedConfig['buttonRecord_status']:
            buttonRecord.setStyleSheet("background-color: green")
            buttonRecord.setText(str(f"Currently not recording: press to start"))
            sharedConfig['buttonRecord_status'] = False
        else:
            buttonRecord.setStyleSheet("background-color: red")
            buttonRecord.setText(str(f"Currently recording: press to stop"))
            sharedConfig['buttonRecord_status'] = True

    buttonRecord.clicked.connect(slot_RecordButton)

    ###########################################################################
    # Live plot
    ###########################################################################

    # add a canvas for matplotlib
    from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
    from matplotlib.figure import Figure
    import numpy as np
    import time
    from scipy import interpolate

    if verbose: print("creating dynamic canvas")
    # Create the dynamic canvas
    dynamic_canvas = FigureCanvas(Figure(figsize=(10, 10)))
    dynamic_ax     = dynamic_canvas.figure.subplots()

    # Config variables
    #heightGraph          = 0.0012 # 0.014           # defined with the buttons on left layout already; 120 # max range of EMG data is 11 mV # zoom

    numberPointsToUpdateGraph  = 50                  # disregarded : we get the whole available array. if we just update the last x values we might miss frames on the plot!
    previewData_size           = 6000                # number of points to plot in the graph (we take x latest values in corresponding emg array)

    # Initialise our arrays
    global previewData_DELT
    global previewData_BIC
    previewData_DELT = np.zeros(previewData_size)    # list with complete data before reduction
    previewData_BIC  = np.zeros(previewData_size)

    # Prepare data
    xaxis     = np.array(range(0,previewData_size))
    data_zero = np.array([0] * previewData_size)     # data to plot, start with 0
    line1,    = dynamic_ax.plot(xaxis, data_zero , color='red',   label='Emg_0 DELT'   )
    line2,    = dynamic_ax.plot(xaxis, data_zero , color='blue',  label='Emg_1 BIC'    )
    line3,    = dynamic_ax.plot(xaxis,  [  sharedConfig['emg_thresh_noStim'][0] ] * previewData_size, color='red',  linestyle='dashdot', label='DELT thresh')
    line5,    = dynamic_ax.plot(xaxis,  [  sharedConfig['emg_thresh_noStim'][1] ] * previewData_size, color='blue', linestyle='dashdot', label='BIC thresh' )
    #line4,    = dynamic_ax.plot(xaxis, data_zero , color='red',   label='DELT_max'     )
    #line6,    = dynamic_ax.plot(xaxis, data_zero , color='blue',  label='BIC_max'      )
    line7,    = dynamic_ax.plot(xaxis,  [ +sharedConfig['stim_thresh'] ] * previewData_size , color='black', label='Stim Filter Thresh' )
    line8,    = dynamic_ax.plot(xaxis,  [ -sharedConfig['stim_thresh'] ] * previewData_size , color='black')
    line9,    = dynamic_ax.plot(xaxis,  [  sharedConfig['emg_thresh_withStim'][0] ] * previewData_size , color='red',  linestyle='dotted', label='DELT thresh stim')
    line10,   = dynamic_ax.plot(xaxis,  [  sharedConfig['emg_thresh_withStim'][1] ] * previewData_size , color='blue', linestyle='dotted', label='BIC thresh stim')

    # Pretty up the graph
    dynamic_ax.legend(loc='upper right')                # add the legends to the graph
    dynamic_ax.set_ylim(bottom = -sharedConfig['gui_zoom'], top = +sharedConfig['gui_zoom'])
    dynamic_ax.set_xlim(left=0, right=previewData_size)

    # discarded: variables to keep count of the frames gotten from shared queue and displayed on graph
    # discarded: we discarded the blocking queue and moved on to a manual timing/refreshrate to query the array!
    # this work as plot refresh rate is always < emg refresh rate
    global frameNumberPlot
    queryNumberPlot = 0 # number of times the plot looked for new data in the queue
    frameNumberPlot = 0 # number of times a new frame has been read by plot from queue

    def update_canvas():
                
        global frameNumberPlot
        global timer
        global previewData_DELT
        global previewData_BIC
        global streamTimeElapsed

        ###
        # Update statuses and paramters
        ###
        
        # check if stim connected
        if sharedConfig['stim_connected']:
            paramStimConnected2.setText(f"{sharedConfig['stim_connected']}")
            paramStimConnected2.setStyleSheet("QLabel{color: green;font-size:20px;font-family:'Orbitron'}")
        # also edit the text saying if one muscle or the other is active : biofeedback
        if sharedConfig['active_delt']:
            delt_activ_label2.setText(f"{sharedConfig['active_delt']}")
            delt_activ_label2.setStyleSheet("QLabel{color: green;font-size:20px;font-family:'Orbitron'}")
        else:
            delt_activ_label2.setText(f"{sharedConfig['active_delt']}")
            delt_activ_label2.setStyleSheet("QLabel{color: red;font-size:20px;font-family:'Orbitron'}")
        if sharedConfig['active_bic']:
            bic_activ_label2.setText(f"{sharedConfig['active_bic']}")
            bic_activ_label2.setStyleSheet("QLabel{color: green;font-size:20px;font-family:'Orbitron'}")
        else:
            bic_activ_label2.setText(f"{sharedConfig['active_bic']}")
            bic_activ_label2.setStyleSheet("QLabel{color: red;font-size:20px;font-family:'Orbitron'}")
        
        # and the controller value + pulse intensity
        controller_value_label2.setText(f"{sharedConfig['controller_value']}")
        if sharedConfig['controller_on']:
            # if auto, the current pulse_intensity is max_allowed_intensity * sharedData['controller_value']
            new_pulse_intensity = sharedConfig['controller_value'] * sharedConfig['stim_max_intensitiy']
            pulse_intensity_label2.setText(f"{new_pulse_intensity:.3f}")
        else:
            # if mode manual, it is set manually in the editbox
            pulse_intensity_label2.setText(f"{sharedConfig['pulse_intensity_man']}")

        # refresh rate of the plot
        if timer.interval != sharedConfig['gui_refresh_delay']:
            timer.interval=(sharedConfig['gui_refresh_delay'])

        # update query counter
        frameNumberPlot+=1
        if verbose: print(f'updating canvas ({frameNumberPlot})')

        ###
        # Getting EMG data
        ###

        # Here, we have to decide which data to get depending on what is asked in the gui : raw/rms/filtered
        # we have already initialised previewData_BIC, now we want to replace it by taking the latest frames

        # special case before emg has started streaming (takes typically 2 seconds)
        # we have initialised the plot with 0s
        #if len(sharedData['emg_raw_delt']) >= previewData_size:
        # edit : that would cause a delay to display 
        # options could be to
            # a) concat what we get from sharedData to previewData_DELT, and remove len(previewData) values from the beginning of previewData_DELT
            # b) add the number of known frames into a the preview array
            # c) initialise the shared ones with 0s as well
            # --> going with a)

        dataToAddToPreview_DELT = None
        dataToAddToPreview_BIC  = None

        # get the selected type
        if verbose: print(f"Asking for previewtype {sharedConfig['gui_preview_type']}")
        if sharedConfig['gui_preview_type'] == 'raw':
            
            if  debug: print(f"getting raw filtered")
            dataToAddToPreview_DELT = sharedData['emg_delt_raw']
            dataToAddToPreview_BIC  = sharedData['emg_bic_raw']

        elif sharedConfig['gui_preview_type'] == 'filt':
            
            if  debug: print(f"getting filtered data")
            dataToAddToPreview_DELT = sharedData['emg_delt_filt']
            dataToAddToPreview_BIC  = sharedData['emg_bic_filt']

        elif sharedConfig['gui_preview_type'] == 'rms':
            
            if  debug: print(f"getting rms")
            dataToAddToPreview_DELT = sharedData['emg_delt_rms']
            dataToAddToPreview_BIC  = sharedData['emg_bic_rms']

        if verbose: print(f"\nPlot: added {len(dataToAddToPreview_DELT)} values of {sharedConfig['gui_preview_type']}")
        
        # if it's at least the size of what data we want to preview, we take it
        
        if len(dataToAddToPreview_DELT) >= previewData_size:
            previewData_DELT = dataToAddToPreview_DELT
        # otherwise, we add 0s behind
        else:
            previewData_DELT = np.concatenate( (np.zeros(previewData_size-len(dataToAddToPreview_DELT)),dataToAddToPreview_DELT), axis=0 )
        if len(dataToAddToPreview_BIC) >= previewData_size:
            previewData_BIC = dataToAddToPreview_BIC
        # otherwise, we add 0s behind
        else:
            previewData_BIC  = np.concatenate( (np.zeros(previewData_size-len(dataToAddToPreview_BIC)), dataToAddToPreview_BIC),  axis=0 )
        """
        # we concatenate this to the previewData
        previewData_DELT = np.concatenate( (previewData_DELT,dataToAddToPreview_DELT), axis=0 )
        previewData_BIC  = np.concatenate( (previewData_BIC, dataToAddToPreview_BIC),  axis=0 )
        
        # we remove length of frame from beginning of array
        previewData_DELT = previewData_DELT[len(dataToAddToPreview_DELT):]
        previewData_BIC  = previewData_BIC[len(dataToAddToPreview_BIC):]
        """

        ###
        # Update graph with this current data
        ###

        behind_iters   = sharedConfig['emg_iter']        - sharedConfig['rms_iter']        # how many iterations behind is the filter from the raw delsys
        behind_lengths = len(sharedData['emg_delt_raw']) - len(sharedData['emg_delt_rms']) # how many batches of data behind is the filter from the raw delsys
        newTitle = f"iters: plot {frameNumberPlot}, emg: {sharedConfig['emg_iter']}, filter: {sharedConfig['filter_iter']}, rms: {sharedConfig['rms_iter']} ⇒ behind_iters {behind_iters}"
        dynamic_ax.set_title(newTitle)        

        ###
        # Feedback on keeping up
        ###
        
        if sharedConfig['streamTimeStart']:
            streamTimeElapsed = round( (time.time() - sharedConfig['streamTimeStart']), 2)
        
        if behind_iters>1:
            label_updateKeepup.setStyleSheet("QLabel{color: red;font-size:20px;font-family:'Orbitron'}")
            label_updateKeepup.setText(f"Keeping up: RMS is behind RAW by {behind_iters} batches (t = {streamTimeElapsed}s)")
        else:
            label_updateKeepup.setStyleSheet("QLabel{color: green;font-size:20px;font-family:'Orbitron'}")
            label_updateKeepup.setText(f"Keeping up: RMS is up to date with RAW (t = { str(streamTimeElapsed).zfill(2)}s )")

        ###
        # Plot
        ###

        # Downsample
        if sharedConfig['gui_downsample'] > 0:

            timesHalf = sharedConfig['gui_downsample'] # number of time to reduce data by half

            # emg0
            line1.set_data(downSampleData(xaxis,times=timesHalf), downSampleData(previewData_DELT,times=timesHalf))
            line1.figure.canvas.draw()
            # emg1
            line2.set_data(downSampleData(xaxis,times=timesHalf), downSampleData(previewData_BIC,times=timesHalf))
            line2.figure.canvas.draw()            

        # or full sample
        else:
            # emg0
            line1.set_data(xaxis, previewData_DELT)
            line1.figure.canvas.draw()
            # emg1
            line2.set_data(xaxis, previewData_BIC)
            line2.figure.canvas.draw()
        
        return

    # place the canvas on the widget/window and display
    topRightLayout.addWidget(dynamic_canvas)
    # and in topright layout (the live plot), add a row
    label_updateKeepup = QLabel(f"Keeping up:")
    label_updateKeepup.setAlignment(Qt.AlignCenter)
    topRightLayout.addWidget(label_updateKeepup)
    # define the sanity check (is filter behind raw data get?) based on a) number of iterations, and b) length of arrays
    # 2do: add exception for RMS that changes the number of points
    global behind_iters
    global behind_lengths
    behind_iters   = 0
    behind_lengths = 0
    
    global streamTimeElapsed
    streamTimeElapsed = 0

    # Set the timer (loop of auto-update of the graph) in ms
    global timer
    timer = dynamic_canvas.new_timer(interval=sharedConfig['gui_refresh_delay']) # get with timer.interval # update with timer.interval=(3)
    timer.add_callback(update_canvas)
    timer.start()

    ###
    # Finalise and display
    ###
    
    # Keyboard shortcuts to close the graph
    def slot_closeFromKeyboard():
        print("keyboard shortcut to close the window")
        w.close() # close the main widget
    for key in ["Ctrl+w","Ctrl+q","Escape","q"]:
        QShortcut(QKeySequence(key), w).activated.connect(slot_closeFromKeyboard) # w.close()
    
    centerPoint = QDesktopWidget().availableGeometry().center()
    
    w.show()    
    #w.resize(1600,900) # fixed size  
    w.resize(1200,600) # fixed size  
    w.activateWindow() # bring window to front
    #w.moveCenter(centerPoint)
    #w.showMaximized()
    #w.raise_()
    app.exec()         # create the window (required for multithreading)

if __name__ == "__main__":
    
    print("orlau_live_plot (GUI) loaded as main")
    
