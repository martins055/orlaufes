# -*- coding: utf-8 -*-
##############################################
# Controller
##############################################

"""
Controller function

# Inputs:
    Processed deltoid EMG: delt_EMG
    Threshold for deltoid EMG: thres_delt_EMG
    Processed biceps EMG: bic_EMG
    Threshold for biceps EMG: thres_bic_EMG
    Current triceps stimulation value (between 0 and 1): tric_stim

# Output:
    New triceps stimulation value (between 0 and 1): new_tric_stim

# Parameter:
    Rate of change of triceps stimulation: lambda
    If lambda was 0.5,  the stimulation would go from 0 to 1 in about 10 steps
    If lambda was 0.05, the stimulation would go from 0 to 1 in about 100 steps

# Find desired triceps stimulation value based on deltoid and biceps EMG:
    IF delt_EMG>thres_delt_EMG AND bic_EMG<thres_bic_EMG
    THEN des_tric_stim = 1
    ELSE des_tric_stim = 0

# Find the new triceps stimulation value so the stimulation ramps towards the desired stimulation
    new_tric_stim = tric_stim  - lambda*( tric_stim - des_tric_stim )
"""

def controller(delt_EMG, thres_delt_EMG, bic_EMG, thres_bic_EMG, tric_stim, stim_lambda=0.5, verbose=False):

    des_tric_stim = None

    if (delt_EMG > thres_delt_EMG) and (bic_EMG < thres_bic_EMG):
        des_tric_stim = 1

    else:
        des_tric_stim = 0

    new_tric_stim = tric_stim - stim_lambda*(tric_stim - des_tric_stim)


    return new_tric_stim
