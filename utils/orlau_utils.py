# -*- coding: utf-8 -*-

from termcolor import colored
import colorama
colorama.init()
import numpy as np
import scipy.interpolate as interpolate

def showTitle(text, color="blue"):
    n = len(text) + 4
    horizontal = '#' * n
    print("\n##############################\n# {}\n##############################\n".format(colored(text, color, attrs=['bold']),))

def printColor(text, color="blue"):
    print("{}".format(colored(text, color, attrs=['bold']),))
    
def show(data, size=4):
    print(np.around(data,size))
    
def convert_to_rms(muscle_raw, analogFrequency=2000, power=2, highFreq=10, lowFreq=400, window=200):

    """
    analogFreq = 2000 # default is 2000 on our delsys
    power      = 2    # 2
    high       = 10   # 10
    low        = 400  # 450
    window     = 200  # 200
    """

    def window_rms(a, window_size):
        a2 = np.power(a,power)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, 'valid'))

    # Bandpass
    #high                 = highFreq / (0.5*analogFrequency) # have to divide by "frequency/2" so 0.5*acq.GetAnalogFrequency()
    #low                  = lowFreq  / (0.5*analogFrequency)
    #b, a                 = scipySignal.butter(4, [high,low], btype='bandpass')
    #result_bandpass      = scipySignal.filtfilt(b, a, muscle_raw)
    
    # Rectification
    result_rectification = abs(muscle_raw)
    result_rms           = window_rms(result_rectification, window)
    
    return result_rms

def time_normalise(data, length=100):

    arr_ref                = np.empty((1,length,))
    arr_ref[:]             = np.nan
    arr2                   = data
    arr2_interp            = interpolate.interp1d(np.arange(arr2.size),arr2)
    time_normalised_array  = arr2_interp(np.linspace(0,arr2.size-1,arr_ref.size))
    
    return time_normalised_array
