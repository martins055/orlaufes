# -*- coding: utf-8 -*-
'''
full delsys SDK:       http://www.delsys.com/integration/sdk/
fork of fork based on: https://github.com/axopy/pytrigno
'''
import sys, os, copy, time
CURR_DIR = os.path.dirname(os.path.realpath(__file__))     # get current directory variable
os.chdir(os.path.abspath(os.path.dirname(__file__)))       # change current working directory
sys.path += [CURR_DIR]                                     # add relative folders to path, to load our modules easily without installing them

import socket
import struct
import numpy
import numpy as np
import time
from orlau_utils import *

class _BaseTrignoDaq(object):
    """
    Delsys Trigno wireless EMG system.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    host : str
        IP address the TCU server is running on.
    cmd_port : int
        Port of TCU command messages.
    data_port : int
        Port of TCU data access.
    rate : int
        Sampling rate of the data source.
    total_channels : int
        Total number of channels supported by the device.
    timeout : float
        Number of seconds before socket returns a timeout exception

    Attributes
    ----------
    BYTES_PER_CHANNEL : int
        Number of bytes per sample per channel. EMG and accelerometer data
    CMD_TERM : str
        Command string termination.

    Notes
    -----
    Implementation details can be found in the Delsys SDK reference:
    http://www.delsys.com/integration/sdk/
    """

    BYTES_PER_CHANNEL = 4
    CMD_TERM = '\r\n\r\n'
    RESPONSE = ''

    def __init__(self, host, cmd_port, data_port, total_channels, timeout):
        self.host           = host           # the ip address on which Delsys TCU is running
        self.cmd_port       = cmd_port       # the command port is 50040 by default
        self.data_port      = data_port      # emg = 50043 ; imu = 50044
        self.total_channels = total_channels # emg = 16    ; imu = 144
        self.timeout        = timeout        # default 10 seconds

        self._min_recv_size = self.total_channels * self.BYTES_PER_CHANNEL # make data accessible once chunk of data has been received (blocking method)

        self._initialize()   # start socket connection to TCU

    def _initialize(self):

        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)

    def start(self):
        """
        Tell the device to begin streaming data.

        You should call ``read()`` soon after this, though the device typically
        takes about two seconds to send back the first batch of data.
        """
        self._send_cmd('START')

    def check_sensor_n_type(self, n):
        """
        Check the type of connected sensor
        """
        cmd = 'SENSOR '+str(n)+' TYPE?'
        self._send_cmd(cmd)

    def check_sensor_n_mode(self, n):
        """
        Check the mode of connected sensor
        """
        cmd = 'SENSOR '+str(n)+' MODE?'
        self._send_cmd(cmd)

    def set_sensor_n_mode(self, n):
        """
        SET the type of connected sensor
		Use command “SENSOR n SETMODE y” to set the mode the given sensor.
		from the SDK
        """
        cmd = 'SENSOR '+str(n)+' SETMODE 39' # we want mode 39 to stream EMG and IMU at the same time on regular avanti 1 emg sensors
        self._send_cmd(cmd)

    def command_quit(self):
      self._send_cmd('STOP')

    def become_master(self):
        """
        make sure we are master, to prevent getting blocked by the TCU if a connection is already ongoing: take over
        """
        self._send_cmd('MASTER')

    def check_sensor_n_start_index(self, n):
        """
        Check the type of connected sensor
        """
        cmd = 'SENSOR ' + str(n) + ' STARTINDEX?'
        self._send_cmd(cmd)

    def check_sensor_n_auxchannel_count(self, n):
        """
        Check the type of connected sensor
        """
        cmd = 'SENSOR ' + str(n) + ' AUXCHANNELCOUNT?'
        self._send_cmd(cmd)

    def check_sensor_channel_unit(self, n, m):
        """
        Check the unit of the channels
        """
        cmd = 'SENSOR ' + str(n) + ' CHANNEL '+str(m)+' UNITS?'
        self._send_cmd(cmd)

    def read(self, num_samples):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Parameters
        ----------
        num_samples : int
            Number of samples to read per channel.

        Returns
        -------
        data : ndarray, shape=(total_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        l_des = num_samples * self._min_recv_size
        l = 0
        packet = bytes()
        while l < l_des:
            try:
                packet += self._data_socket.recv(l_des - l)
            except socket.timeout:
                l = len(packet)
                packet += b'\x00' * (l_des - l)
                raise IOError("Device disconnected.")
            l = len(packet)

        data = numpy.asarray(
            struct.unpack('<'+'f'*self.total_channels*num_samples, packet))
        data = numpy.transpose(data.reshape((-1, self.total_channels)))

        return data

    def stop(self):
        """Tell the device to stop streaming data."""
        self._send_cmd('STOP')

    def quit(self):
        """Use command “QUIT” to stop data collection and close the server session"""
        self._send_cmd('QUIT')
        
    def reset(self):
        """Restart the connection to the Trigno Control Utility server."""
        self._initialize()

    def __del__(self):
        try:
            self._comm_socket.close()
        except:
            pass

    def _send_cmd(self, command):
        self._comm_socket.send(self._cmd(command))
        resp = self._comm_socket.recv(128)
        print("Response from trigno:: ", str(resp, 'utf-8'))
        self._validate(resp)

    @staticmethod
    def _cmd(command):
        return bytes("{}{}".format(command, _BaseTrignoDaq.CMD_TERM),
                     encoding='ascii')

    @staticmethod
    def _validate(response):
        s = str(response)
        if 'OK' not in s:
            print("warning: TrignoDaq command failed: {}".format(s))


class TrignoEMG(_BaseTrignoDaq):
    """
    Delsys Trigno wireless EMG system EMG data.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channel_range : tuple with 2 ints
        Sensor channels to use, e.g. (lowchan, highchan) obtains data from
        channels lowchan through highchan. Each sensor has a single EMG
        channel.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    units : {'V', 'mV', 'normalized'}, optional
        Units in which to return data. If 'V', the data is returned in its
        un-scaled form (volts). If 'mV', the data is scaled to millivolt level.
        If 'normalized', the data is scaled by its maximum level so that its
        range is [-1, 1].
    host : str, optional
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional
        Port of TCU command messages.
    data_port : int, optional
        Port of TCU EMG data access. By default, 50041 is used, but it is
        configurable through the TCU graphical user interface.
    timeout : float, optional
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    rate : int
        Sampling rate in Hz.
    scaler : float
        Multiplicative scaling factor to convert the signals to the desired
        units.
    """

    def __init__(self, channel_range, samples_per_read, units='V',
                 host='localhost', cmd_port=50040, data_port=50043, timeout=10):
        super(TrignoEMG, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port,
            total_channels=16, timeout=timeout)

        self.channel_range = channel_range
        self.samples_per_read = samples_per_read

        self.rate = 2000

        self.scaler = 1.
        if units == 'mV':
            self.scaler = 1000.
        elif units == 'normalized':
            # max range of EMG data is 11 mV
            self.scaler = 1 / 0.011

    def set_channel_range(self, channel_range):
        """
        Sets the number of channels to read from the device.

        Parameters
        ----------
        channel_range : tuple
            Sensor channels to use (lowchan, highchan).
        """
        self.channel_range = channel_range
        self.num_channels = channel_range[1] - channel_range[0] + 1

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(TrignoEMG, self).read(self.samples_per_read)
        data = data[self.channel_range[0]:self.channel_range[1]+1, :]
        return self.scaler * data


class TrignoAccel(_BaseTrignoDaq):
    """
    Delsys Trigno wireless EMG system accelerometer data.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channel_range : tuple with 2 ints
        Sensor channels to use, e.g. (lowchan, highchan) obtains data from
        channels lowchan through highchan. Each sensor has three accelerometer
        channels.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    host : str, optional
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional
        Port of TCU command messages.
    data_port : int, optional
        Port of TCU accelerometer data access. By default, 50042 is used, but
        it is configurable through the TCU graphical user interface.
    timeout : float, optional
        Number of seconds before socket returns a timeout exception.
    """
    def __init__(self, channel_range, samples_per_read, host='localhost',
                 cmd_port=50040, data_port=50044, timeout=10):
        super(TrignoAccel, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port,
            total_channels=144, timeout=timeout)

        self.channel_range = channel_range
        self.samples_per_read = samples_per_read

        self.rate = 148.1

    def set_channel_range(self, channel_range):
        """
        Sets the number of channels to read from the device.

        Parameters
        ----------
        channel_range : tuple
            Sensor channels to use (lowchan, highchan).
        """
        self.channel_range = channel_range
        self.num_channels = channel_range[1] - channel_range[0] + 1

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(TrignoAccel, self).read(self.samples_per_read)
        data = data[self.channel_range[0]:self.channel_range[1]+1, :]
        return data

"""
def init_system(config_delsys):

    # initialise TCP/IP socket with TCU,and prepare to receive data
    emg = TrignoEMG(channel_range=(0, 7),    samples_per_read=2000,host='127.0.0.1')
    
    # now that we have initialised the socket, send command to put devices in the right mode
    emg.become_master() # make sure we take priority on other existing
    
    # we count the number of muscles that we are using, and set them in the right mode
    for i in range(len(config_delsys['muscles_names'])):
        emg.set_sensor_n_mode(i+1) # Starts at 1.
    
    # finally, open an additional stream for the IMU data
    imu = TrignoAccel(channel_range=(0, 35), samples_per_read=250, host='127.0.0.1')
    
    # return the two objects to main script. We will call emg.read() or imu.read() to stream data.
    return emg, imu
"""

def emg_connect(verbose=True, samples_per_read=300, previewSize=6000):
    
    """
    previewSize = 10000 # number of points to plot in the preview graph
    samples_per_read = 200  # number of frames to request from the emg (1 frame is sent every 13ms)
    """
    
    connectedEMG = False
    devEMG       = None
  
    i=1
    while not connectedEMG:

        printColor("connecting to EMG try {}".format(i), color='white')       
        try:

            devEMG = TrignoEMG(channel_range=(0, 16), samples_per_read=samples_per_read, host='127.0.0.1', timeout=10)
            connectedEMG = True

        except:
            printColor('Unable to connect to TCU, make sure it is running', color='red')
            i+=1
        time.sleep(3)

    if verbose: printColor('Connected to EMG', color='green')
    devEMG.become_master()
    devEMG.start()
    return devEMG

def emg_calibrate(previewSize=10000, numberFrames=200, sizeGraph=100, emg_calib_duration=20, verbose=True):
    """
    Connect to TCU and record x seconds of emg
    Then filters (RMS) and ask if calibration contains baseline and max activity
    
    previewSize        = number of points to plot in the preview graph
    numberFrames       = number of frames to request from the emg (1 frame is sent every 13ms)
    sizeGraph          = reduced data for preview
    emg_calib_duration = recording time (rest and max contraction for all muscles)
    """

    if verbose: printColor('EMG calibration starting to EMG', color='green')

    # Connect to TCU (open a socket and gain access)
    devEMG = emg_connect(numberFrames=numberFrames, previewSize=previewSize)
    
    # Start streaming data (call .read() to get latest frame sent every 13ms)
    # TCU typically takes ~2 seconds before starting streaming
    devEMG.start()
    
    # Data
    delt_EMG = np.array([]) # where the whole recording is stored
    bic_EMG  = np.array([])

    # Define a timer for the loop
    t_end = time.time() + emg_calib_duration
    i=1

    # Get frames in a loop for x seconds
    while time.time() < t_end:
        
        print("We are iteration {} ({}/{} seconds remaining)".format(i, int(t_end - time.time()), emg_calib_duration))
        
        # Get EMG data
        frame = devEMG.read() # get a new frame (of numberFrames as defined in connection)
        delt_EMG = np.concatenate( (delt_EMG, frame[0]), axis=0)
        bic_EMG  = np.concatenate( (bic_EMG,  frame[1]), axis=0)
        
        i+=1
        
    # Disconnect gracefully
    devEMG.stop()         # stop streaming
    devEMG.command_quit() # close socket connection with TCU
    if verbose: printColor('EMG calibration finished', color='green')
    
    return delt_EMG, bic_EMG

def time_normalise(data, length=100):
    
    arr_ref                = np.empty((1,length,))
    arr_ref[:]             = np.nan
    arr2                   = data
    arr2_interp            = scipy.interpolate.interp1d(np.arange(arr2.size),arr2)
    time_normalised_array  = arr2_interp(np.linspace(0,arr2.size-1,arr_ref.size))

    return time_normalised_array


def window_rms(a, window_size, power=2):
    
    a2     = np.power(a,power)
    window = np.ones(window_size)/float(window_size)
    result = np.sqrt(np.convolve(a2, window, 'valid')) 
    
    return result

def minmax(data, windowTime=1, frequency=2000, verbose=False):
    # cut recording of each emg in small time parts
    # our system is 2000 hz, so 1 second is 2000 frame and this is the default setting
    
    list_rms_minmax = []                   # list that contains the RMS of each 1 second cut
    
    windowTime      = windowTime           # time window to divide our calibration data (seconds)
    windowFrames    = windowTime*frequency # 2000hz
    
    i               = 0                    # start the frame number at 0
    iter_nb         = 0                    # start iteration count at 0
    while i+windowFrames <= len(data):
        
        if verbose: print("doing iteration number {}: frames {} to {}".format(iter, i, i+windowFrames))
        
        this_chunk  = data[i:i+windowFrames]          # get the next second of data (i -> i+2000)
        this_rms    = np.sqrt(np.mean(this_chunk**2)) # get the rms of this chunk of data
        list_rms_minmax.append(this_rms)              # add to our global list
        
        i+=windowFrames                               # add 2000 frames or 1 second for next iteration
        iter_nb+=1                                    # increment iteration count
    
    # lowest is baseline
    baseline_value = min(list_rms_minmax) # the chunk of data with the lowest rms is this one
    baseline_time  = list_rms_minmax.index(baseline_value)# and is situated at this index in the list (therefore at the time windowTime*index)
    
    # max is max
    max_value      = max(list_rms_minmax)
    max_time       = list_rms_minmax.index(max_value)

    return baseline_value, max_value

if __name__ == "__main__":
    
    showTitle('EMG module started as main script. Connecting and running calibration.')
    delt_EMG, bic_EMG = emg_calibrate(verbose=True, emg_calib_duration=5)

    # Baseline and max for each emg   
    bic_baseline,  bic_max  = minmax(bic_EMG)
    delt_baseline, delt_max = minmax(delt_EMG)
    
    ###
    # Plot the raw data
    ##
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    #ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # Plot some data on the axes.
    #ows, cols = delt_EMG.shape
    x = list(range(len(delt_EMG)))
    y0 = delt_EMG
    y1 = bic_EMG

    ax.plot(x, y0, label='delt', color='blue')
    ax.plot(x, y1, label='bic', color='green')
    plt.xlabel('Frames')
    plt.ylabel('EMG in ?v')

    ax.plot(x, np.repeat(bic_baseline,  len(x)), label='bic_base',  color='red')
    ax.plot(x, np.repeat(bic_max,       len(x)), label='bic_max',   color='black')
    ax.plot(x, np.repeat(delt_baseline, len(x)), label='delt_base', color='pink')
    ax.plot(x, np.repeat(delt_max,      len(x)), label='delt_max',  color='yellow')    
    
    plt.legend()
    
