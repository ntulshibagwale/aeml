"""

create_filtered_dataset

Executable code for automating the filtering of AE signal by removing "double" 
events and cycling through all the waveforms for checking.

Upon running this file, user will be prompted to select experiment folder. The
experiment folder must contains "waves" and "times" files, and must be named in
the format that corresponds to how the script pulls metadata from the file name
string. DOUBLE CHECK THIS! 

Code will save filtered .json file in the selected experiment folder.

Additionally, code will output .txt files detailing filtered / rejected events.

Kiran Lochun

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
import json
from tkinter import filedialog
from tkinter import *
import datetime
import sys 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Boiler plate code to ensure scripts can import code from aemlfunctions.
# Adds project directory to python path.
proj_dir = os.path.dirname(os.path.abspath(os.curdir))
if sys.path[0] != proj_dir:
    sys.path.insert(0, proj_dir)
print(f"Project directory: {proj_dir}")  
from waves.misc import flatten
from waves.visuals import create_figure, plot_signal
from waves.load_data import read_ae_file

def auto_filter_ae(ae_file, time_file, max_reverb_time, data_directory):
    """
    
    Function reads in all waveforms, removes all the waveforms which occur too
    close together, and returns an array of the filtered waveforms. Outputs in
    working directory a .txt file of rejected indices.
    
    Parameters
    ----------
    ae_file : str
        File path of .txt file containing all waveforms (voltage over time)
    time_file : str
        File path to .txt file containing events wrt to time.
    max_reverb_time : float
        Criterion for filtering. Events within max_reverb_time are filtered.
    data_directory : str
        Directory selected by user which contains "_times" and "_waves" .txt's.

    Returns
    -------
    filtered_signals : array-like
        Each index of list points to list of waveform signals from a sensor.
        These are the filtered waveforms. It is important to use in conjunction
        with filtered_event to determine event #.
        i.e filtered_signals[0] will return all the filtered AE hits from 
        sensor 1. Note, that filtered_signalswaves[0][5] will return the 6th
        filtered signal (with sig_length size), whose event # is given by 
        filtered_events[5]. Not necessarily the 6th wave from original test.
    filtered_events : array-like
        List of event numbers corresponding to filtered events.
    fs : int
        Sampling frequency. Typically 10**7 Hz or 10 MHz with Digital Wave.
    channel_num : int
        Number of channels / AE sensors used.
    sig_length : int
        Number of samples in waveform event / hit. 
        
    """
    # Load in events and times (times .txt file created at AE computer)
    time_file = data_directory + '/' + time_file
    ae_file = data_directory + '/' + ae_file
    df = pd.read_csv(time_file, sep = '\t')
    time = np.array(df['Time'])
    ev = np.array(df['Event No.'])
    
    # Read in all waves along with signal processing parameters
    print(ae_file)
    signals, _, fs, channel_num, sig_length = read_ae_file(ae_file)
     
    num_events = ev.shape[0]
    filtered_events = [1]    # indices of filtered events
    rejected_events = []     # indices of rejected events
    
    # Loop through events and remove "double" / "triple" events 
    for i in range (1, num_events): 
        if (time[i] - time[i-1] >= max_reverb_time): # events sufficiently far
            filtered_events.append(ev[i])
        else: # events occur too close together
            rejected_events.append(ev[i])
     
    # Keep filtered waveforms   
    filtered_signals = []
    for channel in signals:
        channel = np.array(channel)
        channel = channel[np.array(filtered_events)-1]
        filtered_signals.append(channel)
    
    # Document filtered / rejected events in .txt
    with open(time_file[0:time_file.find('times')] + 
              'rejected_events.txt', 'w') as f:
        int_string = [str(x) for x in rejected_events]
        int_string = ' '.join(int_string)
        write_string = 'The auto-filtered events are events #: ' + int_string
        f.write(write_string)
    
    return filtered_signals, filtered_events, fs, channel_num, sig_length

def wave_viz_filter(signals, ev, time_file, dt, sig_length):
    """
    
    Loop through waveforms, visualize, user indicates whether to keep or to
    get rid of.

    Parameters
    ----------
    signals : array-like
        Each index of list points to list of waveform signals from a sensor.
    ev : array-like
        List of event numbers of waveforms associated with signals variable.
        All sensors trigger at same time, so events are equivalent in time. 
    time_file : str
        File path to .txt file containing events wrt to time.
    dt : float
        Time between samples. Computed by 1 / sampling frequency (fs).
    sig_length : int
        Number of samples in waveform event.
        
    Returns
    -------
    filtered_signals : array-like
        List of filtered waves. Each index points to a wave of sig_length.
    filtered_events : array-like
        List of event numbers corresponding to filtered events.
        
    """
    # Track events filtered
    rejected_events = []
    filtered_events = []
    filtered_indices = []
    
    # Loop through all waveforms 
    for i in range(len(ev)): # iterate through each event     
           
        # Plot signal from each sensor 
        fig,spec2 = create_figure(f'Event: {ev[i]}',
                                  columns=1,rows=len(signals),
                                  width=15,w_space=0.6, h_space=0.5,
                                  height=10, axes_font_size=15,
                                  tick_font_size=14,
                                  suptitle_font_size=20,
                                  legend_font_size=14)
        
        for j in range(len(signals)): # Iterate through each channel
            # if (hanning):
            #     signals[j][i] = signals[j][i]*np.hanning(sig_length)
            ax = fig.add_subplot(spec2[j,0])
            ax = plot_signal(ax, signals[j][i], dt, sig_length)
            plt.title(f'Channel {j+1}')
           
        plt.show()
        inp = input("Keep? (Y/N)")
        if (inp == 'N' or inp == 'n'):
            rejected_events.append(ev[i])
        else:
            filtered_events.append(ev[i])
            filtered_indices.append(i)

    # Keep filtered waveforms using the user-indicated indices 
    filtered_signals = []
    for channel in signals: # loop through each sensor
        filtered_signals.append(channel[filtered_indices])
      
    # Document filtered / rejected events in .txt
    with open(time_file[0:time_file.find('times')] + 
              'rejected_events.txt', 'a') as f:
        int_string = [str(x) for x in rejected_events]
        int_string = ' '.join(int_string)
        write_string = '\nThe manually filtered events are events #: ' + \
            int_string
        f.write(write_string)
        int_string = [str(x) for x in filtered_events]
        int_string = ' '.join(int_string)
        write_string = '\nThe remaining events are events #: ' + \
            int_string
        f.write(write_string)
        
    return filtered_signals, filtered_events

def make_data_set():
    """
        
    The function loads in ALL the files from provided data directory, then
    performs the automatic filtering operation (checking closeness of events),
    then prompts the user to select which waveforms should stay and which ones
    should go (The Clash). 
    
    Each filtered waveform will then be paired with metadata extracted from
    the file name. i.e. angle = 'xxdeg', location = 'tope', length = '2in'
    
    A dataset is filed in the user selected folder alongside with a .json file
    that stores the filtered waveform and metadata. File is given today's date.
    
    Parameters
    ----------
    None.

    Returns
    -------
    None. However, a .json file is created in specified data directory.

    """
    # User selects directory to load in data from
    root = Tk()
    root.filename = filedialog.askdirectory(
        title = "Select folder containing AE data")
    root.destroy()
    data_directory = root.filename
    
    # Pull data files from specified directory
    os.chdir(data_directory) 
    files = [f for f in os.listdir(data_directory) 
                 if isfile(join(data_directory, f))] 
    
    # Separate files into raw and filter type
    time_files = [f for f in files if '_times' in f]
    raw_files = [f for f in files if '_waves' in f]
    
    # inp = input("Apply Hanning window? (Y/N)")
    # if (inp == 'Y' or inp == 'y'):
    #     hanning = True
    # else:
    #     hanning = False
    
    # Raw AE data and metadata
    waves = []     # voltage signal
    angle = []     # angle at which plb was done
    location = []  # source type
    length = []    # boundary conditions
    event = []     # event number from AE experiment
    sensor = []    # which sensor it came from
    distance = []  # distance wrt to middle

    # Loop through each pair of raw and time files
    for idx, _ in enumerate(raw_files):
        
        # Get raw AE data file and corresponding time file
        raw = raw_files[idx]
        times = time_files[idx]
        
        # Get time-criteria filtered waveforms and signal processing parameters
        
        signals, ev, fs, channel_num, sig_length = auto_filter_ae(
                            ae_file = raw, time_file = times, 
                            max_reverb_time =.02,
                            data_directory = data_directory)
        
        # User filters remaining waveforms by visually inspecting
        signals, ev = wave_viz_filter(signals, ev, times, dt = 1/fs,
                                      sig_length = sig_length)
        
        # Append filtered waves wrt channel with corresponding event number
        for channel_idx,channel in enumerate(signals):
            channel = channel.tolist() # convert np array to list
            waves.append(channel)
            event.append([int(item) for item in ev])
            # Get metadata from file name, separated by underscore
            # NOTE: Need to figure out a better way for this!
            meta = raw_files[idx].split('_') # split into array of strings
            angle.append([meta[1] for i in range(len(ev))])
            length.append([meta[2] for i in range(len(ev))])
            location.append([meta[3] for i in range(len(ev))])
            distance.append([meta[4] for i in range(len(ev))])
            sensor.append([(channel_idx+1) for i in range(len(ev))])
    
    # Remove a dimension 
    waves = flatten(waves)
    event = flatten(event)
    angle = flatten(angle)
    location = flatten(location)
    length = flatten(length) 
    sensor = flatten(sensor)
    distance = flatten(distance)

    # Create dataset in appropriate folder
    dataset = {'waves' : waves,
               'event' : event, 
               'angle' : angle,
               'location' : location,
               'length' : length,
               'sensor' : sensor,
               'distance' : distance}
    
    os.chdir(data_directory)
    
    # Get json file name from selected folder
    dataset_name = os.path.basename(os.path.normpath(data_directory))
    now = datetime.datetime.now()
    time_stamp = str(now.strftime("%Y%m%d_"))
    # if (hanning):
    #     hAddendum = '_H'
    # else:
    #     hAddendum = '_noH'
    dataset_name = time_stamp + dataset_name + ".json"
    
    os.chdir('..')
    os.chdir('..')
    os.chdir('./filtered')
    with open(dataset_name, "w") as outfile:
        json.dump(dataset, outfile)
        
if __name__ == '__main__':
    make_data_set()
    