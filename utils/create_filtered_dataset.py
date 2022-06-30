"""

Code for automating the filtering of AE signal by removing "double" events
and cycling through all the waveforms for checking.

Upon running this file, user will be prompted to select experiment folder. The
experiment folder must contains "waves" and "times" files, and must be named in
the format that corresponds to how the script pulls metadata from the file name
string. DOUBLE CHECK THIS! 

Code will save filtered .json file in the selected experiment folder.

Additionally, code will output .txt files detailing filtered / rejected events.

Kiran

Updated: 2022-06-29

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

def auto_filter_ae(ae_file, time_path, channel_num, max_reverb_time,
                   sig_length, data_directory):
    """
    
    Function reads in all waveforms, removes all the waveforms which occur too
    close together, and returns an array of the filtered waveforms. Outputs in
    working directory a .txt file of rejected indices.
    
    Parameters
    ----------
    ae_file : str
        File path to .txt file containing waveforms.
    time_path : str
        File path to .txt file containing events wrt to time.
    channel_num : int
        Channel to be read in, indexed from 0 (Channel 1).
    max_reverb_time : float
        Criterion for filtering. Events within max_reverb_time are filtered.
    sig_length : int
        Number of samples in waveform event.

    Returns
    -------
    filteredSignals : array-like
        List of filtered waves. Each index points to a wave of sig_length.
    filteredEvents : array-like
        List of indices of filtered events.

    """
    # Load in events and times (times .txt file created at AE computer)
    time_path = data_directory + '/' + time_path
    ae_file = data_directory + '/'+ ae_file
    df = pd.read_csv(time_path, sep = '\t')
    time = np.array(df['Time'])
    ev = np.array(df['Event No.'])
    
    # Read in all waves
    print(ae_file)
    signals, _ = read_ae_file(ae_file, channel_num, sig_length)
    signals = np.array(signals) 
    
    numEvents = ev.shape[0]
    filteredEvents = [1]    # indices of filtered events
    rejectedEvents = []     # indices of rejected events
    
    # Loop through events and remove "double" / "triple" events 
    for i in range (1, numEvents): 
        if (time[i] - time[i-1] >= max_reverb_time): # events sufficiently far
            filteredEvents.append(ev[i])
        else: # events occur close together
            rejectedEvents.append(ev[i])
     
    # Keep filtered waveforms        
    filteredSignals = signals[np.array(filteredEvents)-1]
    
    # Document filtered / rejected events in .txt
    with open(time_path[0:time_path.find('times')] + 
              'rejectedEvents.txt', 'w') as f:
        intString = [str(x) for x in rejectedEvents]
        intString = ' '.join(intString)
        writeString = 'The auto-filtered events are events #: ' + intString
        f.write(writeString)
    
    return filteredSignals, filteredEvents

def read_ae_file(fname, channel_num, sig_length=1024):
    """
    
    Function loads in .txt waveform files generated from proprietary AE 
    softwares. Returns waveforms with event numbers.

    Parameters
    ----------
    fname : str
        File path of .txt file containing all waveforms (voltage over time).
    channel_num : int
        Channel to be read in, indexed from 0 (Channel 1).
    sig_length : int, optional
        Number of samples in waveform event. The default is 1024.

    Returns
    -------
    signals : array-like
        List of waveform signals.
    ev : array-like
        List of event numbers, indexed from 1 (first event in test).

    """
    f = open(fname)
    lines = f.readlines()[1:]
    channel = np.array([
        float(line.split()[channel_num]) for line in lines])
    f.close()
    signals = []
    for i in range(0,len(channel),sig_length):
        signals.append(channel[i:i+sig_length])
    ev = np.arange(len(signals))+1
    
    return signals, ev

def wave_viz_filter(signals, sig_length, dt, time_path, events):
    """
    
    Loop through waveforms, visualize, user indicates whether to keep or to
    get rid of.

    Parameters
    ----------
    signals : array-like
        List of waveform signals.
    sig_length : int
        Number of samples in waveform event.
    dt : float
        Time between samples. Also, 1 / sampling frequency.
    time_path : str
        File path to .txt file containing events wrt to time.

    Returns
    -------
    filteredSignals : array-like
        List of filtered waves. Each index points to a wave of sig_length.

    """
    # Plotting parameters
    duration = sig_length*dt*10**6
    time = np.linspace(0,duration, sig_length)
    
    # Track events filtered
    rejectedEvents = []
    keptEvents = []
    rejectedIndices = []
    
    # Loop through all waveforms 
    for i in range(signals.shape[0]):
        plt.plot(time,signals[i])
        plt.ylabel('Amplitude')
        plt.xlabel('Time (us)')
        plt.xlim([0,duration])
        plt.show()
        inp = input("Keep? (Y/N)")
        if (inp == 'N' or inp == 'n'):
            rejectedEvents.append(events[i])
            rejectedIndices.append(i)
        else:
            keptEvents.append(events[i])
    print(events)
    print(keptEvents)
    print(rejectedEvents)
    # Get filtered signals using the user-indicated indices
    for j in range(len(rejectedIndices)):
        signals = np.delete(signals, (rejectedIndices[j]), axis = 0)
    
    # Document filtered / rejected events in .txt
    with open(time_path[0:time_path.find('times')] + 
              'rejectedEvents.txt', 'a') as f:
        intString = [str(x) for x in rejectedEvents]
        intString = ' '.join(intString)
        writeString = '\nThe manually filtered events are events #: ' + \
            intString
        f.write(writeString)
        intString = [str(x) for x in keptEvents]
        intString = ' '.join(intString)
        writeString = '\nThe remaining events are events #: ' + \
            intString
        f.write(writeString)
        
    return signals, keptEvents

def flatten(t): # flattens out a list of lists (CHECK)
    return [item for sublist in t for item in sublist]

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

    Returns
    -------
    None.

    """
    # User selects directory to load in data from
    root = Tk()
    root.filename = filedialog.askdirectory(title = "Select folder")
    root.destroy()
    data_directory = root.filename
    
    # Pull data files from specified directory
    os.chdir(data_directory) 
    files = [f for f in os.listdir(data_directory) 
                 if isfile(join(data_directory, f))] 
    
    # Separate files into raw and filter type
    time_files = [f for f in files if 'times' in f]
    raw_files = [f for f in files if '_waves' in f]
    
    # Raw AE data and metadata
    waves = []
    angle = []
    location = []
    length = []
    event = []
    
    for idx, _ in enumerate(raw_files):
        
        # Get raw data file and corresponding time file
        raw = raw_files[idx]
        times = time_files[idx]
        
        # Get filtered waveforms
        v0, ev = auto_filter_ae(ae_file = raw, time_path = times, 
                            channel_num=0, max_reverb_time =.02,
                            sig_length = 1024, data_directory = data_directory)
        
        # User filters by visually inspecting
        v0, ev = wave_viz_filter(v0, 1024, 10**-7, times, ev)
        
        # Append filtered wave with corresponding event number
        waves.append(v0.tolist())
        ev = [int(item) for item in ev]
        event.append(ev)

        # Get metadata from file name, which needs to follow a specific format
        angle.append([raw_files[idx][7:12] for i in range(len(v0))])
        length.append([raw_files[idx][13:16] for i in range(len(v0))])
        location.append([raw_files[idx][17:21] for i in range(len(v0))])
    
    # Remove a dimension 
    waves = flatten(waves)
    event = flatten(event)
    angle = flatten(angle)
    location = flatten(location)
    length = flatten(length) 
    
    # Create dataset in appropriate folder
    dataset = {'waves':waves, 'event': event, 
               'angle':angle, 'location':location,
               'length':length}
    
    os.chdir(data_directory)
    
    # Get json file name from selected folder
    dataset_name = os.path.basename(os.path.normpath(data_directory))
    now = datetime.datetime.now()
    time_stamp = str(now.strftime("%Y%m%d_"))
    dataset_name = time_stamp + dataset_name + ".json"
    
    with open(dataset_name, "w") as outfile:
        json.dump(dataset, outfile)
        
if __name__ == '__main__':
    make_data_set()
    