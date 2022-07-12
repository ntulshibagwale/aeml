"""

Code for retrieving trained model results and json datasets from github.

Nick Tulshibagwale

"""
from waves.ml.model_architectures import NeuralNetwork_01, NeuralNetwork_02
import torch
import json, requests
import numpy as np
import os
from os.path import isfile, join
from tkinter import filedialog
from tkinter import *

def read_ae_file(ae_file):
    """
    
    Function loads in experimental AE data from .txt files generated from
    Digital Wave software. 
    
    Parameters
    ----------
    ae_file : str
        File path of .txt file containing all waveforms (voltage over time).
    
    Returns
    -------
    signals : array-like
        Each index of list points to list of waveform signals from a sensor.
        i.e signals[0] will return all the AE hits from sensor 1, and 
        signals[0][0] will return the 1st waveform from sensor 1 (with size
        sig_length).
    ev : array-like
        List of event numbers, indexed from 1 (first event in test). All 
        sensors trigger at same time, so events are equivalent in time.
    fs : int
        Sampling frequency. Typically 10**7 Hz or 10 MHz with Digital Wave.
    channel_num : int
        Number of channels / AE sensors used.
    sig_length : int
        Number of samples in waveform event / hit. 

    """
    # Read in .txt file generated from Digital Wave DAQ / Software
    f = open(ae_file)
    data = f.readlines()
    f.close()
    
    # Get the signal processing parameters from header
    header = data[0]
    fs = int(header.split()[0]) * 10**6  # DAQ sampling freq (usually 10 MHz)
    sig_length = int(header.split()[2])  # Number of samples in waveform event
    channel_num = int(header.split()[3]) # Number of AE sensors used
    
    # Read in waveform data and turn into list of sensors pointing to AE hits
    lines = data[1:]
    signals = [] 
    
    # Loop through the columns taken from .txt file (waves from each sensor)
    for channel in range(0,channel_num):
        # Get data from the sensor's respective column
        v = np.array([float(line.split()[channel]) for line in lines])
        # Turn the long appended column into separate AE hits using sample num
        z = []
        for i in range(0,len(v),sig_length):
            z.append(v[i:i+sig_length])    
        signals.append(z)
    
    # Create array of corresponding event numbers 
    ev = np.arange(len(signals[0]))+1 # all sensors have same number of events

    return signals, ev, fs, channel_num, sig_length

def get_ml_dataset_paths():
    """
    
    User selects path to ml dataset folder and function returns paths to the
    contains .json files.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    train_path : str
        Path to .json file containing training data in selected ml folder.
    valid_path : str
        Path to .json file containing validation data in selected ml folder.
    test_path : str
        Path to .json file containing test data in selected ml folder.

    """   
    # User selects directory to load in data from
    root = Tk()
    root.filename = filedialog.askdirectory(
        title = "Select folder containing ml dataset")
    root.destroy()
    data_directory = root.filename
    
    # Pull data files from specified directory
    os.chdir(data_directory) 
    files = [f for f in os.listdir(data_directory) 
                 if isfile(join(data_directory, f))] 
    
    # Separate files into raw and filter type
    train_path = [f for f in files if '_train' in f]
    valid_path = [f for f in files if '_valid' in f]
    test_path =  [f for f in files if '_test' in f]
    
    train_path = data_directory + '/' + train_path[0]
    valid_path = data_directory + '/' + valid_path[0]
    test_path = data_directory + '/' + test_path[0]
    
    print(train_path)
    print(valid_path)
    print(test_path)
    
    return train_path, valid_path, test_path

def load_json_file_from_path(path):
    """
    
    Given .json file path, loads in the file and returns the dictionary.

    Parameters
    ----------
    path : str
        Path name the PLB_data.json file is located at. 

    Returns
    -------
    data : dict
        Returns raw wave data along with experimental metadata, such as 
        location, angle, distance, etc. Depends on experiment.

    """
    with open(path) as json_file:
        data = json.load(json_file)

    for key in data.keys():
        data[key]  = np.array(data[key])
        
    return data

def load_trained_model_from_path(pth_path, model_num, feature_dim,
                                 num_classes):
    """
    
    Loads up a trained model from the file path of the pth file. Need to 
    specify details on the model architecture.

    Parameters
    ----------
    pth_path : str
        File path to pth, ex: './experiment_01/10_3000_0.001_adam_mse.pth'.
    model_num : int
        Specific model to load up, refer to model architectures for types.
    feature_dim : int
        Dimension of input, can be acquired from associated pickle file.
    num_classes : int, optional
        Dimension of output, associated with classification. The default is 5.

    Returns
    -------
    model : torch object
        Loaded model with trained parameters acquired from pth file.

    """
    if model_num == 1:   # classification
        model = NeuralNetwork_01(feature_dim,num_classes)
    elif model_num == 2: # regression
        model = NeuralNetwork_02(feature_dim)
        
    model.load_state_dict(torch.load(pth_path))
    
    print(model)
    
    return model

def load_json_file_from_github(url):
    """
    
    Given url of a .json file on a github repository. This function can load 
    in the dataset as a dictionary.

    Parameters
    ----------
    url : string, optional
        URL to json file on github repository. For example, 
        'https://raw.githubusercontent.com/ntulshibagwale/aeml/master/data/
        220426_PLB_data.json'.

    Returns
    -------
    dataset : dict
        Dataset.

    """
    resp = requests.get(url)
    dataset = json.loads(resp.text)
    for key in dataset.keys():
        dataset[key]  = np.array(dataset[key])
        
    return dataset

if __name__ == "__main__":
    get_ml_dataset_paths()

    
