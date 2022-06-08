"""

Script for creating .json dataset file with filtered AE data and metadata. The
.json file is convenient for ML applications and general data processing.

Script has very specific requirements. The data directory must ONLY contain
waveform files and filter files. The waveform files must be named in the format that corresponds to how the script pulls metadata from the file name string. 

There can be no other files in this directory.

Updated: 2022-06-08

"""
import os
from os.path import isfile, join
import json
from .ae_measure2 import filter_ae
from .ae_functions import flatten

def make_data_set(
        data_directory='E:/file_cabinet/phd/projects/aeml/data/natfreq/dataset_files',
        write_directory='E:/file_cabinet/phd/projects/aeml/data/natfreq/',
        dataset_name='220608_natfreqdataset.json'):
    
    # Pull data files from specified directory
    os.chdir(data_directory) 
    files = [f for f in os.listdir(data_directory) 
                 if isfile(join(data_directory, f))] 
    
    # Separate files into raw and filter type
    filter_files = [f for f in files if 'filter' in f]
    raw_files = [f for f in files if 'wave' in f]
    
    # Raw AE data and metadata
    waves = []
    angle = []
    location = []
    length = []
    
    for idx, _ in enumerate(raw_files): 
        
        # Get raw data file and corresponding filter file
        raw = raw_files[idx]
        filter = filter_files[idx]
        
        # Get filtered waveforms
        v0, ev = filter_ae(raw, filter, channel_num=0)
        waves.append(v0.tolist())
        
        # Get metadata from file name, which needs to follow a format
        angle.append([raw_files[idx][7:12] for i in range(len(v0))])
        location.append([raw_files[idx][13:16] for i in range(len(v0))])
        length.append([raw_files[idx][17:20] for i in range(len(v0))])
    
    # Remove a dimension
    waves = flatten(waves)
    angle = flatten(angle)
    location = flatten(location)
    length = flatten(length) 
    
    # Create dataset in appropriate folder
    dataset = {'waves':waves, 'angle':angle, 'location':location,
               'length':length}
    os.chdir(write_directory)
    with open(dataset_name, "w") as outfile:
        json.dump(dataset, outfile)
