"""

create_ml_datasets

Used for creating training, validation, test datasets from filtered data.

Kiran Lochun

"""
import sys
import os
from os.path import isfile, join
import numpy as np
from tkinter import filedialog
from tkinter import *
import json
import random
import math
import datetime
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Boiler plate code to ensure scripts can import code from waves package.
# Adds project directory to python path.
proj_dir = os.path.dirname(os.path.abspath(os.curdir))
if sys.path[0] != proj_dir:
    sys.path.insert(0, proj_dir)
print(f"Project directory: {proj_dir}")  

from waves.misc import flatten
from waves.load_data import load_json_file_from_path
from waves.misc import dict_to_list
from waves.visuals import create_figure

def splitDataset(trainProp, crossProp):
    
    # Select filtered .json files to load in
    # Each .json file contains waveforms and metadata from a given experiment
    # that was filtered.
    root = Tk()
    root.filename = filedialog.askopenfilenames(
        title = "Select .json files for creation of ML datasets")
    root.destroy()
    files = root.filename
    
    # Merge the filtered datasets into single dictionary
    ml_dataset = {}
    for path in files:
        data = load_json_file_from_path(path)
        if not ml_dataset.keys(): # if empty, create keys
            for key in data.keys():
                ml_dataset[key] = data[key]
        else: # if not empty, append to existing keys
            for key in data.keys():
                ml_dataset[key] = np.concatenate((ml_dataset[key],data[key]))
    
    # Dictionary containing indices for each dataset    
    partition = {'train': [], 'valid': [], 'test': []}
    
    num_examples = len(ml_dataset['waves'])
    indexList = list(range(0,num_examples)) # index all examples
    partition['random_seed'] = 2
    random.seed(partition['random_seed']) # RNG Seed
    random.shuffle(indexList)
    trainEnd = math.floor(num_examples * trainProp)
    crossEnd = trainEnd + (math.floor(num_examples * crossProp))
    partition['train'] = indexList[0:trainEnd]
    partition['valid'] = indexList[trainEnd:crossEnd]
    partition['test'] = indexList[crossEnd:]
    partition['files'] = files
    
    # Select output folder (create a folder in ml_datasets)
    root = Tk()
    root.filename = filedialog.askdirectory(
        title = "Select folder to save files")
    root.destroy()
    data_directory = root.filename
    dataset_name = os.path.basename(os.path.normpath(data_directory))
    now = datetime.datetime.now()
    time_stamp = str(now.strftime("%Y%m%d_"))
    dataset_name = time_stamp + dataset_name
    os.chdir(data_directory)
        
    # Create dictionaries for partitioned data in prep for .json file
    train = dict.fromkeys(ml_dataset)
    valid = dict.fromkeys(ml_dataset)
    test =  dict.fromkeys(ml_dataset)
    
    ml_dataset = dict_to_list(ml_dataset) # convert to list for indexing
    
    for idx, key in enumerate(train.keys()): # loop thru keys (waves, ev, etc)
        train[key] = ml_dataset[idx][partition['train']].tolist()
        valid[key] = ml_dataset[idx][partition['valid']].tolist()
        test[key]  = ml_dataset[idx][partition['test']].tolist()

    chartProps = ['location', 'distance']
    for a, prop in enumerate(chartProps):
        vals = set(train[prop])
        trainCount = np.zeros(len(vals))
        crossCount = np.zeros(len(vals))
        testCount = np.zeros(len(vals))
        results = {
        'Train': countCategories(vals, trainCount, train, prop).astype(int),
        'Cross': countCategories(vals, crossCount, valid, prop).astype(int),
        'Test': countCategories(vals, testCount, test, prop).astype(int),
        }
        plotDist(results, list(vals), prop, data_directory)
    # File output, save to selected folder
    # Datasets, parameter .txt file, and parameter .json file
    # Plot of distributions (to be done)
    with open(dataset_name + '_train.json', "w") as outfile:
        json.dump(train, outfile)

    with open(dataset_name + '_valid.json', "w") as outfile:
        json.dump(valid, outfile)

    with open(dataset_name + '_test.json', "w") as outfile:
        json.dump(test, outfile)

    # Document distribution of indices in .txt
    with open(dataset_name + '_params.txt', 'w') as f:

        f.write('The loaded in files are: \n' )
        for file in files:
            f.write(str(file))
            f.write('\n')
            
        f.write('\nRandom seed: \n' )
        f.write(str(partition['random_seed']))
        f.write('\n')
        
        f.write('\nTrain indices: \n')        
        f.write(' '.join([str(x) for x in partition['train']]))
        f.write('\n')
        
        f.write('\nValid indices: \n')        
        f.write(' '.join([str(x) for x in partition['valid']]))
        f.write('\n')

        f.write('\nTest indices: \n')        
        f.write(' '.join([str(x) for x in partition['test']]))
        f.write('\n')
        
def countCategories(vals, countArray, data, prop):
    for x in range(len(data[prop])):
        for a, i in enumerate(vals):
            if (data[prop][x] == i):
                countArray[a] = countArray[a]+1        
    return countArray

def plotDist(results, category_names, prop, data_directory):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    
    cmap = plt.get_cmap('plasma')
    color_values = np.linspace(0.15, 0.85, data.shape[1])
    category_colors = cmap(color_values)

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.6 else 'black'
        ax.bar_label(rects, label_type='center', color=text_color, 
                     fontsize = 16)
    
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='medium')
    ax.set_title('Distribution of ' + prop.capitalize() + 's', loc = 'right',
                 fontsize =18)
    fig.savefig(data_directory + '/' + prop + '_dist')
    
if __name__ == '__main__':
    
    # Input train and valid proportions
    # Test proportion is whatever's left.
    train_proportion = 0.5
    valid_proportion = 0.2
    # test_proportion = 0.3
    splitDataset(train_proportion,valid_proportion) 
    