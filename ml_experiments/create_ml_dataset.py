"""

create_ml_datasets

Used for creating training, validation, test datasets from filtered data.

Kiran Lochun

"""
import sys
import os
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

from waves.load_data import load_json_file_from_path
from waves.misc import dict_to_list

def splitDataset(trainProp, crossProp, filterProp = '', recursing = False, 
                 indexList = [], data_directory = '', ml_dataset = {}, a= 0,
                 partition = {}):
    """
    
    Splits json file(s) into training, cross validation, and test sets. Can 
    split along a single property (e.x. sensors, location).

    Parameters
    ----------
    trainProp : float
        Proportion of data that will be used for training.
    crossProp : float
        Proportion of data that will be used for cross-validation.
    Note: 1 - trainProp - crossProp is the proportion of data that will be 
    used for testing.
    filterProp (optional): string
        Property that will be separated by. Only done with 'sensor' so far,
        others may be buggy.
        
    """
    # Select filtered .json files to load in
    # Each .json file contains waveforms and metadata from a given experiment
    # that was filtered.
    if (not recursing):
        root = Tk()
        root.filename = filedialog.askopenfilenames(
            title = "Select .json files for creation of ML datasets")
        root.destroy()
        files = root.filename
        
        # Merge the filtered datasets into single dictionary
        for path in files:
            data = load_json_file_from_path(path)
            if not ml_dataset.keys(): # if empty, create keys
                for key in data.keys():
                    ml_dataset[key] = data[key]
            else: # if not empty, append to existing keys
                for key in data.keys():
                    ml_dataset[key] = np.concatenate((ml_dataset[key],
                                                      data[key]))
        
        # Dictionary containing indices for each train, valid, test dataset  
        num_examples = len(ml_dataset['waves'])
        indexList = list(range(0,num_examples)) # index all examples
        partition = {'train': [], 'valid': [], 'test': []}
        partition['random_seed'] = 2
        partition['files'] = files
        random.seed(partition['random_seed']) # RNG Seed
        if (filterProp != ''):
            indexList = indexByProp(filterProp, ml_dataset,
                                    list(set(ml_dataset[filterProp])))
        
        # Select output folder (create a folder in ml_datasets to save dataset)
        root = Tk()
        root.filename = filedialog.askdirectory(
            title = "Select folder to save files")
        root.destroy()
        data_directory = root.filename
        
    # Begin recursion, if appropriate
    if (filterProp != '' and not recursing):
        for a, subList in enumerate(indexList):
            splitDataset(trainProp[a], crossProp[a],filterProp, True, subList,
                         data_directory, ml_dataset, a, partition)
            
    # Where the magic happens
    if ((filterProp != '' and recursing) or 
        (filterProp == '' and not recursing)):
        if (filterProp == '' and not recursing):
            trainProp = trainProp[0]
            crossProp = crossProp[0]
        else:
            partition['train'] = []
            partition['valid'] = []
            partition['test'] = []
        
        # Create dictionaries for partitioned data in prep for .json file
        num_examples = len(indexList)
        train = dict.fromkeys(ml_dataset)
        valid = dict.fromkeys(ml_dataset)
        test =  dict.fromkeys(ml_dataset)
        ml_dataset = dict_to_list(ml_dataset) # set as list for easy indexing
        trainEnd = math.floor(num_examples * trainProp)
        crossEnd = trainEnd + (math.floor(num_examples * crossProp))
        if (trainProp != 0):
            partition['train'] = indexList[0:trainEnd]
        if (crossProp != 0):
            partition['valid'] = indexList[trainEnd:crossEnd]
        if (1-(trainProp + crossProp) != 0):
            partition['test'] = indexList[crossEnd:]
        for idx, key in enumerate(train.keys()): # loop thru keys (waves, etc)
            train[key] = ml_dataset[idx][partition['train']].tolist()
            valid[key] = ml_dataset[idx][partition['valid']].tolist()
            test[key]  = ml_dataset[idx][partition['test']].tolist()
    
        # Visualize distribution of data in datasets
        chartProps = ['location', 'distance'] # Properties to be plotted
        for prop in chartProps:
            vals = set(test[prop]) # IMPORTANT!!! If the test set is ever empty, the program will crash because of this line.
            trainCount = np.zeros(len(vals))
            crossCount = np.zeros(len(vals))
            testCount = np.zeros(len(vals))
            results = {
            'Train':countCategories(vals, trainCount, train, prop).astype(int), 
            'Cross':countCategories(vals, crossCount, valid, prop).astype(int),
            'Test': countCategories(vals, testCount, test, prop).astype(int),
            }
            if (recursing):
                plotDist(results, list(vals), prop, data_directory, a)
            else:
                plotDist(results, list(vals), prop, data_directory)
                
        # File output, save to selected folder
        # Datasets, parameter .txt file, and parameter .json file
        # Plot of distributions (to be done)
        dataset_name = os.path.basename(os.path.normpath(data_directory))
        now = datetime.datetime.now()
        time_stamp = str(now.strftime("%Y%m%d_"))
        dataset_name = time_stamp + dataset_name
        if (recursing):
            dataset_name += "_0" + str(a)
        os.chdir(data_directory)
        if (trainProp != 0): # NOTE; can we make it so that function throws an error
        # if any proportion is zero instead of multiple if statements?
            with open(dataset_name + '_train.json', "w") as outfile:
                json.dump(train, outfile)
        if (crossProp != 0):
            with open(dataset_name + '_valid.json', "w") as outfile:
                json.dump(valid, outfile)
        if (1-(trainProp + crossProp) != 0):
            with open(dataset_name + '_test.json', "w") as outfile:
                json.dump(test, outfile)
    
        # Output .txt with loaded in files and indices
        with open(dataset_name + '_params.txt', 'w') as f:
            f.write('The loaded in files are: \n' )
            for file in partition['files'],:
                f.write(str(file))
                f.write('\n') 
            f.write('\nRandom seed: \n' )
            f.write(str(partition['random_seed']))
            f.write('\n')
            if (trainProp != 0):
                f.write('\nTrain indices: \n')        
                f.write(' '.join([str(x) for x in partition['train']]))
                f.write('\n')
            if (crossProp != 0):
                f.write('\nValid indices: \n')        
                f.write(' '.join([str(x) for x in partition['valid']]))
                f.write('\n')
            if (1-(trainProp + crossProp) != 0):
                f.write('\nTest indices: \n')        
                f.write(' '.join([str(x) for x in partition['test']]))
                f.write('\n')
        
def countCategories(vals, countArray, data, prop):
    """
    
    Helper function that counts instances of a specific property in a dataset
    i.e. the number of waveforms that were broken at the front.

    Parameters
    ----------
    vals : set
        Set of values of the property (e.x. tope, front, backcorner, or 
        sensors 1,2,3, & 4)
    countArray : array-like
        Where the counts are stored. Has the same length as vals.
    data : dictionary
        Where data is stored, imported from filtered .json files.
    prop: string
        Property to count, e.x. 'sensor', 'location'
    Returns
    ----------
    countArray: array-like
        Described above.
        
    """
    for x in range(len(data[prop])):
        for a, i in enumerate(vals):
            if (data[prop][x] == i):
                countArray[a] += 1    
    return countArray

def plotDist(results, category_names, prop, data_directory, a=-1):
    """
    
    Plots stacked horizontal barcharts of the distrbution of properties in
    training, cross-validation, and test sets.

    Parameters
    ----------
    results : dictionary
        Has the results of countCategories separated by their prescence in
        training, cross-validation, test.
    category_names : list
        List of values of the property (e.x. tope, front, backcorner, or 
        sensors 1,2,3, & 4)
    prop : string
        Property to graph, e.x. 'sensor', 'location'
    """
    if (np.all(results['Train'] == 0)):
        del results['Train']
    if (np.all(results['Cross'] == 0)):
        del results['Cross']
    if (np.all(results['Test'] == 0)):
        del results['Test']
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
    filename = data_directory + '/' + prop + '_dist'
    if (a != -1):
        filename += '_' + str(a)
    fig.savefig(filename)
    
def indexByProp(filterProp, ml_dataset, propList):
    """
    
    Helper function that generates a 2-D list of indices sorted by property.
    Example: a 2-D list of two lists containing the shuffled indices of
    sensors 1 & 2 in element zero and sensors 3 & 4 in element one.

    Parameters
    ----------
    filterProp: string
        Property that will be separated by. Only done with 'sensor' so far,
        others may be buggy.
    ml_dataset: dictionary
        Where data is stored, imported from filtered .json files.
    propList: list
        List of values of the property (e.x. tope, front, backcorner, or 
        sensors 1,2,3, & 4)
    Returns
    ----------
    tempList: 2-D list
        Described above.
    """
    num_examples = len(ml_dataset['waves'])
    divideFactor = 1
    if (filterProp == 'sensor'):
        divideFactor = 2
    tempIndexes = []
    tempList = []
    for j in range(len(propList)):
        for i in range(num_examples):
            if (ml_dataset[filterProp][i] == propList[j]):
                tempIndexes.append(i)
    for i in range(1, divideFactor+1):
        tempList.append(tempIndexes[int((num_examples/divideFactor)*(i-1)):int((num_examples/divideFactor)*(i))])
    for subList in tempList:
        random.shuffle(subList)
    return tempList

if __name__ == '__main__':
    
    # Input train and valid proportions
    # Test proportion is whatever's left.
    train_proportion = [0.5, 0]
    valid_proportion = [0.2, 0]
    # test_proportion = [0.3, 1]
    splitDataset(train_proportion,valid_proportion, 'sensor') 
    