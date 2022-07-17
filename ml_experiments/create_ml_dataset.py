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

def split_dataset(train_percent, valid_percent, property_to_filter = '', 
                 recursing = False, index_list = [], data_directory = '', 
                 ml_dataset = {}, a= 0, partition = {}):
    """
    
    Splits json file(s) into training, validation, and test sets. Can split 
    along a single property (e.x. sensors, location).

    Parameters
    ----------
    train_percent : float
        Percentage of data that will be used for training.
    valid_percent : float
        Percentage of data that will be used for validation.
        (Note: 1 - train_percent - valid_percent is the percentage of data that
        will be used for testing.)
    property_to_filter (optional): string
        Property that will be separated by. Only done with 'sensor' so far,
        others may be buggy.
        
    """
    
    if (not recursing): # true on first call
        
        # Select filtered .json files to load in
        # Each .json file contains waveforms and metadata from a given 
        # experiment that was filtered.
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
        partition = {'train': [], 'valid': [], 'test': []}
        partition['random_seed'] = 2
        partition['files'] = files
        random.seed(partition['random_seed']) # RNG Seed
        
        # We need to shuffle index_list? I don't think we're shuffling?
        index_list = list(range(0,num_examples)) # index all examples
        random.shuffle(index_list) # Nick added back
        
        if (property_to_filter != ''): # if a property is specified 
            index_list = index_by_property(property_to_filter, ml_dataset,
                                    list(set(ml_dataset[property_to_filter])))
            # Index list is now a 2D list, separating indices corresponding
            # to different properties
            
        # Select output folder (create a folder in ml_datasets to save dataset)
        root = Tk()
        root.filename = filedialog.askdirectory(
            title = "Select folder to save files")
        root.destroy()
        data_directory = root.filename
            
    # Begin recursion, function calls itself after initial split of data
    if (property_to_filter != '' and not recursing):
        for a, sub_list in enumerate(index_list):
            split_dataset(train_percent[a], valid_percent[a],
                          property_to_filter, recursing = True,
                          index_list = sub_list,
                          data_directory = data_directory, 
                          ml_dataset = ml_dataset, a = a, 
                          partition = partition)
            
    # Run if :
    # Properties are specified and also currently recurring
    # No properties were specified, and not doing recursion call
    if ((property_to_filter != '' and recursing) or # yes property, recurring
        (property_to_filter == '' and not recursing)): # no property, no recurs
        
        # no prop, no recursion, use specified percentages for datasets
        if (property_to_filter == '' and not recursing):
            train_percent = train_percent[0] 
            valid_percent = valid_percent[0]
        else: # empty the partition indices
            partition['train'] = []
            partition['valid'] = []
            partition['test'] = []
        
        # Create dictionaries for partitioned data in prep for .json file
        num_examples = len(index_list)
        train = dict.fromkeys(ml_dataset) # get same keys
        valid = dict.fromkeys(ml_dataset)
        test =  dict.fromkeys(ml_dataset)
        ml_dataset = dict_to_list(ml_dataset) # switch to list for easy looping
        
        # Using percentages to divide up examples via indexing
        train_end = math.floor(num_examples * train_percent)
        valid_end = train_end + (math.floor(num_examples * valid_percent))
        if (train_percent != 0):
            partition['train'] = index_list[0:train_end]
        if (valid_percent != 0):
            partition['valid'] = index_list[train_end:valid_end]
        if (1-(train_percent + valid_percent) != 0):
            partition['test'] = index_list[valid_end:]
            
        # Split data
        for idx, key in enumerate(train.keys()): # loop thru keys (waves, etc)
            train[key] = ml_dataset[idx][partition['train']].tolist()
            valid[key] = ml_dataset[idx][partition['valid']].tolist()
            test[key]  = ml_dataset[idx][partition['test']].tolist()
    
        # Visualize distribution of specifed properties in datasets
        chart_properties = ['location', 'distance',
                            'angle', 'length', 'sensor'] # Properties plotted
        for prop in chart_properties:
            
            vals = set(test[prop]) # set: get possible entries
            # Not sure I understand why this is is test? This may need to
            # change
            
            # IMPORTANT!!! If the test set is ever empty, the program will 
            # crash because of the previous line.
            
            # Determine how many examples have a given property
            # Do this for each dataset
            train_count = np.zeros(len(vals))
            valid_count = np.zeros(len(vals))
            test_count = np.zeros(len(vals))
            results = {'Train':count_categories(vals, train_count, train,
                                                prop).astype(int), 
                       'Valid':count_categories(vals, valid_count, valid,
                                                prop).astype(int),
                       'Test': count_categories(vals, test_count, test,
                                                prop).astype(int)}
            
            if (recursing):
                plot_distribution(results, list(vals), prop, data_directory, a)
            else:
                plot_distribution(results, list(vals), prop, data_directory)
                
        # File output, save to selected folder:
        # datasets, parameter .txt file, and parameter .json fies, plots
        dataset_name = os.path.basename(os.path.normpath(data_directory))
        now = datetime.datetime.now()
        time_stamp = str(now.strftime("%Y%m%d_"))
        dataset_name = time_stamp + dataset_name
        if (recursing):
            dataset_name += str(a).zfill(2)
        os.chdir(data_directory)
        if (train_percent != 0): # Q: Is this needed?
            with open(dataset_name + '_train.json', "w") as outfile:
                json.dump(train, outfile)
        if (valid_percent != 0):
            with open(dataset_name + '_valid.json', "w") as outfile:
                json.dump(valid, outfile)
        if (1-(train_percent + valid_percent) != 0):
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
            if (train_percent != 0):
                f.write('\nTrain indices: \n')        
                f.write(' '.join([str(x) for x in partition['train']]))
                f.write('\n')
            if (valid_percent != 0):
                f.write('\nValid indices: \n')        
                f.write(' '.join([str(x) for x in partition['valid']]))
                f.write('\n')
            if (1-(train_percent + valid_percent) != 0):
                f.write('\nTest indices: \n')        
                f.write(' '.join([str(x) for x in partition['test']]))
                f.write('\n')
        
def count_categories(vals, count_array, data, prop):
    """
    
    Helper function that counts instances of a specific property in a dataset
    i.e. the number of waveforms that were broken at the front.

    Parameters
    ----------
    vals : set
        Set of values of the property (e.x. tope, front, backcorner, or 
        sensors 1,2,3, & 4)
    count_array : array-like
        Where the counts are stored. Has the same length as vals. So for ex,
        if vals is {tope, front}, then count_array will be [# of tope, # of 
        front].
    data : dictionary
        Where data is stored, imported from filtered .json files.
    prop: string
        Property to count, e.x. 'sensor', 'location'
        
    Returns
    ----------
    count_array: array-like
        Described above.
        
    """
    # Loop through all examples in given dataset (ex: train)
    for x in range(len(data[prop])): 
        # Check which value the ex takes on, add to appropriate index
        for a, i in enumerate(vals): 
            if (data[prop][x] == i):
                count_array[a] += 1    
                
    return count_array

def plot_distribution(results, category_names, prop, data_directory, a=-1):
    """
    
    Plots stacked horizontal barcharts of the distribution of properties in
    training, validation, and test sets.

    Parameters
    ----------
    results : dictionary
        Has the results of count_categories separated by their presence in
        training, validation, test.
    category_names : list
        List of values of the property (e.x. tope, front, backcorner, or 
        sensors 1,2,3, & 4)
    prop : string
        Property to graph, e.x. 'sensor', 'location'
        
    """
    if (np.all(results['Train'] == 0)):
        del results['Train']
    if (np.all(results['Valid'] == 0)):
        del results['Valid']
    if (np.all(results['Test'] == 0)):
        del results['Test']
        
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    
    cmap = plt.get_cmap('plasma')
    color_values = np.linspace(0.15, 0.85, data.shape[1])
    category_colors = cmap(color_values)
    
    fig, ax = plt.subplots(figsize=(9, 6))
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
                     fontsize = 26)
    
    ax.legend( loc='lower right', fontsize=16)
    ax.set_title(prop.capitalize() + 's ' + 'Distribution' , loc = 'right',
                 fontsize = 28)
    plt.rc('ytick', labelsize=24)  # fontsize of the y tick labels
    filename = data_directory + '/' + prop + '_distribution'
    if (a != -1):
        filename += '_' + str(a)
    fig.savefig(filename)
    
def index_by_property(property_to_filter, ml_dataset, property_list):
    """
    
    Helper function that generates a 2-D list of indices sorted by property.
    Example: a 2-D list of two lists containing the shuffled indices of
    sensors 1 & 2 in element zero and sensors 3 & 4 in element one.

    Parameters
    ----------
    property_to_filter: string
        Property that will be separated by. Only done with 'sensor' so far,
        others may be buggy.
    ml_dataset: dictionary
        Where data is stored, imported from filtered .json files.
    property_list: list
        List of values of the property (e.x. tope, front, backcorner, or 
        sensors 1,2,3, & 4)
        
    Returns
    ----------
    tempList: 2-D list
        Described above.
        
    """
    num_examples = len(ml_dataset['waves'])
    divideFactor = 1
    if (property_to_filter == 'sensor'):
        divideFactor = 2
    tempIndexes = []
    tempList = []
    
    # Loop through all properties
    for j in range(len(property_list)):
        # Loop through all examples and append index when equal to property
        for i in range(num_examples):
            if (ml_dataset[property_to_filter][i] == property_list[j]):
                tempIndexes.append(i)
                
    # Pull out correct indices corresponding to a property in property_list
    for i in range(1, divideFactor+1):
        tempList.append(tempIndexes[
            int((num_examples/divideFactor)*(i-1)):
                int((num_examples/divideFactor)*(i))])
            
    for subList in tempList:
        random.shuffle(subList)
        
    return tempList

if __name__ == '__main__':
    
    # Input train and valid proportions, test proportion is remainder
    train_percent = [0.5, 0.5]
    valid_percent = [0.2, 0.2] # why array?
    # test_percent = [0.3, 1]
    
    split_dataset(train_percent,valid_percent,property_to_filter='') 
    