"""

acoustic_emission_dataset

Loads in waveform files and performs any transforms on the data. To be used 
with pytorch data loaders.

Kiran Lochun

Updated: 2022-07-08

"""

import os
from os.path import isfile, join
import numpy as np
from tkinter import filedialog
from tkinter import *
import json
import random
import math
import datetime


def flatten(t): # flattens out a list of lists (CHECK)
    return [item for sublist in t for item in sublist]

def flatten2D(arr):
    if (len(arr) == 1):
        arr = arr[0]
    else:
        tempArr = arr[0]
        for i in range(1, len(arr)):
            tempArr = np.concatenate((tempArr, arr[i]), axis = 0)
        arr = tempArr
    return arr.tolist()

def splitDataset(trainProp, crossProp):
    
    root = Tk()
    root.filename = filedialog.askopenfilenames(
        title = "Select json files containing ML data")
    root.destroy()
    files = root.filename
    waves = []
    event = []
    angle = []
    location = []
    length = []
    #sensor = []
    for path in files:
        with open(path) as json_file:
            PLB = json.load(json_file)
        for key in PLB.keys():
            PLB[key]  = np.array(PLB[key])
        waves.append(PLB['waves'])           # List of raw waveforms
        event.append(PLB['event'])
        angle.append(PLB['angle'])
        location.append(PLB['location'])
        length.append(PLB['length'])
        #sensor.append(PLB['sensor'])
    waves = flatten2D(waves)
    event = flatten2D(event)
    angle = flatten2D(angle)
    location = flatten2D(location)
    length = flatten2D(length)
    #sensor = flatten2D(sensor)
    partition = {'train': [], 'cross': [], 'test': []}
    indexList = list(range(0,len(waves)))
    random.seed(2) # RNG Seed
    random.shuffle(indexList)
    trainEnd = math.floor(len(waves) * trainProp)
    crossEnd = trainEnd + (math.floor(len(waves) * crossProp))
    partition['train'] = indexList[0:trainEnd]
    partition['cross'] = indexList[trainEnd:crossEnd]
    partition['test'] = indexList[crossEnd:]
    root = Tk()
    root.filename = filedialog.askdirectory(
        title = "Select folder to save files")
    root.destroy()
    data_directory = root.filename
    dataset_name = os.path.basename(os.path.normpath(data_directory))
    now = datetime.datetime.now()
    time_stamp = str(now.strftime("%Y%m%d_"))
    dataset_name = time_stamp + dataset_name
    trainingDataset = {'waves' : waves[np.array(partition['train'])],
                       'event' : event[partition['train']], 
                       'angle' : angle[partition['train']],
                       'location' : location[partition['train']],
                       'length' : length[partition['train']],
                       #'sensor' : sensor
                       }
    with open(dataset_name + '_training.json', "w") as outfile:
        json.dump(trainingDataset, outfile)
    crossDataset =    {'waves' : waves[partition['cross']],
                       'event' : event[partition['cross']], 
                       'angle' : angle[partition['cross']],
                       'location' : location[partition['cross']],
                       'length' : length[partition['cross']],
                       #'sensor' : sensor
                       }
    with open(dataset_name + '_cross.json', "w") as outfile:
        json.dump(crossDataset, outfile)
    testDataset =     {'waves' : waves[partition['test']],
                       'event' : event[partition['test']], 
                       'angle' : angle[partition['test']],
                       'location' : location[partition['test']],
                       'length' : length[partition['test']],
                       #'sensor' : sensor
                       }
    with open(dataset_name + '_test.json', "w") as outfile:
        json.dump(testDataset, outfile)

if __name__ == '__main__':
    splitDataset(.5,.2) # train and cross proportions. test proportion is whatever's left.
    