"""

acoustic_emission_dataset

Loads in waveform files and performs any transforms on the data. To be used 
with pytorch data loaders.

Nick Tulshibagwale

Updated: 2022-06-02

"""

import torch
from torch.utils.data import Dataset
from torch import tensor
import numpy as np

from waves.load_data import load_json_file_from_path
from waves.signal_processing import fft

class AcousticEmissionDataset(Dataset):
    
    # Constructor
    def __init__(self,path,sig_len,dt,low_pass,high_pass,fft_units,
                 num_bins, feature):
      
        self.path = path
        self.sig_len = sig_len
        self.dt = dt
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.fft_units = fft_units
        self.num_bins = num_bins
        self.feature = feature
        
        # Load in AE Data
        print(f"Loading in Dataset from {path}")
        data = load_json_file_from_path(path)
        print("Successfully loaded in.")

        # Separate dict into arrays
        waves = data['waves']           # List of raw waveforms
        self.waves = tensor(waves,dtype=torch.float32,requires_grad=False)                          
        locations = data['location']    # where pencil broken
        self.length = data['length']
        self.event = data['event']
        self.angle = data['angle']
        self.distance = data['distance']
        self.sensor = data['sensor']
        
        # Location is the source we are trying to identify
        # Map location to a number (starting from 0)
        label = []
        for location in locations:
            if location == 'front':
                label.append(0)
            elif location == 'tope':
                label.append(1)
            elif location == 'backcorner':
                label.append(2)
        
        self.label = tensor(label,dtype=torch.int64,requires_grad=False)
      
        # One hot encode the label
        label_one_hot = tensor(label,dtype=torch.int64,requires_grad=False)
        label_one_hot = torch.nn.functional.one_hot(label_one_hot.long()) 
        self.label_one_hot = label_one_hot.float()
        self.n_samples = self.label.shape[0]    # Number of samples/labels

        print(f"Shape of waves is: {self.waves.shape}")
        print(f"Datatype of waves is: {self.waves.dtype}")
        print("waves requires grad:", self.waves.requires_grad)
        print(f"Shape of label is: {self.label.shape}")
        print(f"Datatype of label is: {self.label.dtype}")
        print("Label requires grad:", self.label.requires_grad)
        print(f"Ex: label[0] = {label[0]}")
        print("label_one_hot is the one hot encoding of label for source." ,
              " For ex: [1 0 0] is a front occuring PLB.")
        print(f"Shape of label_one_hot is: {self.label_one_hot.shape}")
        print(f"Datatype of label_one_hot is: {self.label_one_hot.dtype}")
        print("label_one_hot requires grad: ",
              self.label_one_hot.requires_grad)
        print(f"Ex: label_one_hot[0] = {label_one_hot[0]}")
        print("")
        print(f"AcousticEmissionDataset loaded in using {path}!\n")
        
    def __getitem__(self,index):
        """
        
        Function called when object is indexed. The transformation of data 
        occurs in this getter function. In other words, the constructor reads
        in the raw data filtered by hand, and this function contains the 
        sequence of remaining processing on the data to extract features used
        in ML models.
        
        index (int): the index of the sample and label
        
        return:
        (x,y): feature vector and label corresponding to single event
        
        """

        if self.feature == 'waveform':
            x = self.waves[index]      
        if self.feature == 'fft':
            _,x = fft(self.dt, self.waves[index], low_pass=self.low_pass,
                    high_pass = self.high_pass)
        
        x = tensor(x,dtype=torch.float32,requires_grad=False)
            
        y = self.label_one_hot[index] 
              
        return x, y # input example, label
    
    def __len__(self): # number of samples
        return self.n_samples
  
    def _get_angle_subset(self, specific_angles):
        """
        
        Get subset of AE data based on angle.

        Parameters
        ----------
        specific_angles : array-like
            List of angles that a sub-dataset will be composed of.

        Returns
        -------
        subset : Pytorch Dataset Object
            Object for training / testing, containing specified angles.

        """
        label = self.label # use int label for which angle it is
        indices = []
        for idx, target in enumerate(label): # loop through all data
            if self.angle[target] in specific_angles:
                indices.append(idx) # if angle is a match remember index
        
        subset = torch.utils.data.Subset(self,indices) # get subset
        
        return subset