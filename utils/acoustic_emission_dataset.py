"""

acoustic_emission_dataset

Loads in waveform files and performs any transforms on the data. To be used 
with pytorch data loaders.

Nick Tulshibagwale

Updated: 2022-06-02

"""

import torch
from torch.utils.data import Dataset
from .load_data import load_dataset_from_github
#from load_data import load_dataset_from_github
#from ae_measure2 import wave2vec
#from ae_measure2 import fft

from torch import tensor
from .ae_measure2 import wave2vec
from .ae_measure2 import fft
import numpy as np

class AcousticEmissionDataset(Dataset):
    
    # Constructor
    def __init__(self,url,sig_len,dt,low_pass,high_pass,fft_units,
                 num_bins):
      
        self.url = url
        self.sig_len = sig_len
        self.dt = dt
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.fft_units = fft_units
        self.num_bins = num_bins
        
        # Load in AE Data
        data = load_dataset_from_github(url)
        
        # Separate dict into arrays
        waves = data['waves']           # List of raw waveforms
        #targets = data['target']       # waveform labels (legacy PLB data)
        locations = data['location']    # where pencil broken
        self.lengths = data['length']
        self.events = data['event']
        self.angles = data['angle']
        
        # Map location to a number (starting from 0)
        targets = []
        for location in locations:
            if location == 'topp':
                targets.append(0)
            elif location == 'side':
                targets.append(1)
            elif location == 'tope':
                targets.append(2)
        
        #self.angles = data['target_angle']  # angle, one hot encoded
        self.targets = tensor(targets,dtype=torch.int64,requires_grad=False)
      
        # One hot encode
        targets_one_hot = tensor(targets,dtype=torch.int64,requires_grad=False)
        targets_one_hot = torch.nn.functional.one_hot(targets_one_hot.long()) 
        self.targets_one_hot = targets_one_hot.float()
        print("targets_one_hot is the one hot encoding for angle. ex: [1 0 1]")

        self.n_samples = self.targets.shape[0]    # Number of samples/labels
        self.waves = tensor(waves,dtype=torch.float32,requires_grad=False)                          
                   
        print("")
        print(f"Shape of waves is: {self.waves.shape}")
        print(f"Datatype of waves is: {self.waves.dtype}")
        print("waves requires grad:", self.waves.requires_grad)
        print(f"Shape of targets is: {self.targets.shape}")
        print(f"Datatype of targets is: {self.targets.dtype}")
        print("targets requires grad:", self.targets.requires_grad)
        print(f"Ex: {targets[0]}")
        print(f"Shape of targets_one_hot is: {self.targets_one_hot.shape}")
        print(f"Datatype of targets_one_hot is: {self.targets_one_hot.dtype}")
        print("targets_one_hot requires grad:", \
              self.targets_one_hot.requires_grad)
        print(f"Ex: {targets_one_hot[0]}")
        print("")
        
        print("AcousticEmissionDataset loaded in!\n")
        
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
        x = self.waves[index]      
        y = self.targets_one_hot[index] 
              
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
        targets = self.targets # use int label for which angle it is
        indices = []
        for idx, target in enumerate(targets): # loop through all data
            if self.angles[target] in specific_angles:
                indices.append(idx) # if angle is a match remember index
        
        subset = torch.utils.data.Subset(self,indices) # get subset
        
        return subset