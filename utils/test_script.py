import numpy as np
from acoustic_emission_dataset import AcousticEmissionDataset
from model_architectures import NeuralNetwork_02
import pickle
import sys 
import datetime
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# SIGNAL PROCESSING CONSTANTS
SIG_LEN = 1024           # [samples / signal] ;
DT = 10**-7              # [seconds] ; sample period / time between samples
LOW_PASS = 50*10**3      # [Hz] ; low frequency cutoff
HIGH_PASS = 800*10**3    # [Hz] ; high frequency cutoff
FFT_UNITS = 1000         # FFT outputs in Hz, this converts to kHz
NUM_BINS = 26            # For partial power

# ML HYPERPARAMETERS
EPOCHS = 500             # training iterations
LEARNING_RATE = 1e-3     # step size for optimizer
BATCH_SIZE = 10          # for train and test loaders
ARCHITECTURE = 1
# NB: To vary autoencoder architecture, must do in class definition file

# FILE I/O
JSON_DATA_FILE = 'https://raw.githubusercontent.com/ntulshibagwale/aeml/master/data/natfreq/220617_natfreqdataset.json'

if __name__ == "__main__":
    
    # Load AE data
    ae_dataset = AcousticEmissionDataset(JSON_DATA_FILE,SIG_LEN,DT,LOW_PASS,
                                         HIGH_PASS,FFT_UNITS,NUM_BINS)
    #angles = ae_dataset.angles # what the one hot encoded targets map to
    #num_classes = len(angles)  # how many diff angles, for model output dim
    #example_feature_vec, _ = ae_dataset[0] # to determine feature dim
    #feature_dim = example_feature_vec.shape[0] # for model creation input dim
    