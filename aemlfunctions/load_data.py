"""

Code for retrieving trained model results and json datasets from github.

Nick Tulshibagwale

"""
from acoustic.ml.model_architectures import NeuralNetwork_01, NeuralNetwork_02
import torch
import json, requests
import numpy as np

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

    
