#!pip install wandb 
import wandb

from waves.misc import flatten
from waves.load_data import load_json_file_from_path
from waves.load_data import get_ml_dataset_paths
from waves.ml.acoustic_emission_dataset import AcousticEmissionDataset
from waves.ml.model_architectures import NeuralNetwork_01, NeuralNetwork_02, NeuralNetwork_03

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from torch import tensor
import torch.nn as nn
from tqdm.notebook import tqdm
import re, json, requests

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(config=None): 

    with wandb.init(project=None, config=config):

        print('---------- HYPERPARAMETERS ---------------------------------------')
        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        for key,value in config.items():
            print(key, ':', value)
        print("")
        
        print('---------- DATA AND MODEL ----------------------------------------')
                
        # Create model, get train and validation data, set loss function / optimizer
        model, train_loader, valid_loader, loss_func, optimizer = make(config)
        print(model)
        print("")

        print('---------- TRAINING ----------------------------------------------')
        
        print("Begin model training...\n")         # Training!
        train(model, train_loader, loss_func, optimizer, config)
        print("Training completed.\n")
        
        print('---------- EVALUATION --------------------------------------------')

        print("Begin model evaluation...\n")
        print("\nEvaluating results on VALIDATION data...\n")
        eval_label, eval_predicted, data_indices, all_examples, all_labels = \
            evaluate(model, valid_loader, config, valid = True)
        print("Evaluation completed.\n")

        print('---------- SAVE MODEL --------------------------------------------')
        
        print('Saving model to run directory...\n') # Save model local and to w&b 
        
        # If running locally, uncomment this line
        # torch.save(model.state_dict(), wandb.run.dir+'\model.pth')
        
        # If running google colab, use below, otherwise comment out.
        torch.save(model.state_dict(),'model.pth')
        wandb.save('model.pth') 

        print('\nModel was trained and evaluated.')
        print('Results, log, and trained model are backed up to W&B!\n')

        print('---------- DONE --------------------------------------------------')

    return model

def make(config):
    
    # Get the train and validation data from selected ml folder
    train, valid, feature_dim, num_classes = get_train_and_valid_data(config) 
    wandb.log({"feature_dim": feature_dim})
    wandb.log({"num_classes": num_classes})

    # Log a plot illustrating ratio of test to train data for each class
    y_valid = torch.argmax(valid[:][1], dim=1)
    y_train = torch.argmax(train[:][1], dim=1)
    wandb.sklearn.plot_class_proportions(y_train,y_valid, config.location)
        
    # Create data loaders
    # Note: dataset is shuffled at this step in the loader
    train_loader = make_loader(train, batch_size=config.batch_size)
    valid_loader = make_loader(valid, batch_size=config.batch_size)
  
    # Make the model according to configuration and available models
    if config.architecture == 1: # 1 layer
        model = NeuralNetwork_01(feature_dim, num_classes, config.hidden_units).to(device)
    if config.architecture == 2: # 2 layer
        model = NeuralNetwork_02(feature_dim, num_classes, config.hidden_units).to(device)
    if config.architecture == 3: # 3 layer
        model = NeuralNetwork_03(feature_dim, num_classes, config.hidden_units).to(device)
        
    # Make the loss and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    # Document shapes of batches
    for ex_idx, (batch_x,batch_y,index) in enumerate(train_loader):
        print(f"Shape of x batch is: {batch_x.shape}")
        print(f"Datatype of x batch is: {batch_x.dtype}\n")
        print(f"Shape of y batch is: {batch_y.shape}")
        print(f"Datatype of y batch is: {batch_y.dtype}\n")
        break

    return model, train_loader, valid_loader, loss_func, optimizer

def get_train_and_valid_data(config):
    
    print('Getting train and validation test sets...')
    train = AcousticEmissionDataset(config.train_path,
                                     config.sig_len,
                                     config.dt,
                                     config.low_pass,
                                     config.high_pass,
                                     config.fft_units,
                                     config.num_bins,
                                     config.feature)
    
    valid = AcousticEmissionDataset(config.valid_path,
                                     config.sig_len,
                                     config.dt,
                                     config.low_pass,
                                     config.high_pass,
                                     config.fft_units,
                                     config.num_bins,
                                     config.feature)
    
    example_x, example_y, _ = train[0] # to determine feature dim

    feature_dim = example_x.shape[0]   # for model creation input dim
    num_classes = example_y.shape[0]   # for model creation output dim
    
    print(f'\nfeature_dim : {feature_dim}')
    print(f'num_classes : {num_classes}\n')

    return train, valid, feature_dim, num_classes

def make_loader(dataset, batch_size): # NOTE THAT IT SHUFFLES
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def train(model, loader, loss_func, optimizer, config):
    
    # Tell wandb to watch what the model gets up to: gradients, weights, etc
    wandb.watch(model, loss_func, log="all", log_freq=20)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (waves, label, index) in enumerate(loader):

            loss = train_batch(waves, label, model, optimizer, loss_func)
            example_ct +=  len(waves) # since batch contains multiple examples
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

def train_batch(waves, label, model, optimizer, loss_func):
    
    # Use gpu
    waves, label = waves.to(device), label.to(device)

    # Forward pass ➡
    outputs = model(waves)
    loss = loss_func(outputs, label)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def evaluate(model, loader, config, test = False, valid = False): # would work with test or validation dataset
    model.eval()

    # Run the model on the provided examples
    with torch.no_grad():
        
        eval_predicted = []
        eval_label = []
        correct, total = 0, 0
        i = 0
        
        all_examples = []
        all_labels = []
        data_indices = []
        
        for waves, label, index in loader:
            waves, label = waves.to(device), label.to(device)
            outputs = model(waves)
            _, predicted = torch.max(outputs, 1)
            _, label = torch.max(label, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
            eval_predicted.append(predicted) # ex. [0,0,1]
            eval_label.append(label)
            data_indices.append(index) # original dataset indices prior to shuffle
            all_examples.append(waves) # appends the whole batch
            all_labels.append(label)

        print(f"Accuracy of the model on the {total} " +
              f"evaluation waves: {100 * correct / total}%")
        
    # Obnoxious coding to get into some correct format
    eval_predicted = [t.cpu().numpy() for t in eval_predicted]
    eval_label = [t.cpu().numpy() for t in eval_label]
    data_indices = [t.cpu().numpy() for t in data_indices]
    all_examples = [t.cpu().numpy() for t in all_examples]
    all_labels = [t.cpu().numpy() for t in all_labels]
    
    # Obnoxious coding to remove dimension caused by batches
    eval_label = flatten(eval_label) 
    eval_predicted = flatten(eval_predicted)
    data_indices = flatten(data_indices)
    all_examples = flatten(all_examples)
    all_labels = flatten(all_labels)

    # Compute Precision, Recall, Accuracy
    print("")
    if valid:
        print("Validation Set Metrics:")
    if test:
        print("Test Set Metrics:")
    print(classification_report(eval_predicted,eval_label,
                                target_names=config.location))
    metrics = classification_report(eval_predicted,eval_label,
                                target_names=config.location,
                                    output_dict=True)
    wandb.log(metrics)
    
    # Record labels predicted and the actual labels
    if valid:
        wandb.log({"valid_accuracy": correct / total})
        wandb.log({"valid_label": eval_label,"valid_predicted": eval_predicted})
        print('\nModel succesfully evaluated using validation set! \n')
    if test:
        wandb.log({"test_accuracy": correct / total})
        wandb.log({"test_label": eval_label,"test_predicted": eval_predicted})
        print('\nModel succesfully evaluated using test set! \n')
    
    # Logging to know indices wrt original dataset 
    # since DataLoaders shuffle when batching
    wandb.log({"data_indices": data_indices}) 
    
    return eval_label, eval_predicted, data_indices, all_examples, all_labels

def get_run(run_file_path):
    # get hyperparameters / configuration dict for specified W&B run
    api = wandb.Api()
    run = api.run(run_file_path)

    return run.config, run.summary

def get_test_data(config): # get test data associated with config file

    test = AcousticEmissionDataset(config.test_path,
                                         config.sig_len,
                                         config.dt,
                                         config.low_pass,
                                         config.high_pass,
                                         config.fft_units,
                                         config.num_bins,
                                         config.feature)

    example_x, example_y, index = test[0] # to determine feature dim

    feature_dim = example_x.shape[0] # for model creation input dim
    num_classes = example_y.shape[0] # for model creation output dim
    
    print(f'\nfeature_dim : {feature_dim}')
    print(f'num_classes : {num_classes}\n')

    return test, feature_dim, num_classes

def load_trained_model(run_file_path, config, feature_dim, num_classes):
        
    if config.architecture == 1: # 1 layer
        model = NeuralNetwork_01(feature_dim, num_classes, config.hidden_units).to(device)
    if config.architecture == 2: # 2 layer
        model = NeuralNetwork_02(feature_dim, num_classes, config.hidden_units).to(device)
    if config.architecture == 3: # 3 layer
        model = NeuralNetwork_03(feature_dim, num_classes, config.hidden_units).to(device)
        
    # Get model file, puts it into current work directory
    model_pth_wb_path = wandb.restore('model.pth', run_path=run_file_path)

    # Reinstate trained weights on created model
    model.load_state_dict(torch.load(model_pth_wb_path.name))

    return model # from previous trained run

def evaluate_trained_model_on_test_set(run_file_path, project='robustness',
                                       config=None,run_summary=None): 
    
    # tell wandb to get started
    with wandb.init(project=project,config=config):
        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        print("Confirm hyperparameters match with expected:")
        for key,value in config.items():
            print(key, ':', value)
        print("")
        
        print("Loading in test data...\n")
        # Load test data
        test,feature_dim,num_classes = get_test_data(config)

        print('-----------------------------------------------------------------')
        print("Load model in based on configuration file...\n")
        model = load_trained_model(run_file_path,config,feature_dim,num_classes)
        
        test_loader = make_loader(test, batch_size=config.batch_size)

        print("\nEvaluate trained model on test data...\n")
        # Evaluate on test dataset
        test_label, test_predicted, data_indices, all_examples, all_labels = evaluate(model, test_loader, config, test = True)

        mislabeled = []
        for idx, _ in enumerate(test_predicted):
            if (test_predicted[idx] != test_label[idx]):
                mislabeled.append(idx)

    return test, test_loader, test_predicted, test_label, mislabeled, data_indices, all_examples, all_labels

