#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

import os
import sys

import joblib
import numpy as np
import torch

from helper_code import *
from physionet2023.modeling.convTST import train_fn

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    state_dict = train_fn(data_folder, log=False)
    torch.save(state_dict, f"{model_folder}/state_dict")


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    raise NotImplemented


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    raise NotImplemented


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    raise NotImplemented


# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    raise NotImplemented
