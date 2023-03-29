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
from physionet2023.modeling.convTST import (
    config_factory,
    lightning_tst_factory,
    single_dl_factory,
    train_fn,
)

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
    state_dict = torch.load(f"{model_folder}/state_dict")

    return state_dict


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    tst_config = config_factory()

    # TODO: this dl factory method is v. wasteful, consider refactoring to a single dl generated in load_challenge_models
    dl = single_dl_factory(
        tst_config, [patient_id], data_folder, for_testing=True, quality_cutoff=0.0
    )

    model = lightning_tst_factory(tst_config, dl.dataset)
    model.load_state_dict(models)

    model = model.eval()

    with torch.no_grad():
        preds = list()

        for X, _ in dl:
            preds.append(model(X))

    preds = torch.concat(preds)  # concatenate all batches
    # TODO: it seems like sometimes the probability distribution here is bimodal (either 1 or 5)
    # This is obviously undersireable; should think about this more
    preds = preds.mean(dim=0)  # average the probs for every class

    predicted_CPC = int(preds.argmax()) + 1

    # NOTE: this could result in a predicted CPC in one class, but a predicted outcome in another class
    # E.g. [0.0, 0.4, 0.2, 0.2, 0.2]
    #   predicted CPC = 2 (good outcome)
    #   predicted outcome = 1 (bad outcome)
    #   predicted outcome probability (of bad outcome) = 60%
    probability_density_bad = float(preds[4] + preds[3] + preds[2])
    probability_density_good = float(preds[1] + preds[0])

    if probability_density_bad > probability_density_good:
        outcome_binary = 1.0
    else:
        outcome_binary = 0.0

    outcome_probability = probability_density_bad

    return outcome_binary, outcome_probability, predicted_CPC


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    raise NotImplementedError()


# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    raise NotImplementedError()
