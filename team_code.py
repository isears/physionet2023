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
from physionet2023 import config
from physionet2023.modeling.rawWaveformTST import (
    config_factory,
    lightning_tst_factory,
    single_dl_factory,
    train_fn,
)
from physionet2023.modeling.scoringUtil import (
    regression_to_probability,
    regression_to_probability_smooth,
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

    if config.gpus_available > 0:
        model.to("cuda")

    with torch.no_grad():
        preds = list()

        for X, _ in dl:
            if config.gpus_available > 0:
                X = X.to("cuda")

            preds.append(model(X))

    avg_pred = torch.concat(preds).mean().cpu()

    outcome_probability = regression_to_probability_smooth(avg_pred)

    predicted_CPC = int(avg_pred.round())
    outcome_binary = int(outcome_probability.round())

    return outcome_binary, float(outcome_probability), predicted_CPC


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
