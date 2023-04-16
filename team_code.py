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
import pytorch_lightning as pl
import torch

from helper_code import *
from physionet2023 import LabelType, config
from physionet2023.modeling.convTST import (
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
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        logger=False,
        devices=1,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
    )
    tst_config = config_factory()
    dl = single_dl_factory(tst_config, pids=None, data_path=data_folder)
    model = lightning_tst_factory(tst_config, dl.dataset)
    trainer.fit(model, train_dataloaders=dl)

    torch.save(model.state_dict(), f"{model_folder}/state_dict")


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
        tst_config,
        [patient_id],
        data_folder,
        quality_cutoff=0.0,
    )

    # Need to manually override the label type so that the dataset doesn't try to access labels
    dl.dataset.label_type = LabelType.DUMMY

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

        if len(preds) > 0:
            pred = preds[0]

            outcome_probability = pred[:, 2] + pred[:, 3] + pred[:, 4]

            predicted_CPC = int(outcome_probability.argmax())
            outcome_binary = int(outcome_probability.round())
        else:
            outcome_binary = 1
            outcome_probability = 1.0
            predicted_CPC = 5

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
