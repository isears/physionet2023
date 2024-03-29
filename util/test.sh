#!/bin/bash

docker run \
--rm \
--ipc=host \
--runtime=nvidia \
-v ./cache/models:/challenge/model \
-v ./data:/challenge/test_data \
-v ./cache/test_outputs:/challenge/test_outputs \
-v ./data:/challenge/training_data \
physionet2023 \
python run_model.py model test_data test_outputs