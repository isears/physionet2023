import os
import sys

from sklearn.metrics import roc_auc_score

from physionet2023.dataProcessing.exampleUtil import *
from physionet2023.modeling.scoringUtil import compute_challenge_score


def load_metadata_file(path):
    # Load non-recording data.
    patient_metadata_raw = load_text_file(path)
    patient_metadata = {
        line.split(": ")[0]: line.split(": ")[-1]
        for line in patient_metadata_raw.split("\n")
    }
    return patient_metadata


if __name__ == "__main__":
    # Assuming data locations
    data_path = "./data"
    outputs_path = "cache/test_outputs"

    labels = list()
    preds = list()

    for patient_id in [d for d in os.listdir(data_path) if d.startswith("ICARE_")]:
        # Define file location.
        patient_metadata_file = os.path.join(data_path, patient_id, patient_id + ".txt")
        patient_metadata = load_metadata_file(patient_metadata_file)
        labels.append(float(float(patient_metadata["CPC"]) > 2))

        prediction_metadata_file = os.path.join(
            outputs_path, patient_id, patient_id + ".txt"
        )
        prediction_metadata = load_metadata_file(prediction_metadata_file)
        preds.append(float(prediction_metadata["Outcome probability"]))

    labels = np.array(labels)
    preds = np.array(preds)
    print(compute_challenge_score(labels, preds))
    print(roc_auc_score(labels, preds))
