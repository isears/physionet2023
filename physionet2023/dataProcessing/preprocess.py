"""
Dataset-wide standardization requires preprocessing beyond what streaming dataloaders can offer

Also need to preserve a train / valid split so that standardization can be applied to each subset separately
"""
from sklearn.model_selection import train_test_split

from physionet2023.dataProcessing.datasets import PatientDataset

if __name__ == "__main__":
    # Do the train / valid split
    patient_ds = PatientDataset("./data")
    train_pids, test_pids = train_test_split(
        patient_ds.patient_ids, test_size=0.2, random_state=42
    )

    # Iterate over all records in train / valid to do FFT and determine mean / std
    for patient_metadata, recording_metadata, recordings in patient_ds:
        pid = patient_metadata["Patient"]
        raise NotImplemented
