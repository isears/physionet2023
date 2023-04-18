import numpy as np
import torch
from tqdm import tqdm

from physionet2023.dataProcessing.TuhDatasets import TuhPatientDataset

if __name__ == "__main__":
    ds = TuhPatientDataset()

    # max = 4  # TODO: debug only
    for idx, patient_path in enumerate(tqdm(ds.patient_paths)):
        patient_id = patient_path.split("/")[-1]

        try:
            X = ds[idx]
            torch.save(X, f"cache/tuh_cache/{patient_id}.pt")
        except Exception as e:
            print(f"Error encountered with {patient_id}, skipping...")

        # if idx == max:
        #     break
