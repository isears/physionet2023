import numpy as np
import torch
from tqdm import tqdm

from physionet2023.dataProcessing.TuhDatasets import TuhPatientDataset

if __name__ == "__main__":
    ds = TuhPatientDataset()
    all_x = list()

    # max = 4  # TODO: debug only
    for idx, patient_path in enumerate(tqdm(ds.patient_paths)):
        patient_id = patient_path.split("/")[-1]

        try:
            X = ds[idx]
            torch.save(X, f"cache/tuh_cache/{patient_id}.pt")
            all_x.append(X)
        except Exception as e:
            print(f"Error encountered with {patient_id}, skipping...")

        # if idx == max:
        #     break

    stacked_x = torch.stack(all_x, dim=-1)

    means = torch.mean(stacked_x, dim=-1)
    stds = torch.std(stacked_x, dim=-1)

    print(means)
    print(stds)

    torch.save(means, "cache/tuh_cache/_means.pt")
    torch.save(stds, "cache/tuh_cache/_stds.pt")
