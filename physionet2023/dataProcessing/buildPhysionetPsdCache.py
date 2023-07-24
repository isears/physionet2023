import torch
from tqdm import tqdm

from physionet2023 import config
from physionet2023.dataProcessing.patientDatasets import PhysionetPsdDataset

if __name__ == "__main__":
    ds = PhysionetPsdDataset()
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=32,
    )

    psds = list()
    metadatas = list()
    ys = list()

    for psd, metadata, label in tqdm(dl, total=len(dl)):
        psds.append(psd)
        metadatas.append(metadata)
        ys.append(label)

    psds = torch.concat(psds).float()
    metadatas = torch.concat(metadatas).float()
    ys = torch.concat(ys).float()

    torch.save(psds, f"./cache/physionet_cache/psd.pt")
    torch.save(metadatas, f"./cache/physionet_cache/metadata.pt")
    torch.save(ys, f"cache/physionet_cache/label.pt")
