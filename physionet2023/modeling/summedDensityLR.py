import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from physionet2023 import config
from physionet2023.dataProcessing.patientDatasets import PatientDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling.scoringUtil import compute_challenge_score


def lr_collate(batch):
    y_coll = torch.stack([y for _, y in batch])
    x_summs = list()

    for X, _ in batch:
        # iterate over channels (18)
        x_summ = torch.stack([X[:, :, cdx].sum() for cdx in range(0, X.shape[-1])])
        x_summs.append(x_summ)

    x_coll = torch.stack(x_summs)
    # Standard scaling
    x_coll = (x_coll - x_coll.mean()) / x_coll.std()

    return x_coll.numpy(), (y_coll.numpy() > 2.0).astype(float)


def load_data():
    patient_ids = PatientDataset().patient_ids

    train_ids, test_ids = train_test_split(patient_ids, test_size=0.1, random_state=42)

    train_ds = SpectrogramDataset(train_ids)
    test_ds = SpectrogramDataset(test_ids)

    train_dl = DataLoader(
        train_ds,
        batch_size=len(train_ds),
        collate_fn=lr_collate,
        num_workers=config.cores_available,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        collate_fn=lr_collate,
        num_workers=config.cores_available,
    )

    train_X, train_y = next(iter(train_dl))
    test_X, test_y = next(iter(test_dl))

    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    print("[*] Loading data...")
    train_X, train_y, test_X, test_y = load_data()
    print("[+] Loaded data:")
    print(f"\tX train: {train_X.shape}")
    print(f"\tY train (+ label prevalence): {train_y.sum() / len(train_y)}")
    print(f"\tX test: {test_X.shape}")
    print(f"\tY test (+ label prevalence): {test_y.sum() / len(test_y)}")

    lr = LogisticRegression()
    lr.fit(train_X, train_y)
    print(f"[+] Successfully fit LR, running evaluation...")

    preds = lr.predict_proba(test_X)[:, 1]

    auroc = roc_auc_score(test_y, preds)
    comp_score = compute_challenge_score(test_y, preds)

    print(f"AUROC: {auroc}")
    print(f"Competition score: {comp_score}")
