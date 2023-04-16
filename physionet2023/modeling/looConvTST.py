import random

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from physionet2023.dataProcessing.patientDatasets import MetadataOnlyDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling.convTST import (
    config_factory,
    lightning_tst_factory,
    single_dl_factory,
)
from physionet2023.modeling.scoringUtil import (
    compute_challenge_score,
    regression_to_probability,
)

if __name__ == "__main__":
    random.seed(42)
    metadata_ds = MetadataOnlyDataset()

    labels = list()
    preds = list()

    all_patient_ids = metadata_ds.patient_ids
    # loo_sample_patient_ids = random.sample(all_patient_ids, 3)
    loo_sample_patient_ids = all_patient_ids

    for loo_pid in tqdm(loo_sample_patient_ids):
        training_pids = [p for p in all_patient_ids if p != loo_pid]

        assert loo_pid not in training_pids

        tst_config = config_factory()
        train_dl = single_dl_factory(tst_config, training_pids)
        valid_dl = single_dl_factory(tst_config, [loo_pid])
        model = lightning_tst_factory(tst_config, train_dl.dataset)

        trainer = pl.Trainer(
            max_epochs=2,
            gradient_clip_val=4.0,
            gradient_clip_algorithm="norm",
            accelerator="gpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
        )

        model.eval()

        this_patient_preds = list()
        this_patient_labels = list()

        for X, y in valid_dl:
            with torch.no_grad():
                this_patient_preds.append(model(X).cpu())
                this_patient_labels.append(y.cpu())

        torch.save(torch.cat(this_patient_preds), f"cache/loo/{loo_pid}.preds.pt")

        preds.append(torch.cat(this_patient_preds).mean())
        labels.append(torch.cat(this_patient_labels).mean())

    preds = (torch.stack(preds) * 4) + 1
    target = (torch.stack(labels) * 4) + 1

    torch.save(preds, "cache/loo/preds.pt")
    torch.save(target, "cache/loo/target.pt")

    preds = regression_to_probability(preds).cpu().numpy()
    target = (target > 2).int().cpu().numpy()

    print(compute_challenge_score(target, preds))
    print(roc_auc_score(target, preds))
