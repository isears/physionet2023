import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score

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
    metadata_ds = MetadataOnlyDataset()

    labels = list()
    preds = list()

    patient_ids = metadata_ds.patient_ids[0:3]  # TODO: debug only

    for pid in patient_ids:
        left_out = pid
        training_pids = [p for p in metadata_ds.patient_ids if p != left_out]

        tst_config = config_factory()
        train_dl = single_dl_factory(tst_config, training_pids)
        valid_dl = single_dl_factory(tst_config, [left_out])
        model = lightning_tst_factory(tst_config, train_dl.dataset)

        trainer = pl.Trainer(
            max_epochs=1,
            gradient_clip_val=4.0,
            gradient_clip_algorithm="norm",
            accelerator="gpu",
            enable_checkpointing=False,
            enable_progress_bar=True,
            logger=False,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
        )

        test_ds = SpectrogramDataset(
            root_folder="./data",
            patient_ids=[left_out],
            for_classification=False,
            normalize=True,
        )

        model.eval()

        for X, y in valid_dl:
            with torch.no_grad():
                preds.append(model(X).cpu())
                labels.append(y.cpu())

    preds = (torch.cat(preds).cpu().numpy() * 4) + 1
    target = (torch.cat(labels).cpu().numpy() * 4) + 1

    preds_prob = regression_to_probability(preds)

    print(compute_challenge_score(target, preds_prob))
    print(roc_auc_score(target, preds_prob))
