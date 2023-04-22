import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold

from physionet2023 import *
from physionet2023.dataProcessing.patientDatasets import MetadataOnlyDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling.encoders.convEncoderTST import (
    ConvEncoderTST,
    config_factory,
)


def fold_generator(tst_config, n_folds=5):
    kf = KFold(n_folds, random_state=42, shuffle=True)
    pids = MetadataOnlyDataset().patient_ids

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(pids)):
        train_pids = [pids[idx] for idx in train_indices]
        test_pids = [pids[idx] for idx in test_indices]

        train_ds, test_ds = tuple(
            SpectrogramDataset(
                patient_ids=group,
                label_type=LabelType.SINGLECLASS,
                preprocess=True,
                last_only=False,
                quality_cutoff=0.0,
            )
            for group in [train_pids, test_pids]
        )

        train_dl, test_dl = tuple(
            torch.utils.data.DataLoader(
                ds,
                num_workers=config.cores_available,
                batch_size=tst_config.batch_size,
                pin_memory=True,
            )
            for ds in [train_ds, test_ds]
        )

        for pid in train_dl.dataset.patient_ids:
            assert pid not in test_dl.dataset.patient_ids

        yield train_dl, test_dl


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "pretrained":
            pretrained = True
        elif sys.argv[1] == "unpretrained":
            pretrained = False
        else:
            raise ValueError(f"Invalid first argument: {sys.argv[1]}")
    else:
        print(f"[*] No argument specified, defaulting to pretrained CV")
        pretrained = True

    cv_results = {}
    tst_config = config_factory()

    for fidx, (train_dl, test_dl) in enumerate(fold_generator(tst_config)):
        print(f"[+] Starting fold {fidx}")
        model = ConvEncoderTST(tst_config, pretrained=pretrained)
        checkpoint_saver = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="cache/checkpoints",
        )

        trainer = pl.Trainer(
            max_epochs=100,
            logger=False,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=True,
                    patience=10,
                    check_finite=False,
                ),
                checkpoint_saver,
            ],
            enable_checkpointing=True,
            val_check_interval=0.1,
        )
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=test_dl)
        results = trainer.test(
            model=model, dataloaders=test_dl, ckpt_path=checkpoint_saver.best_model_path
        )

        for k, v in results[0].items():
            if not k in cv_results:
                cv_results[k] = list()

            cv_results[k].append(v)

    print("CV done")
    for k, v in cv_results.items():
        print()
        print(f"Metric {k}")
        print(v)

        print(f"\tMean: {np.array(v).mean()}")
        print(f"\tStd: {np.array(v).std()}")
