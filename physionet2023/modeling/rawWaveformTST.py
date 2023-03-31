import pandas as pd
import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from physionet2023 import config
from physionet2023.dataProcessing.datasets import (
    FftDataset,
    PatientDataset,
    SampleDataset,
    just_give_me_dataloaders,
)
from physionet2023.modeling import GenericPlTrainer, GenericPlTst
from physionet2023.modeling.scoringUtil import (
    compute_auroc_regressor,
    compute_challenge_score_regressor,
)


class UniformLengthTst(TSTransformerEncoderClassiregressor):
    def forward(self, X):
        padding_masks = torch.ones_like(X[:, 0, :]).bool()
        return super().forward(X.permute(0, 2, 1), padding_masks)


# Need fn here so that identical configs can be generated when rebuilding the model in the competition test phase
def config_factory():
    problem_params = {
        "lr": 1e-4,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 1,
        "n_heads": 8,
        "dim_feedforward": 256,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 16,
    }

    tst_config = TSTConfig(save_path="RawWaveformTst", num_classes=1, **problem_params)

    return tst_config


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = UniformLengthTst(
        **tst_config.generate_model_params(),
        feat_dim=ds.features_dim,
        max_len=ds.sample_len,
    )

    lightning_wrapper = GenericPlTst(tst, tst_config)

    return lightning_wrapper


def single_dl_factory(
    tst_config: TSTConfig, pids: list, data_path: str = None, **ds_args
):
    ds = SampleDataset(
        root_folder=data_path,
        patient_ids=pids,
        for_classification=True,
        normalize=False,
        **ds_args,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=config.cores_available,
        batch_size=tst_config.batch_size,
        # Only pin memory if we have GPUs
        pin_memory=(config.gpus_available > 0),
    )

    return dl


def dataloader_factory(
    tst_config: TSTConfig, data_path: str = "./data", deterministic_split=False
):
    pids = PatientDataset(root_folder=data_path).patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(pids, random_state=42)
    else:
        train_pids, valid_pids = train_test_split(pids)

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    return train_dl, valid_dl


if __name__ == "__main__":
    tst_config = config_factory()

    # wandb_logger = WandbLogger(
    #     project="physionet2023wandb",
    #     config=tst_config,
    #     group="RawWaveformTST",
    #     job_type="train",
    # )

    training_dl, valid_dl = dataloader_factory(tst_config)

    model = lightning_tst_factory(tst_config, training_dl.dataset)

    trainer = GenericPlTrainer(
        logger=None,
        val_check_interval=0.1,
        # log_every_n_steps=7,
        enable_progress_bar=True,
    )  # TODO: add logger when things are actually working

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
        val_dataloaders=valid_dl,
    )
