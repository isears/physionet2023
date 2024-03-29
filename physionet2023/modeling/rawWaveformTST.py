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
from physionet2023.modeling import GenericPlRegressor, GenericPlTrainer, GenericPlTst
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

    lightning_wrapper = GenericPlRegressor(tst, tst_config)

    return lightning_wrapper


def single_dl_factory(
    tst_config: TSTConfig, pids: list, data_path: str = None, **ds_args
):
    ds = FftDataset(
        root_folder=data_path,
        patient_ids=pids,
        for_classification=False,
        normalize=True,
        sample_len=1000,
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
    tst_config: TSTConfig, data_path: str = "./data", deterministic_split=True
):
    pids = PatientDataset(root_folder=data_path).patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(pids, random_state=42, test_size=0.1)
    else:
        train_pids, valid_pids = train_test_split(pids, test_size=0.1)

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    return train_dl, valid_dl


def train_fn(data_folder, log):
    tst_config = config_factory()

    # wandb_logger = WandbLogger(
    #     project="physionet2023wandb",
    #     config=tst_config,
    #     group="RawWaveformTST",
    #     job_type="train",
    # )

    training_dl, valid_dl = dataloader_factory(tst_config, data_path=data_folder)

    model = lightning_tst_factory(tst_config, training_dl.dataset)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=10,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=config.gpus_available,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=15),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        val_check_interval=0.005,
        logger=None,
    )  # TODO: add logger when things are actually working

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
        val_dataloaders=valid_dl,
    )

    return torch.load(checkpoint_callback.best_model_path)["state_dict"]


if __name__ == "__main__":
    train_fn("./data", None)
