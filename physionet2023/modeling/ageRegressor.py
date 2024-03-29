import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import ConvTST, TSTransformerEncoderClassiregressor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from physionet2023 import config
from physionet2023.dataProcessing.datasets import PatientDataset
from physionet2023.dataProcessing.recordingDatasets import (
    SpectrogramAgeDataset,
    SpectrogramDataset,
)
from physionet2023.modeling import GenericPlRegressor, GenericPlTrainer, GenericPlTst


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = ConvTST(
        **tst_config.generate_model_params(),
        spectrogram_dims=ds.dims,
        feat_dim=ds.features_dim,
    )

    lightning_wrapper = GenericPlRegressor(tst, tst_config)

    lightning_wrapper.scorers = [
        MeanAbsoluteError().to("cuda"),
        MeanSquaredError().to("cuda"),
    ]

    return lightning_wrapper


# Need fn here so that identical configs can be generated when rebuilding the model in the competition test phase
def config_factory():
    problem_params = {
        "lr": 1e-3,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 1,
        "n_heads": 8,
        "dim_feedforward": 256,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 8,
    }

    tst_config = TSTConfig(save_path="ConvTst", num_classes=1, **problem_params)

    return tst_config


def single_dl_factory(
    tst_config: TSTConfig, pids: list, data_path: str = "./data", **ds_args
):
    ds = SpectrogramDataset(
        root_folder=data_path,
        patient_ids=pids,
        for_classification=False,
        normalize=True,
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
        train_pids, valid_pids = train_test_split(pids, random_state=42, test_size=0.1)
    else:
        train_pids, valid_pids = train_test_split(pids, test_size=0.1)

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    for pid in train_dl.dataset.patient_ids:
        assert pid not in valid_dl.dataset.patient_ids

    return train_dl, valid_dl


def train_fn(data_path: str = "./data", log: bool = True):
    # torch.set_float32_matmul_precision("medium")
    tst_config = config_factory()

    train_dl, valid_dl = dataloader_factory(
        tst_config, data_path, deterministic_split=True
    )

    model = lightning_tst_factory(tst_config, train_dl.dataset)

    if log:
        logger = WandbLogger(
            project="physionet2023wandb",
            config=tst_config,
            group="ConvTST_age_regressor",
            job_type="train",
        )
    else:
        logger = None

    trainer = GenericPlTrainer(logger, enable_progress_bar=True)

    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

    return trainer.get_best_params()


if __name__ == "__main__":
    params = train_fn(data_path="./data", log=False)
    torch.save(params, "./cache/age_models/convTST.pt")
