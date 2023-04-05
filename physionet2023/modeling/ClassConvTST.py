import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import ConvTST, TSTransformerEncoderClassiregressor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from physionet2023 import config
from physionet2023.dataProcessing.patientDatasets import MetadataOnlyDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling import GenericPlRegressor, GenericPlTrainer, GenericPlTst


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = ConvTST(
        **tst_config.generate_model_params(),
        spectrogram_dims=ds.dims,
        feat_dim=ds.features_dim,
    )

    lightning_wrapper = GenericPlTst(tst, tst_config)

    return lightning_wrapper


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
    tst_config: TSTConfig,
    data_path: str = "./data",
    deterministic_split=False,
    test_size=0.1,
):
    metadata_ds = MetadataOnlyDataset(root_folder=data_path)

    patient_ids = [pids for pids, _, _ in metadata_ds]
    cpcs = [metadata["CPC"] for _, metadata, _ in metadata_ds]

    if deterministic_split:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

    train_idx, valid_idx = next(sss.split(patient_ids, cpcs))

    train_pids = [patient_ids[idx] for idx in train_idx]
    valid_pids = [patient_ids[idx] for idx in valid_idx]

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    for pid in train_dl.dataset.patient_ids:
        assert pid not in valid_dl.dataset.patient_ids

    return train_dl, valid_dl


def train_fn(data_path: str = "./data", log: bool = True, test=False):
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
            group="ConvTST_classifier",
            job_type="train",
        )
    else:
        logger = None

    trainer = GenericPlTrainer(
        "./cache/convTST", logger, enable_progress_bar=True, es_patience=7
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

    model.load_state_dict(trainer.get_best_params())

    trainer.test(model=model, dataloaders=valid_dl)

    return trainer.get_best_params()


if __name__ == "__main__":
    train_fn(data_path="./data", log=False, test=True)
