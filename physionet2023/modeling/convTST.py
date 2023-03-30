import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import ConvTST, TSTransformerEncoderClassiregressor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from physionet2023 import config
from physionet2023.dataProcessing.datasets import PatientDataset
from physionet2023.dataProcessing.recordingDatasets import SpectrogramDataset
from physionet2023.modeling.scoringUtil import (
    ClassifierAUROC,
    ClassifierCompetitionScore,
    CompetitionScore,
    RegressorAUROC,
)


class plConvTst(pl.LightningModule):
    def __init__(self, tst: ConvTST, tst_config: TSTConfig) -> None:
        super().__init__()
        # TODO: damned if you do damned if you don't
        # self.save_hyperparameters(ignore=["tst"])
        self.tst = tst

        self.tst_config = tst_config
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.auroc_scorer = ClassifierAUROC()
        self.competition_scorer = ClassifierCompetitionScore()

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = torch.squeeze(self.tst(X))

        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss, batch_size=self.tst_config.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = torch.squeeze(self.tst(X))

        loss = self.loss_fn(preds, y)
        self.auroc_scorer.update(preds, y)
        self.competition_scorer.update(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        # val_loss = torch.tensor(validation_step_outputs).mean()
        val_auroc = self.auroc_scorer.compute()
        val_competition_score = self.competition_scorer.compute()

        self.log("Validation AUC", val_auroc)
        self.log("Validation Competition Score", val_competition_score)

        self.auroc_scorer.reset()
        self.competition_scorer.reset()

        print(
            f"\nValidation auroc: {val_auroc:.4}, competition: {val_competition_score}"
        )

    def forward(self, X):
        preds = self.tst(X)
        return torch.nn.functional.softmax(preds, dim=1)

    def configure_optimizers(self):
        return self.tst_config.generate_optimizer(self.parameters())


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = ConvTST(
        **tst_config.generate_model_params(),
        spectrogram_dims=ds.dims,
        feat_dim=ds.features_dim,
    )

    lightning_wrapper = plConvTst(tst, tst_config)

    return lightning_wrapper


# Need fn here so that identical configs can be generated when rebuilding the model in the competition test phase
def config_factory():
    problem_params = {
        "lr": 0.0018730765828665077,
        "dropout": 0.06384001841146537,
        "d_model_multiplier": 1,
        "num_layers": 2,
        "n_heads": 4,
        "dim_feedforward": 128,
        "batch_size": 16,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "PlainRAdam",
        "weight_decay": 0.01,
    }

    tst_config = TSTConfig(save_path="ConvTst", num_classes=5, **problem_params)

    return tst_config


def single_dl_factory(
    tst_config: TSTConfig, pids: list, data_path: str = None, **ds_args
):
    ds = SpectrogramDataset(
        root_folder=data_path, patient_ids=pids, for_classification=True, **ds_args
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
    tst_config: TSTConfig, data_path: str = None, deterministic_split=False
):
    pids = PatientDataset(root_folder=data_path).patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(pids, random_state=42)
    else:
        train_pids, valid_pids = train_test_split(pids)

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    return train_dl, valid_dl


def train_fn(data_path: str, log: bool = True):
    tst_config = config_factory()

    if log:
        logger = WandbLogger(
            project="physionet2023wandb",
            config=tst_config,
            group="ConvTST_classifier",
            job_type="train",
        )
    else:
        logger = None

    training_dl, valid_dl = dataloader_factory(tst_config, data_path=data_path)

    model = lightning_tst_factory(tst_config, training_dl.dataset)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    if config.gpus_available > 0:
        accelerator = "gpu"
        devices = config.gpus_available
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=10,
        gradient_clip_val=4.0,
        gradient_clip_algorithm="norm",
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=10,
                check_finite=False,
            ),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        # For when doing sample-based datasets
        val_check_interval=0.1,
        # log_every_n_steps=7,
        logger=logger,
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dl,
        val_dataloaders=valid_dl,
    )

    return torch.load(checkpoint_callback.best_model_path)["state_dict"]


if __name__ == "__main__":
    train_fn(data_path="./data", log=False)
