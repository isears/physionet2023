import pytorch_lightning as pl
import torch
from mvtst.models import TSTConfig
from mvtst.models.loss import NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import (
    ConvTST,
    TSTransformerEncoderClassiregressor,
    _get_activation_fn,
)
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from physionet2023 import LabelType, PhysionetConfig, config
from physionet2023.dataProcessing.patientDatasets import MetadataOnlyDataset
from physionet2023.dataProcessing.recordingDatasets import RecordingDataset
from physionet2023.modeling import GenericPlTrainer, GenericPlTst


class OneDimensionalConvTST(torch.nn.Module):
    def __init__(
        self,
        max_len,
        feat_dim,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        num_classes,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ):
        super(OneDimensionalConvTST, self).__init__()

        self.act = _get_activation_fn(activation)
        self.conv1 = torch.nn.Conv1d(
            in_channels=feat_dim, out_channels=feat_dim, kernel_size=5, dilation=9
        )
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=5)

        self.conv2 = torch.nn.Conv1d(
            in_channels=feat_dim, out_channels=feat_dim, kernel_size=5, dilation=9
        )
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=5)

        # Need to determine the "seq_len" after resultant convolutions w/dummy input to instantiate TST
        conv_test = self._forward_conv(torch.rand(1, feat_dim, max_len))
        post_conv_len = conv_test.shape[-1]

        self.tst = TSTransformerEncoderClassiregressor(
            feat_dim,  # i.e. number of EEG channels
            post_conv_len,  # i.e. final dim of output after convolutions
            d_model,
            n_heads,
            num_layers,
            dim_feedforward,
            num_classes,
            dropout,
            pos_encoding,
            activation,
            norm,
            freeze,
        )

    def _forward_conv(self, X):
        output = self.conv1(X)
        output = self.act(output)
        output = self.maxpool1(output)

        output = self.conv2(output)
        output = self.act(output)
        output = self.maxpool2(output)

        output = torch.flatten(output, 2)

        return output

    def forward(self, X):
        """
        X (EEG spectrogram): (batch_size, n_channels, freq_dim, time_dim)
        """
        output = self._forward_conv(X)

        # Original TST expects (batch_size, seq_len, feat_dim)
        output = output.permute(0, 2, 1)
        padding_masks = torch.ones_like(output[:, :, 0]).bool()

        return self.tst.forward(output, padding_masks)


def lightning_tst_factory(tst_config: TSTConfig, ds):
    tst = OneDimensionalConvTST(
        **tst_config.generate_model_params(),
        max_len=ds.full_record_len,
        feat_dim=len(ds.channels),
    )

    lightning_wrapper = GenericPlTst(tst, tst_config)

    return lightning_wrapper


# Need fn here so that identical configs can be generated when rebuilding the model in the competition test phase
def config_factory():
    problem_params = {
        "lr": 1e-5,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 2,
        "n_heads": 8,
        "dim_feedforward": 256,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 16,
    }

    tst_config = PhysionetConfig(
        save_path="ConvTst", label_type=LabelType.SINGLECLASS, **problem_params
    )

    return tst_config


def single_dl_factory(
    tst_config: PhysionetConfig, pids: list, data_path: str = "./data", **ds_args
) -> torch.utils.data.DataLoader:
    ds = RecordingDataset(
        root_folder=data_path,
        patient_ids=pids,
        label_type=tst_config.label_type,
        preprocess=True,
        # include_static=False,
        quality_cutoff=0.0,
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
    tst_config: PhysionetConfig,
    data_path: str = "./data",
    deterministic_split=False,
    test_size=0.1,
):
    pids = MetadataOnlyDataset(root_folder=data_path).patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(
            pids, random_state=1, test_size=test_size
        )
    else:
        train_pids, valid_pids = train_test_split(pids, test_size=test_size)

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
        "./cache/convTST",
        logger,
        enable_progress_bar=True,
        es_patience=5,
        val_check_interval=0.1,
        max_epochs=5,
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
