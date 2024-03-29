import torch
from mvtst.models import TSTConfig
from sklearn.model_selection import train_test_split

from physionet2023 import LabelType, PhysionetConfig, config
from physionet2023.dataProcessing.datasets import PatientDataset
from physionet2023.dataProcessing.recordingDatasets import (
    RecordingDataset,
    SpectrogramDataset,
)
from physionet2023.modeling import GenericPlTrainer, GenericPlTst


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=5, dilation=9
        )
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=5)
        self.conv2 = torch.nn.Conv1d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=5
        )
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=5)

        # TODO: this got hard-coded but maybe shouldn't be
        self.fc1 = torch.nn.Linear(in_features=21546, out_features=1000)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=1000, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        output = self.fc2(x)
        return output


def model_factory(tst_config, ds):
    cnn = SimpleCNN(len(ds.channels))

    lightning_wrapper = GenericPlTst(cnn, tst_config)

    return lightning_wrapper


# Need fn here so that identical configs can be generated when rebuilding the model in the competition test phase
def config_factory():
    problem_params = {
        "lr": 1e-5,
        "dropout": 0.1,
        "d_model_multiplier": 8,
        "num_layers": 1,
        "n_heads": 8,
        "dim_feedforward": 256,
        "pos_encoding": "learnable",
        "activation": "gelu",
        "norm": "LayerNorm",
        "optimizer_name": "AdamW",
        "batch_size": 64,
    }

    tst_config = PhysionetConfig(
        save_path="SpectrogramCNN", label_type=LabelType.SINGLECLASS, **problem_params
    )

    return tst_config


def single_dl_factory(
    tst_config: PhysionetConfig, pids: list, data_path: str = None, **ds_args
):
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
    tst_config: TSTConfig, data_path: str = None, deterministic_split=False
):
    pids = PatientDataset(root_folder=data_path).patient_ids

    if deterministic_split:
        train_pids, valid_pids = train_test_split(pids, random_state=1, test_size=0.1)
    else:
        train_pids, valid_pids = train_test_split(pids)

    train_dl = single_dl_factory(tst_config, train_pids, data_path)
    valid_dl = single_dl_factory(tst_config, valid_pids, data_path)

    return train_dl, valid_dl


def train_fn(data_path: str, log: bool = True):
    # torch.set_float32_matmul_precision("medium")
    tst_config = config_factory()

    train_dl, valid_dl = dataloader_factory(
        tst_config, data_path, deterministic_split=True
    )

    model = model_factory(tst_config, train_dl.dataset)

    # if log:
    #     logger = WandbLogger(
    #         project="physionet2023wandb",
    #         config=tst_config,
    #         group="ConvTST_classifier",
    #         job_type="train",
    #     )
    # else:
    #     logger = None

    trainer = GenericPlTrainer(
        save_path="cache/simpleCNN", enable_progress_bar=True, val_check_interval=0.1
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

    return trainer.get_best_params()


if __name__ == "__main__":
    train_fn(data_path="./data", log=False)
