"""
Use all the original TST code
"""

import torch.utils.data
from mvtst.models.loss import MaskedMSELoss, NoFussCrossEntropyLoss
from mvtst.models.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)
from mvtst.optimizers import AdamW
from mvtst.running import SupervisedRunner, UnsupervisedRunner

from physionet2023 import config
from physionet2023.dataProcessing.datasets import SampleDataset

if __name__ == "__main__":
    ds = SampleDataset("./data", sample_len=1000)
    train_ds, test_ds = ds.noleak_traintest_split()

    print(f"Training dataset size: {len(train_ds)}")
    print(f"Testing dataset size: {len(test_ds)}")

    tst = TSTransformerEncoderClassiregressor(
        feat_dim=len(ds.channels) + len(ds.static_features),
        max_len=ds.sample_len,
        d_model=128,
        n_heads=16,
        num_layers=5,
        dim_feedforward=256,
        num_classes=1,
        dropout=0.02,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ).to("cuda")

    training_dl = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=train_ds.tst_collate,
        num_workers=config.cores_available,
        batch_size=8,
        pin_memory=True,
    )

    testing_dl = torch.utils.data.DataLoader(
        test_ds,
        collate_fn=test_ds.tst_collate,
        num_workers=config.cores_available,
        batch_size=8,
        pin_memory=True,
    )

    training_runner = SupervisedRunner(
        model=tst,
        dataloader=training_dl,
        device="cuda",
        loss_module=NoFussCrossEntropyLoss(reduction="none"),
        optimizer=AdamW(tst.parameters(), lr=1e-4, weight_decay=0),
    )

    testing_runner = SupervisedRunner(
        model=tst,
        dataloader=testing_dl,
        device="cuda",
        loss_module=NoFussCrossEntropyLoss(reduction="none"),
        optimizer=None,  # Shouldn't need this
    )

    for idx in range(0, 20):
        train_metrics = training_runner.train_epoch(epoch_num=idx)
        test_metrics, _ = testing_runner.evaluate(epoch_num=idx)
        print(f"\n{train_metrics}")
        print(test_metrics)
