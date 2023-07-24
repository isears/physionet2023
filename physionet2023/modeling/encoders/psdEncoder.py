import skorch
from skorch.callbacks import EarlyStopping
import torch
from torch.nn.modules import TransformerEncoder
from physionet2023.dataProcessing.TuhDatasets import load_all_psd


class SimplePsdAutoencoder(torch.nn.Module):
    def __init__(self, n_channels: int, spectrum_size: int, encoding_dim: int) -> None:
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_channels * spectrum_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, encoding_dim),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_channels * spectrum_size),
            torch.nn.Tanh(),
        )

    def forward(self, X):
        flat_channels = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        z = self.encoder(flat_channels)
        x_hat = self.decoder(z)

        x_hat = x_hat.reshape(X.shape)

        return x_hat, z

class AutoEncoderNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred  # <- unpack the tuple that was returned by `forward`
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        loss_l1 = 1e-3 * torch.abs(encoded).sum()
        return loss_reconstruction + loss_l1

if __name__ == "__main__":
    X = load_all_psd()

    m =  AutoEncoderNet(
        SimplePsdAutoencoder,
        module__n_channels=X.shape[-2],
        module__spectrum_size = X.shape[-1],
        module__encoding_dim=16,
        max_epochs=100,
        lr=0.01,
        batch_size=64,
        callbacks=[EarlyStopping(patience=3)],
        criterion=torch.nn.MSELoss()
    )

    m.fit(X, y=X)