import skorch
import torch
from torch.nn.modules import TransformerEncoder


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
        z = self.encoder(X)
        x_hat = self.decoder(z)

        return x_hat


if __name__ == "__main__":
    m =  skorch.NeuralNet(
        SimplePsdAutoencoder,
        module__n_channels=4,
        module__hidden_dim=int((len(df.columns) - 3) / 2),
        max_epochs=100,
        lr=0.01,
        batch_size=64,
        callbacks=[EarlyStopping(patience=3)]
    )