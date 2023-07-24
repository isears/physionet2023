import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(  # like the Composition layer you built
            # Input: (4, 58, 171)
            torch.nn.Conv2d(4, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(4, 16, 2, stride=2, output_padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 4, 2, stride=2, output_padding=(0, 1)),
            torch.nn.Sigmoid()
            # torch.nn.Sigmoid()
        )

    def forward(self, X):
        z = self.encoder(X)
        print(f"encoded dim: {z.shape}")
        xhat = self.decoder(z)
        print(f"Final dim: {xhat.shape}")

        return xhat


if __name__ == "__main__":
    X = torch.zeros((128, 4, 58, 171))
    m = TestModel()
    m(X)

    print("done")
