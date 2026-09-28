import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            features: list = [64, 128, 256, 512]
    ) -> None:
        
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                self._block(in_channels, feature, stride=2)
            )

            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        layers.append(
            nn.Sigmoid()
        )

        self.model = nn.Sequential(*layers)

    def _block(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(
            self,
            x: torch.tensor,
            y: torch.tensor
    ) -> torch.tensor:

        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disc = Discriminator().to(device)
    x = torch.randn(1, 3, 64, 64).to(device)
    out = disc(x, x)
    print('SUCCESS')
