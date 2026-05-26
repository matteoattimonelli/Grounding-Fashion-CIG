import torch 
from torch import nn 
import numpy as np 


import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        conv_filters: list = [64, 128, 256, 512, 512, 512],
        text_embed_dim: int = 512
    ) -> None:
        super(Generator, self).__init__()

        self.text_embed_dim = text_embed_dim

        enc_layers = []
        dec_layers = []

        enc_layers.append(nn.Conv2d(3, conv_filters[0], kernel_size=4, stride=2, padding=1))
        in_channels = conv_filters[0]

        for filter in conv_filters[1:-1]:
            enc_layers.append(self._enc_block(in_channels, filter))
            in_channels = filter

        self.final_enc = self._enc_block(in_channels, conv_filters[-1])
        self.encoder = nn.Sequential(*enc_layers)

        # Project CLIP text embedding to match bottleneck shape
        # self.text_projector = nn.Sequential(
        #     nn.Linear(text_embed_dim, conv_filters[-1]),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (conv_filters[-1], 1, 1))  # (B, C, 1, 1)
        # )

        # Decoder begins with text-conditioned bottleneck
        self.m = self._dec_block(conv_filters[-1] * 2, conv_filters[-1])  # bottleneck + text

        conv_filters.reverse()
        in_channels = conv_filters[0]

        for filter in conv_filters[2:]:
            dec_layers.append(self._dec_block(in_channels, filter))
            in_channels = filter

        dec_layers.append(self._out_block(in_channels))
        self.decoder = nn.Sequential(*dec_layers)

    def _enc_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def _dec_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(out_channels)
        )

    def _out_block(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        final_enc = self.final_enc(enc)  # shape: (B, C, 1, 1)

        B, C = text_embedding.shape
        # Fuse CLIP text + bottleneck
        fused = torch.cat([final_enc, text_embedding.view(B, C, 1, 1)], dim=1)  # shape: (B, C*2, 1, 1)

        d1 = self.m(fused)
        dec = self.decoder(d1)

        return enc, d1, dec



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = Generator().to(device)
    img = torch.randn(32, 3, 64, 64).to(device)
    enc, d1, dec = model(img)
    print('SUCCESS')
