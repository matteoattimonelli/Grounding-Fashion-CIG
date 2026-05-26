import torch
from torch import nn


class GeneratorSkip(nn.Module):
    def __init__(
            self,
            conv_filters: list = [64, 128, 256, 512, 512, 512],
            ) -> None:
        super(GeneratorSkip, self).__init__()
        
        self.e1 = nn.Conv2d(3, conv_filters[0], kernel_size=4, stride=2, padding=1)
        self.e2 = self._enc_block(in_channels=64, out_channels=128)
        self.e3 = self._enc_block(in_channels=128, out_channels=256)
        self.e4 = self._enc_block(in_channels=256, out_channels=512)
        self.e5 = self._enc_block(in_channels=512, out_channels=512)
        self.e6 = self._enc_block(in_channels=512, out_channels=512)

        self.d1 = self._dec_block(in_channels=512, out_channels=512)
        self.d2 = self._dec_block(in_channels=512*2, out_channels=512)
        self.d3 = self._dec_block(in_channels=512*2, out_channels=256)
        self.d4 = self._dec_block(in_channels=256+256, out_channels=128)
        self.d5 = self._dec_block(in_channels=128+128, out_channels=64)
        self.d6 = self._out_block(in_channels=64+64)

    def _enc_block(
            self,
            in_channels: int,
            out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def _dec_block(
            self,
            in_channels: int,
            out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(out_channels)
        )
    
    def _out_block(
            self,
            in_channels: int,
    ) -> nn.Sequential:
        
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d1_pre = self.d1(e6)
        d2 = self.d2(torch.hstack([d1_pre, e5])) 
        d3 = self.d3(torch.hstack([d2, e4]))
        d4 = self.d4(torch.hstack([d3, e3]))
        d5 = self.d5(torch.hstack([d4, e2]))
        d6 = self.d6(torch.hstack([d5, e1]))

        return e5, d1_pre, d6


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GeneratorSkip().to(device)
    img = torch.randn(16, 3, 64, 64).to(device)
    enc, d1, dec = model(img)
    print('SUCCESS')
