import torch 
from torch import nn 


class PantsEncoder(nn.Module):
    def __init__(
            self,
            conv_filters: list = [64, 128, 256, 512, 512]
    ) -> None:
        
        super(PantsEncoder, self).__init__()
        
        enc_layers = []
        enc_layers.append(nn.Conv2d(3, conv_filters[0], kernel_size=4, stride=2, padding=1))

        in_channels = conv_filters[0]
        for filter in conv_filters[1:]:
            enc_layers.append(self._enc_block(in_channels, filter))
            in_channels = filter

        self.encoder = nn.Sequential(*enc_layers)
        del enc_layers
    
    def _enc_block(
            self,
            in_channels: int,
            out_channels: int
    ) -> nn.Sequential:

        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x)
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PantsEncoder().to(device)
    img = torch.randn(16, 3, 64, 64).to(device)
    enc = model(img)
    print('SUCCESS')
