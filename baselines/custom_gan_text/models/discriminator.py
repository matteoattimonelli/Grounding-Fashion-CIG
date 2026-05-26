# import torch
# import torch.nn as nn


# class Discriminator(nn.Module):
#     def __init__(
#             self,
#             in_channels: int = 3,
#             features: list = [64, 128, 256, 512]
#         ) -> None:
        
#         super(Discriminator, self).__init__()

#         self.initial = nn.Sequential(
#             nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
#             nn.LeakyReLU(0.2)
#         )

#         layers = []
#         in_channels = features[0]

#         for feature in features[1:]:
#             layers.append(
#                 self._block(in_channels, feature, stride=1 if feature == features[-1] else 2)
#             )

#             in_channels = feature
            
#         layers.append(
#             nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0)
#             )

#         self.model = nn.Sequential(*layers)

#     def _block(
#             self,
#             in_channels: int,
#             out_channels: int,
#             stride: int = 2
#         ) -> nn.Sequential:
        
#         return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
#                 nn.BatchNorm2d(out_channels),
#                 nn.LeakyReLU(0.2)
#             )

#     def forward(
#             self, 
#             x: torch.tensor, 
#             y: torch.tensor
#         ) -> torch.tensor:
        
#         x = torch.cat([x, y], dim=1)
#         x = self.initial(x)
#         x = self.model(x)
#         return x
    
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        features: list = [64, 128, 256, 512],
        text_embed_dim: int = 512,
        spatial_size: int = 64  # Input image size (assumes square)
    ) -> None:
        super(Discriminator, self).__init__()

        self.text_embed_dim = text_embed_dim
        self.spatial_size = spatial_size

        # Project CLIP embedding to a spatial tensor (B, 1, H, W)
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_dim, spatial_size * spatial_size),
            nn.ReLU(),
            nn.Unflatten(1, (1, spatial_size, spatial_size))
        )

        # New input channels: x (3), y (3), + text map (1) = 7 channels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                self._block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0))

        self.model = nn.Sequential(*layers)

    def _block(self, in_channels: int, out_channels: int, stride: int = 2) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(
        self,
        x: torch.Tensor,         # input image (e.g., condition or real)
        y: torch.Tensor,         # target or generated image
        text_embedding: torch.Tensor  # (B, 512)
    ) -> torch.Tensor:
        B = x.size(0)

        # Project text into (B, 1, H, W)
        text_map = self.text_projector(text_embedding)  # shape: (B, 1, H, W)

        # Ensure same spatial size
        if text_map.shape[-1] != x.shape[-1]:
            text_map = nn.functional.interpolate(text_map, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Concatenate inputs
        input = torch.cat([x, y, text_map], dim=1)  # shape: (B, 7, H, W)

        out = self.initial(input)
        out = self.model(out)
        return out
