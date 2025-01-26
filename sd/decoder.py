import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = SelfAttention(1, in_channels)
        self.norm = nn.GroupNorm(32, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Features, Height, Width)
        
        residue = x

        n, c, h, w = x.shape
        # (Batch_size, Features, Height, Width) -> (Batch_size, Features, Height * Width)
        x = x.view(n, c, h * w)
        
        # (Batch_size, Features, Height * Width) -> (Batch_size, Height * Width, Features)
        x = x.transpose(-1, -2)
        # (Batch_size, Height * Width, Features) -> (Batch_size, Height * Width, Features)
        x = self.attention(x)
        # (Batch_size, Height * Width, Features) -> (Batch_size, Features, Height * Width)
        x = x.transpose(-1, -2)
        # (Batch_size, Features, Height * Width) -> (Batch_size, Features, Height, Width)
        x = x.view((n, c, h, w))
        x += residue
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        if in_channels != out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Height, Width)
        
        residue = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + self.residual_layer(residue)
    

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height // 8, Width // 8) -> (Batch_size, 512, Height // 8, Width // 8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height // 8, Width // 8) -> (Batch_size, 512, Height // 4, Width // 4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height // 4, Width // 4) -> (Batch_size, 512, Height // 2, Width // 2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height // 2, Width // 2) -> (Batch_size, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, 4, Height // 8, Width // 8)
        x /= 0.10215

        for module in self:
            x = module(x)

        # (Batch_size, 3, Height, Width)
        return x