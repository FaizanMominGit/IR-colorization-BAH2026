import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention Block to adaptively recalibrate feature channel weights."""
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block."""
    def __init__(self, num_channels, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True),
            ChannelAttention(num_channels, reduction)
        )

    def forward(self, x):
        return x + self.body(x)

class ResidualGroup(nn.Module):
    """Residual Group containing multiple RCAB blocks."""
    def __init__(self, num_channels, num_rcab=4, reduction=16):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(num_channels, reduction) for _ in range(num_rcab)]
        modules.append(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)

class ThermalSRNet(nn.Module):
    """
    Super-Resolution Deep Network for Thermal Infrared Imagery.
    Upscales input single-channel TIR images by 2x scale factor (200m -> 100m).
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_rg=3, num_rcab=4, scale_factor=2):
        super(ThermalSRNet, self).__init__()
        self.scale_factor = scale_factor

        # Head / Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # Body / Deep feature extraction
        modules_body = [ResidualGroup(num_features, num_rcab=num_rcab) for _ in range(num_rg)]
        modules_body.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

        # Tail / Upsampling layer using PixelShuffle
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True)
        )

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        res += x_head
        out = self.tail(res)
        return out
