import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SemanticThermalGuidance(nn.Module):
    """
    Auxiliary module extracting semantic land-cover features from thermal gradients
    to guide color assignment (preventing color bleeding across boundaries).
    """
    def __init__(self, in_channels=1, feature_dim=64):
        super(SemanticThermalGuidance, self).__init__()
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gradient_conv(x)

class ThermalColorizerNet(nn.Module):
    """
    U-Net based generator architecture with Semantic Guidance for Thermal-to-RGB Translation.
    Input: (B, 1, H, W) single-channel super-resolved TIR image.
    Output: (B, 3, H, W) multi-channel RGB image (Channels 0, 1, 2 representing Blue, Green, Red).
    """
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(ThermalColorizerNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic Guidance Encoder
        self.semantic_guide = SemanticThermalGuidance(in_channels, features[0])

        # Down part of U-Net
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid() # Normalizes RGB outputs in range [0, 1]
        )

    def forward(self, x):
        skip_connections = []

        # Thermal feature guidance gating on initial layer
        guide = self.semantic_guide(x)

        out = x
        for idx, down in enumerate(self.downs):
            out = down(out)
            if idx == 0:
                out = out * guide # Apply semantic feature modulation
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            out = self.ups[idx](out)
            skip_connection = skip_connections[idx // 2]

            if out.shape != skip_connection.shape:
                out = F.interpolate(out, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, out), dim=1)
            out = self.ups[idx + 1](concat_skip)

        return self.final_conv(out)
