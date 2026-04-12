"""
Landmark Detection Neural Network Architectures.

This module contains the CRAUNet (Coordinate-aware Regression Attention U-Net)
architecture used for detecting anatomical landmarks in mammogram images.

Two versions are provided:
- MLOCRAUNet: For MLO view (3 landmarks = 6 outputs)
- CCCRAUNet: For CC view (1 landmark = 2 outputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """Upsampling block with convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AddCoordinates(nn.Module):
    """Layer to add X,Y coordinate channels at runtime for spatial awareness."""
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = input_tensor.size()
        x_coords = torch.linspace(0, 1, width, device=input_tensor.device).repeat(batch_size, height, 1)
        y_coords = torch.linspace(0, 1, height, device=input_tensor.device).repeat(batch_size, width, 1).transpose(1, 2)
        coords = torch.stack((x_coords, y_coords), dim=1)
        return torch.cat((input_tensor, coords), dim=1)


class CoordConv(nn.Module):
    """CoordConv layer that adds coordinate channels before convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(CoordConv, self).__init__()
        self.add_coords = AddCoordinates()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionBlock(nn.Module):
    """Attention gate block for focusing on relevant features."""
    
    def __init__(self, F_g: int, F_l: int, n_coefficients: int):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi


class CRAUNet(nn.Module):
    """
    Coordinate-aware Regression Attention U-Net for MLO landmark detection.
    
    This is the original architecture matching the trained checkpoint.
    Uses CoordConv in first layer, attention gates, and regression head.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_features: Number of output features (6 for MLO: 3 landmarks x 2 coords)
    """
    
    def __init__(self, in_channels: int = 1, out_features: int = 6):
        super(CRAUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder with CoordConv first layer
        self.Conv1 = CoordConv(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder with attention
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        # Final layers for regression (matching checkpoint structure)
        self.last = nn.Conv2d(64, out_features, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        # Decoder path with attention
        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.UpConv2(d2)

        # Regression output
        out = self.last(d2)
        out = F.adaptive_avg_pool2d(out, (16, 16))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


class CCCRAUNet(nn.Module):
    """
    Coordinate-aware Regression Attention U-Net for CC landmark detection.
    
    Similar to CRAUNet but with different FC layer structure matching CC checkpoint.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_features: Number of output features (2 for CC: 1 landmark x 2 coords)
    """
    
    def __init__(self, in_channels: int = 1, out_features: int = 2):
        super(CCCRAUNet, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        # Encoder with coordinate convolution
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = nn.Sequential(
            CoordConv(in_channels, filters[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])

        # Decoder with attention
        self.Up5 = UpConv(filters[4], filters[3])
        self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], n_coefficients=filters[2])
        self.Up_conv5 = ConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], n_coefficients=filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], n_coefficients=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], n_coefficients=32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])

        # Global pooling and regression head (matching CC checkpoint)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters[0], 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # Global pooling and regression
        out = self.global_pool(d2)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
