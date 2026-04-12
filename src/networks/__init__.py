"""
Neural network architectures for mammogram analysis.

This module contains:
- CRAUNet: Coordinate-aware Regression Attention U-Net for MLO landmark detection
- CCCRAUNet: Coordinate-aware Regression Attention U-Net for CC landmark detection
"""

from .landmark_models import CRAUNet, CCCRAUNet

__all__ = ['CRAUNet', 'CCCRAUNet']
