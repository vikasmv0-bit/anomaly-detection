"""
models/mobilenet_extractor.py
------------------------------
MobileNetV2-based spatial feature extractor.

Extracts deep features from cropped object regions using a pretrained
MobileNetV2 model. The penultimate layer outputs a 1280-dim feature
vector which is projected down to SPATIAL_FEATURE_DIM (128) via a
learned linear layer.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("MobileNetExtractor")


class MobileNetFeatureExtractor(nn.Module):
    """MobileNetV2 feature extractor with learned projection.

    Args:
        output_dim:  Desired output feature dimension (default 128).
        pretrained:  Whether to load ImageNet pretrained weights.
        freeze_backbone: If True, freeze MobileNet weights (only train projection).
    """

    def __init__(
        self,
        output_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        proj_path: Optional[str] = None,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Load MobileNetV2
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)

        # Remove the classifier head — keep only features
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: 1280 → output_dim
        self.projection = nn.Sequential(
            nn.Linear(1280, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if proj_path:
            if os.path.exists(proj_path):
                self.projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
                logger.info("Loaded MobileNet projection weights from %s", proj_path)
            else:
                os.makedirs(os.path.dirname(proj_path), exist_ok=True)
                torch.save(self.projection.state_dict(), proj_path)
                logger.info("Generated and saved new MobileNet projection weights to %s", proj_path)

        # Standard ImageNet preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            "MobileNetV2 extractor ready — output_dim=%d, frozen=%s",
            output_dim, freeze_backbone,
        )

    def preprocess_crop(self, crop_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess a single crop tensor (C, H, W) with values in [0, 255]."""
        x = crop_tensor.float() / 255.0
        return self.transform(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            x: Tensor of shape (B, 3, 224, 224), normalized.

        Returns:
            Tensor of shape (B, output_dim).
        """
        with torch.no_grad():
            features = self.backbone(x)  # (B, 1280)
        projected = self.projection(features)  # (B, output_dim)
        return projected

    def extract_from_crops(
        self,
        crops: List[torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Extract features from a list of crop tensors.

        Args:
            crops: List of (3, H, W) tensors (uint8 or float).
            device: Target device.

        Returns:
            Tensor of shape (N, output_dim) or (1, output_dim) if no crops.
        """
        if not crops:
            return torch.zeros(1, self.output_dim, device=device)

        processed = []
        for crop in crops:
            proc = self.preprocess_crop(crop)
            processed.append(proc)

        batch = torch.stack(processed).to(device)  # (N, 3, 224, 224)
        return self.forward(batch)

    def extract_mean_feature(
        self,
        crops: List[torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Extract and mean-pool features across all crops.

        Returns:
            Tensor of shape (output_dim,).
        """
        features = self.extract_from_crops(crops, device)
        return features.mean(dim=0)  # (output_dim,)
