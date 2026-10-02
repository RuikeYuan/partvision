"""Model: a pretrained backbone + a separate classification head.

Keeping the head separate lets us
  * expose the pooled backbone features as an embedding for kNN search,
  * freeze the backbone for a head-only warm-up,
  * give backbone and head different learning rates.

Backbones: torchvision ResNet / ConvNeXt / EfficientNetV2, and DINOv2 ViT via timm
(optional dependency, imported only when a DINOv2 arch is requested).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

RESNETS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
CONVNEXTS = ("convnext_tiny", "convnext_small")
EFFICIENTNETS = ("efficientnet_v2_s", "efficientnet_v2_m")
# our arch name -> timm model name
DINOV2 = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
          "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m"}
SUPPORTED = RESNETS + CONVNEXTS + EFFICIENTNETS + tuple(DINOV2)


def build_backbone(arch: str, pretrained: bool, image_size: int = 224) -> tuple[nn.Module, int]:
    """Return (backbone that outputs pooled features, embedding dim)."""
    if arch not in SUPPORTED:
        raise ValueError(f"arch must be one of {SUPPORTED}")
    if arch in DINOV2:
        import timm
        if image_size % 14:
            raise ValueError(f"DINOv2 needs image_size divisible by 14 (got {image_size})")
        # num_classes=0 -> forward() returns the pooled CLS token
        backbone = timm.create_model(DINOV2[arch], pretrained=pretrained, num_classes=0, img_size=image_size)
        return backbone, backbone.num_features
    backbone = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    if arch in RESNETS:
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        # ConvNeXt: classifier = [LayerNorm2d, Flatten, Linear]; EfficientNet: [Dropout, Linear]
        dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
        if arch in EFFICIENTNETS:
            backbone.classifier[0] = nn.Identity()   # our head has its own dropout
    return backbone, dim


class PartNet(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool = False, dropout: float = 0.2,
                 image_size: int = 224):
        super().__init__()
        self.backbone, self.embed_dim = build_backbone(arch, pretrained, image_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embed_dim, num_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def param_groups(self, lr: float, backbone_mult: float) -> list[dict]:
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": lr * backbone_mult},
            {"params": list(self.head.parameters()), "lr": lr},
        ]


def load_partnet(state_dict: dict, arch: str, num_classes: int, image_size: int = 224) -> PartNet:
    model = PartNet(arch, num_classes, pretrained=False, image_size=image_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model
