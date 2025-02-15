r"""
Uses symbolic tracing to remove redundant (e.g. classification) weights from a backbone
network.

Notes
-----
Deprecated our own FX-tracing in favor of the ``torchvision`` implementation, which
has adopted the same functionality.
"""

import torch
import torch.fx
import torch.nn

__all__ = ["extract_features"]


def extract_features(
    model: torch.nn.Module, features: dict[str, str] | list[str]
) -> torch.fx.GraphModule:
    from torchvision.models.feature_extraction import create_feature_extractor

    gm = create_feature_extractor(model, features)
    return gm
