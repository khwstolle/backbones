r"""
Uses symbolic tracing to remove redundant (e.g. classification) weights from a backbone
network.

Notes
-----
Deprecated in favor of the ``torchvision`` implementation, which has adopted the same
functionality.
"""

from torchvision.models.feature_extraction import create_feature_extractor

__all__ = ["create_feature_extractor"]

