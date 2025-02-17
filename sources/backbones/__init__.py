r"""
Backbones
=========

A library of backbone architectures for neural networks, standardized for easy
use in research and development.

This library was created to solve the following problems.

1. Reduce boilerplate that comes with (efficiently) extracting features from
    some pre-trained network.

2. Implement compatability layers to port weights between different distributors
    of pre-trained weights.

3. Allow both training and inference of the sub-networks.
"""

from .resnet import _modules
from . import swin
from ._features import *
from ._interface import *
from ._io import *
from ._normalize import *

__version__ = "1.3.0"
