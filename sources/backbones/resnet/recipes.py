r"""
Build recipes for creating a backbone module from a name and corresponding weights.
"""

from functools import partial
from ._modules import BasicBlock, Bottleneck, ResNet


def resnet18() -> ResNet:
    """
    ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    """
    return ResNet(
        BasicBlock,
        (2, 2, 2, 2),
    )


def resnet34() -> ResNet:
    r"""
    ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    """
    return ResNet(
        BasicBlock,
        (3, 4, 6, 3),
    )


def resnet50() -> ResNet:
    r"""
    ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    """
    return ResNet(
        Bottleneck,
        (3, 4, 6, 3),
    )


def resnet101() -> ResNet:
    r"""
    ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    """
    return ResNet(
        partial(Bottleneck)(3, 4, 23, 3),
    )


def resnet152() -> ResNet:
    r"""
    ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    """
    return ResNet(
        partial(Bottleneck),
        (3, 8, 36, 3),
    )


def resnext50_32x4d() -> ResNet:
    r"""
    ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return ResNet(
        partial(Bottleneck, groups=32, group_width=4),
        (3, 4, 6, 3),
    )


def resnext101_32x8d() -> ResNet:
    r"""
    ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return ResNet(
        partial(Bottleneck, groups=32, group_width=8),
        (3, 4, 23, 3),
    )


def resnext101_64x4d() -> ResNet:
    r"""
    ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return ResNet(
        partial(Bottleneck, group_width=64),
        (3, 4, 23, 3),
    )


def wide_resnet50() -> ResNet:
    """Wide ResNet-50 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    """
    return ResNet(
        partial(Bottleneck, group_width=128),
        (3, 4, 6, 3),
    )


def wide_resnet101() -> ResNet:
    """Wide ResNet-101 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    """
    return ResNet(
        partial(Bottleneck, group_width=128),
        (3, 4, 23, 3),
    )
