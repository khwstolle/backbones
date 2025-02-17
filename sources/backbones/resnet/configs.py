from typing import Final

from unipercept.config.language import call, partial

from ._modules import BasicBlock, Bottleneck, ResNet

__all__ = [
    "RESNET_18",
    "RESNET_34",
    "RESNET_50",
    "RESNET_101",
    "RESNET_152",
    "RESNEXT_50_32x4D",
    "RESNEXT_101_32x8D",
    "RESNEXT_101_64x4D",
    "WIDE_RESNET_50",
    "WIDE_RESNET_101",
]


RESNET_18: Final = call(ResNet)(
    block=partial(BasicBlock)(),
    layers=(2, 2, 2, 2),
)


RESNET_34: Final = call(ResNet)(
    block=partial(BasicBlock)(),
    layers=(3, 4, 6, 3),
)


RESNET_50: Final = call(ResNet)(
    block=partial(Bottleneck)(),
    layers=(3, 4, 6, 3),
    expansion=4,
)


RESNET_101: Final = call(ResNet)(
    block=partial(Bottleneck)(),
    layers=(3, 4, 23, 3),
    expansion=4,
)


RESNET_152: Final = call(ResNet)(
    block=partial(Bottleneck)(),
    layers=(3, 8, 36, 3),
    expansion=4,
)


RESNEXT_50_32x4D: Final = call(ResNet)(
    block=partial(Bottleneck)(groups=32, group_width=4),
    layers=(3, 4, 6, 3),
    expansion=4,
)


RESNEXT_101_32x8D: Final = call(ResNet)(
    block=partial(Bottleneck)(groups=32, group_width=8),
    layers=(3, 4, 23, 3),
    expansion=4,
)


RESNEXT_101_64x4D: Final = call(ResNet)(
    block=partial(Bottleneck)(group_width=64),
    layers=(3, 4, 23, 3),
    expansion=4,
)


WIDE_RESNET_50: Final = call(ResNet)(
    block=partial(Bottleneck)(group_width=128),
    layers=(3, 4, 6, 3),
    expansion=4,
)


WIDE_RESNET_101: Final = call(ResNet)(
    block=partial(Bottleneck)(group_width=128),
    layers=(3, 4, 23, 3),
    expansion=4,
)
