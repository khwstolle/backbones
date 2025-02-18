from typing import Final

from laco.language import call, partial

from ._modules import BasicBlock, Bottleneck, ResNet

__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4D",
    "resnext101_32x8D",
    "resnext101_64x4D",
    "wide_resnet50",
    "wide_resnet101",
]


resnet18: Final = call(ResNet)(
    block=partial(BasicBlock)(),
    layers=(2, 2, 2, 2),
)


resnet34: Final = call(ResNet)(
    block=partial(BasicBlock)(),
    layers=(3, 4, 6, 3),
)


resnet50: Final = call(ResNet)(
    block=partial(Bottleneck)(),
    layers=(3, 4, 6, 3),
    expansion=4,
)


resnet101: Final = call(ResNet)(
    block=partial(Bottleneck)(),
    layers=(3, 4, 23, 3),
    expansion=4,
)


resnet152: Final = call(ResNet)(
    block=partial(Bottleneck)(),
    layers=(3, 8, 36, 3),
    expansion=4,
)


resnext50_32x4D: Final = call(ResNet)(
    block=partial(Bottleneck)(groups=32, group_width=4),
    layers=(3, 4, 6, 3),
    expansion=4,
)


resnext101_32x8D: Final = call(ResNet)(
    block=partial(Bottleneck)(groups=32, group_width=8),
    layers=(3, 4, 23, 3),
    expansion=4,
)


resnext101_64x4D: Final = call(ResNet)(
    block=partial(Bottleneck)(group_width=64),
    layers=(3, 4, 23, 3),
    expansion=4,
)


wide_resnet50: Final = call(ResNet)(
    block=partial(Bottleneck)(group_width=128),
    layers=(3, 4, 6, 3),
    expansion=4,
)


wide_resnet101: Final = call(ResNet)(
    block=partial(Bottleneck)(group_width=128),
    layers=(3, 4, 23, 3),
    expansion=4,
)
