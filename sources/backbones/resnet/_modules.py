r"""
Modules for building ResNet-like models.
"""

import typing
from collections import OrderedDict

from torch import Tensor, nn

__all__ = ["ResNet", "BasicBlock", "Bottleneck"]

type NormType = typing.Callable[[int], nn.Module]
type ActivationType = typing.Callable[[], nn.Module]


class InplaceReLU(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


DEFAULT_NORM: typing.Final[NormType] = nn.BatchNorm2d
DEFAULT_ACTIVATION: typing.Final[ActivationType] = InplaceReLU


class BlockProtocol(typing.Protocol):
    def __new__(  # noqa: PLR0913
        cls,
        dim_in: int,
        dim_out: int,
        stride: int,
        residual: nn.Module | None,
        dilation: int,
        expansion: int,
        /,
        *,
        norm: NormType,
        activation: ActivationType,
        **kwargs: typing.Any,
    ) -> nn.Module: ...


class BasicBlock(nn.Module):
    stride: typing.Final[int]

    def __init__(  # noqa: PLR0913
        self,
        dim_in: int,
        dim_out: int,
        stride: int,
        residual: nn.Module | None,
        dilation: int = 1,
        expansion: int = 1,
        /,
        *,
        norm: NormType = DEFAULT_NORM,
        activation: ActivationType = DEFAULT_ACTIVATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # noqa: W291

        if dilation != 1:
            msg = (
                f"{self.__class__.__name__} does not support {dilation=} (expected 1)."
            )
            raise ValueError(msg)
        if expansion != 1:
            msg = (
                f"{self.__class__.__name__} does not support {expansion=} (expected 1)."
            )
            raise ValueError(msg)

        self.conv1 = _build_conv2d_33(dim_in, dim_out, stride)
        self.norm1 = norm(dim_out)
        self.activation = activation()
        self.conv2 = _build_conv2d_33(dim_out, dim_out)
        self.norm2 = norm(dim_out)
        self.residual = residual if residual is not None else nn.Identity()
        self.stride = stride

    @typing.override
    def forward(self, x: Tensor) -> Tensor:
        res = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)

        return self.activation(out + res)


class Bottleneck(nn.Module):
    stride: typing.Final[int]

    def __init__(  # noqa: PLR0913
        self,
        dim_in: int,
        dim_out: int,
        stride: int,
        residual: nn.Module | None,
        dilation: int,
        expansion: int,
        /,
        groups: int = 1,
        group_width: int = 64,
        *,
        norm: NormType = DEFAULT_NORM,
        activation: ActivationType = DEFAULT_ACTIVATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        width = int(dim_out * (group_width / 64.0)) * groups

        self.conv1 = _build_linear2d(dim_in, width)
        self.norm1 = norm(width)
        self.conv2 = _build_conv2d_33(width, width, stride, groups, dilation)
        self.norm2 = norm(width)
        self.conv3 = _build_linear2d(width, dim_out * expansion)
        self.norm3 = norm(dim_out * expansion)
        self.activation = activation()
        self.residual = residual if residual is not None else nn.Identity()
        self.stride = stride

    @typing.override
    def forward(self, x: Tensor) -> Tensor:
        res = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        return self.activation(out + res)


class ResNet(nn.Module):
    def __init__(  # noqa: C901, PLR0913
        self,
        block: type[BlockProtocol],
        layers: tuple[int, int, int, int],
        *,
        expansion: int = 1,
        groups: int = 1,
        group_width: int = 64,
        norm: NormType = DEFAULT_NORM,
        activation: ActivationType = DEFAULT_ACTIVATION,
        num_classes: int | None = None,
        use_dilation: list[bool] | None = None,
    ) -> None:
        super().__init__()

        inplanes = 64
        self.dilation = 1
        if use_dilation is None:
            use_dilation = [False, False, False]
        if len(use_dilation) != 3:  # noqa: PLR2004
            msg = (
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {use_dilation}"
            )
            raise ValueError(msg)
        self.groups = groups
        self.group_width = group_width

        self.stem = nn.Sequential(
            OrderedDict(
                {
                    "conv": nn.Conv2d(
                        3, inplanes, kernel_size=7, stride=2, padding=3, bias=False
                    ),
                    "norm": norm(inplanes),
                    "activation": activation(),
                    "pool": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                }
            )
        )

        inplanes = 64
        dilation = 1

        def _build_layer(
            planes: int,
            blocks: int,
            stride: int,
            dilate: bool,
        ) -> nn.Sequential:
            nonlocal inplanes, dilation

            residual = None
            previous_dilation = dilation
            if dilate:
                dilation *= stride
                stride = 1
            if stride != 1 or inplanes != planes * expansion:
                residual = nn.Sequential(
                    OrderedDict(
                        {
                            "conv": _build_linear2d(
                                inplanes, planes * expansion, stride
                            ),
                            "norm": norm(planes * expansion),
                        }
                    )
                )

            layers = nn.Sequential()
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride,
                    residual,
                    previous_dilation,
                    expansion,
                    norm=norm,
                    activation=activation,
                )
            )
            inplanes = planes * expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        inplanes,
                        planes,
                        1,
                        None,
                        dilation,
                        expansion,
                        norm=norm,
                        activation=activation,
                    )
                )

            return layers

        self.ext1 = _build_layer(64, layers[0], 1, False)
        self.ext2 = _build_layer(128, layers[1], 2, use_dilation[0])
        self.ext3 = _build_layer(256, layers[2], 2, use_dilation[1])
        self.ext4 = _build_layer(512, layers[3], 2, use_dilation[2])

        if num_classes is not None:
            self.head = nn.Sequential(
                OrderedDict(
                    {
                        "pool": nn.AdaptiveAvgPool2d((1, 1)),
                        "flat": nn.Flatten(1),
                        "proj": nn.Linear(512 * expansion, num_classes),
                    }
                )
            )
        else:
            self.head = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Bottleneck) and m.norm3.weight is not None:
                nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock) and m.norm2.weight is not None:
                nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

    def _forward_resnet(self, x: Tensor) -> Tensor:
        for layer in (
            self.stem,
            self.ext1,
            self.ext2,
            self.ext3,
            self.ext4,
            self.head,
        ):
            x = layer(x)
        return x

    @typing.override
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_resnet(x)


def _build_conv2d_33(
    dim_in: int, dim_out: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    r"""
    Create a 3x3 convolution layer.
    """
    return nn.Conv2d(
        dim_in,
        dim_out,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _build_linear2d(dim_in: int, dim_out: int, stride: int = 1) -> nn.Conv2d:
    r"""
    Create a pointwise convolution layer.
    """
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)
