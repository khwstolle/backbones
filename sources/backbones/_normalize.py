r"""
Normalization transforms that transform input data to match the image statistics
used to train some backbone network.
"""

import json
from collections.abc import Callable
from typing import Any, Final, Self, override

import PIL.Image as pil_image
from torch import Tensor
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as T

__all__ = ["Normalize", "Denormalize"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

type RGB = tuple[float, float, float]


class Normalize(Transform):
    mean: Final[RGB]
    std: Final[RGB]
    inplace: Final[bool]

    def __init__(self, mean: RGB, std: RGB, inplace: bool = False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    @override
    def check_inputs(self, flat_inputs: Any) -> Any:
        if _check_types_any(flat_inputs, pil_image.Image):
            msg = f"{type(self).__name__}() does not support PIL images."
            raise TypeError(msg)

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(
            T.normalize, inpt, mean=self.mean, std=self.std, inplace=self.inplace
        )

    @classmethod
    def from_metadata(cls, data: dict[str, str]) -> Self:
        return cls.from_json(data["normalization"])

    @classmethod
    def from_json(cls, data: str, **kwargs: Any) -> Self:
        data = json.loads(data)
        if not isinstance(data, dict):
            msg = f"Expected normalization JSON to be a dict, got: {data}"
            raise TypeError(msg)

        def _check_stats(s: Any) -> None:
            if not isinstance(s, list):
                msg = f"Expected normalization stats to be a list, got: {s}"
                raise TypeError(msg)
            if not all(isinstance(v, float) for v in s):
                msg = f"Expected floats for normalization stats, got: {s}"
                raise TypeError(msg)

        img_mean = data["mean"]
        _check_stats(img_mean)
        img_std = data["std"]
        _check_stats(img_std)

        return cls(img_mean, img_std, **kwargs)


class Denormalize(Normalize):
    r"""
    The inverse of :class:`Normalize`.
    """

    def transform(self, inpt: Tensor, params: dict[str, Any]) -> Tensor:
        inv_mean = [-m for m in self.mean]
        inv_std = [1.0 / s for s in self.std]
        inv_zero = [0] * len(inv_std)

        outp = self._call_kernel(
            T.normalize, inpt, mean=inv_mean, std=inv_zero, inplace=self.inplace
        )
        outp = self._call_kernel(
            T.normalize, inpt, mean=inv_zero, std=inv_std, inplace=True
        )

        assert outp.shape == inpt.shape

        return outp


def _check_types(
    obj: Any, types_or_checks: tuple[type | Callable[[Any], bool], ...]
) -> bool:
    for type_or_check in types_or_checks:
        if (
            isinstance(obj, type_or_check)
            if isinstance(type_or_check, type)
            else type_or_check(obj)
        ):
            return True
    return False


def _check_types_any(
    flat_inputs: list[Any], *types_or_checks: type | Callable[[Any], bool]
) -> bool:
    return any(_check_types(inpt, types_or_checks) for inpt in flat_inputs)


def _check_types_all(
    flat_inputs: list[Any], *types_or_checks: type | Callable[[Any], bool]
) -> bool:
    for type_or_check in types_or_checks:
        for inpt in flat_inputs:
            if (
                isinstance(inpt, type_or_check)
                if isinstance(type_or_check, type)
                else type_or_check(inpt)
            ):
                break
        else:
            return False
    return True
