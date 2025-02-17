r"""
Uses symbolic tracing to remove redundant (e.g. classification) weights from a backbone
network.

Notes
-----
Deprecated our own FX-tracing in favor of the ``torchvision`` implementation, which
has adopted the same functionality.
"""

import copy
import enum
import typing

import torch
import torch.fx
import torch.nn

__all__ = ["FeatureInfo", "FeatureFormat", "FeatureDict", "extract_features", "probe_features"]


class FeatureInfo(typing.TypedDict):
    """
    Information about a feature.

    Properties
    ----------
    channels : int
        The number of channels of the feature.
    stride : int
        The stride of the feature (with respect to the input image).
    """

    channels: int
    stride: int

type FeatureDict = dict[str, FeatureInfo]

class FeatureFormat(enum.StrEnum):
    """The format of the extracted features."""

    CHW = "CHW"
    HWC = "HWC"

FeatureFormatType = typing.Literal["CHW", "HWC"] | FeatureFormat

type FeatureModule = typing.Callable[[torch.Tensor], typing.Mapping[str, torch.Tensor]] | torch.nn.Module | torch.fx.GraphModule

def probe_features(
    bb: FeatureModule,
    order: FeatureFormat,
    shape: tuple[int, int] = (512, 256),
) -> FeatureDict:
    r"""
    Infer the feature information from the model by priming it with random data.

    Parameters
    ----------
    bb
        The model that returns a dict of features.
    order
        The dimension order of the extracted features.
    shape
        The shape of the test data, which is used to infer the stride of the features.

    Returns
    -------
    FeatureDict
        A dictionary mapping the feature names to their respective feature information.
    """

    bb = copy.deepcopy(bb)
    if isinstance(bb, torch.nn.Module | torch.fx.GraphModule):
        bb.eval() # type: ignore[union-attr]

    def _probe_output_shapes(
        mod: FeatureModule,
    ) -> dict[str, torch.Size]:

        with torch.no_grad():
            inp = torch.randn((8, 3, *shape), dtype=torch.float32)
            out = mod(inp)
        if out is None or not out:
            msg = (
                f"Failed to infer shapes from {type(mod)} with {shape=}. "
                f"Invalid result {out=!r}"
            )
            raise ValueError(msg)
        if not all(isinstance(v, torch.Tensor) for v in out.values()):
            msg = (
                f"Failed to infer shapes from {type(mod)} with {shape=}. "
                f"Not all results are Tensors: {out}"
            )
            raise ValueError(msg)
        return {k: v.shape for k, v in out.items()}

    def _shape_to_info(
        shape: torch.Size,
    ) -> FeatureInfo:
        match order:
            case FeatureFormat.CHW:
                c, h, w = map(int, shape[1:])
            case FeatureFormat.HWC:
                h, w, c = map(int, shape[1:])
            case _:
                msg = f"Cannot infer feature info for {order}"
                raise NotImplementedError(msg)

        stride_h = shape[0] // h
        stride_w = shape[1] // w

        # TODO: This is not always true, e.g. for Swin Transformers with patch size 4 and stride 2, add support.
        assert stride_h == stride_w, (
            f"Stride must be equal in both dimensions, got {stride_h} and {stride_w}. "
            f"Size of input was {shape}, size of output was {shape}, order is {order}."
        )

        stride = stride_h

        assert c > 0, c
        assert stride > 0, shape

        return FeatureInfo(channels=c, stride=stride)
    return {k: _shape_to_info(v) for k, v in _probe_output_shapes(bb).items()}


def extract_features(
    model: torch.nn.Module, features: dict[str, str] | list[str]
) -> torch.fx.GraphModule:
    from torchvision.models.feature_extraction import create_feature_extractor

    gm = create_feature_extractor(model, features)
    return gm
