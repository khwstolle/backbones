import json
import pathlib
import pydoc
from collections.abc import Callable
from itertools import chain
from typing import Any, NamedTuple, TypeGuard, cast

import safetensors
import safetensors.torch
import torch
import torch.nn
import torch.types

from backbones._extract import extract_features

__all__ = [
    "load_meta",
    "check_meta",
    "load_weights",
    "save_weights",
    "load_data",
    "check_data",
    "load_model",
]

type PathLike = pathlib.Path | str
type DeviceLike = str | int
type WeightMeta = dict[str, str]
type WeightData = dict[str, torch.Tensor]


def load_data(path: PathLike, *, device: DeviceLike) -> WeightData:
    r"""
    Read a state dict and respective metadata from a weights file.

    The device must be explicitly passed to prevent needless device copies in user
    code.
    """
    path = _parse_path(path)
    data = cast(object, safetensors.torch.load_file(path, device=device))  # type: ignore[arg-type]
    if not check_data(data):
        if isinstance(data, dict):
            key_types = str({type(k) for k in data})  # type: ignore[arg-type]
            val_types = str({type(v) for v in data.values()})  # type: ignore[arg-type]
        else:
            key_types = val_types = "unknown"
        msg = (
            f"Expected states to be a mapping {{str}} -> {{Tensor}}, got {key_types} "
            f"-> {val_types}"
        )
        raise TypeError(msg)
    return data


def check_data(state: object) -> TypeGuard[WeightData]:
    if not isinstance(state, dict):
        msg = f"Expected state to be a dict, got {type(state)}"
        raise TypeError(msg)
    state = cast(dict[Any, Any], state)
    return all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    )


def load_meta(path: PathLike, *, missing_ok: bool = True) -> WeightMeta:
    path = _parse_path(path)
    with _open(path, device="cpu") as file:  # type: ignore[arg-type]
        data = cast(object, file.metadata())  # type: ignore[arg-type]
    if data is None and missing_ok:
        return {}
    if not check_meta(data):
        if isinstance(data, dict):
            key_types = str({type(k) for k in data})  # type: ignore[arg-type]
            val_types = str({type(v) for v in data.values()})  # type: ignore[arg-type]
        else:
            key_types = val_types = "unknown"
        msg = (
            f"Expected metadata to be a mapping str -> Tensor, got {key_types} -> "
            f"{val_types}"
        )
        raise TypeError(msg)
    return data


def check_meta(meta: object) -> TypeGuard[WeightMeta]:
    if not isinstance(meta, dict):
        msg = f"Expected metadata to be a dict, got {type(meta)}"
        raise TypeError(msg)
    meta = cast(dict[Any, Any], meta)
    return all(isinstance(v, str) for v in chain(meta.values(), meta.keys()))


class Weights(NamedTuple):
    data: WeightData
    meta: WeightMeta


def load_weights(
    path: PathLike, *, device: DeviceLike
) -> tuple[WeightData, WeightMeta]:
    r"""
    Load weights.
    """
    path = _parse_path(path)
    return Weights(load_data(path, device=device), load_meta(path))


def save_weights(wt: Weights | tuple[WeightData, WeightMeta], path: PathLike):
    r"""
    Save weights.
    """
    path = _parse_path(path)
    data, meta = wt
    safetensors.torch.save_file(data, path, meta)


def load_model(path: PathLike, *, device: DeviceLike) -> torch.fx.GraphModule:
    path = _parse_path(path)
    data, meta = load_weights(path, device=device)

    # Find the config - i.e. callable to initialize the model
    config = _locate_config(meta["config"])
    model = config().to(device)

    # Use FX-tracing to select the relevant features from this model
    features = json.loads(meta["features"])
    assert isinstance(features, list | dict), type(features)
    model = extract_features(model, features)

    # Load the state dict
    keys_miss, keys_ndef = model.load_state_dict(data, strict=False)
    if (num_miss := len(keys_miss)) + (num_ndef := len(keys_ndef)) > 0:
        err_miss, err_ndef = (
            ("\n - " + "\n - ".join(keys)) if len(keys) > 0 else "(none)"
            for keys in (keys_miss, keys_ndef)
        )
        msg = (
            f"Found missing ({num_miss}) or undefined ({num_ndef}) weights in "
            f"weights file: {path}\n\n "
            f"Missing: {err_miss}\n\nUndefined: {err_ndef}"
        )
        raise RuntimeError(msg)
    return model


def _locate_config(path: pathlib.Path) -> Callable[[], torch.nn.Module]:
    fn = pydoc.locate(path)
    if not callable(fn):
        msg = f"Could not locate configuration function. Got: {fn}"
        raise ValueError(msg)
    return fn


def _open(path: pathlib.Path, *, device: DeviceLike = "cpu") -> safetensors.safe_open:
    path = pathlib.Path(path)
    return safetensors.safe_open(str(path), "torch", device=device)


def _parse_path(path: PathLike) -> pathlib.Path:
    return pathlib.Path(path).resolve()
