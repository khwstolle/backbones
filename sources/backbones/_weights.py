import pathlib
from itertools import chain
from typing import Any, NamedTuple, TypeGuard, cast

import safetensors
import safetensors.torch
import torch
import torch.types

__all__ = ["load_metadata", "check_metadata", "load_statedict", "check_statedict"]

type PathLike = pathlib.Path | str
type DeviceLike = str | int
type Metadata = dict[str, str]
type StateDict = dict[str, torch.Tensor]


class Weights(NamedTuple):
    state: StateDict
    meta: Metadata


def load_statedict(path: PathLike, *, device: DeviceLike) -> StateDict:
    r"""
    Read a state dict from a weights file.

    The device must be explicitly passed to prevent needless device copies in user
    code.
    """
    data = cast(object, safetensors.torch.load_file(path, device=device))  # type: ignore[arg-type]
    if not check_statedict(data):
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


def check_statedict(state: object) -> TypeGuard[StateDict]:
    if not isinstance(state, dict):
        msg = f"Expected state to be a dict, got {type(state)}"
        raise TypeError(msg)
    state = cast(dict[Any, Any], state)
    return all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    )


def load_metadata(path: PathLike, *, missing_ok: bool = True) -> Metadata:
    path = pathlib.Path(path)
    with _open(path, device="cpu") as file:  # type: ignore[arg-type]
        data = cast(object, file.metadata())  # type: ignore[arg-type]
    if data is None and missing_ok:
        return {}
    if not check_metadata(data):
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


def check_metadata(meta: object) -> TypeGuard[Metadata]:
    if not isinstance(meta, dict):
        msg = f"Expected metadata to be a dict, got {type(meta)}"
        raise TypeError(msg)
    meta = cast(dict[Any, Any], meta)
    return all(isinstance(v, str) for v in chain(meta.values(), meta.keys()))


def _open(path: PathLike, *, device: DeviceLike = "cpu") -> safetensors.safe_open:
    path = pathlib.Path(path)
    return safetensors.safe_open(str(path), "torch", device=device)
