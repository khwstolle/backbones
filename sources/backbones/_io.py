import pathlib
import warnings
from datetime import datetime
from typing import Any, TypeGuard, cast, overload

import safetensors
import safetensors.torch
import torch
import torch.nn
import torch.types

__all__ = [
    "load_weights",
    "save_weights",
    "load_meta",
    "save_meta",
    "StateDict",
]

type PathLike = pathlib.Path | str
type DeviceLike = str | int

type StateDict = dict[str, torch.Tensor]


@overload
def load_weights(
    path: PathLike, model: torch.nn.Module, *, device: DeviceLike
) -> None: ...


@overload
def load_weights(
    path: PathLike, model: None = None, *, device: DeviceLike
) -> StateDict: ...


def load_weights(
    path: PathLike, model: torch.nn.Module | None = None, *, device: DeviceLike
) -> StateDict | None:
    r"""
    Read a state dict and respective metadata from a weights file.

    The device must be explicitly passed to prevent needless device copies in user
    code.
    """
    path = _parse_path(path)

    if model is not None:
        keys_missing, keys_unexpected = safetensors.torch.load_model(
            model, path, strict=False, device=device
        )
        if len(keys_unexpected) > 0:
            msg = f"Unexpected keys: {keys_unexpected}."
            warnings.warn(msg, stacklevel=2)
        if len(keys_missing) > 0:
            msg = f"Missing keys: {keys_missing}."
            raise RuntimeError(msg)
        return None

    data = cast(object, safetensors.torch.load_file(path, device=device))  # type: ignore[arg-type]
    if not check_weights(data):
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


def check_weights(state: object) -> TypeGuard[StateDict]:
    if not isinstance(state, dict):
        msg = f"Expected state to be a dict, got {type(state)}"
        raise TypeError(msg)
    state = cast(dict[Any, Any], state)
    return all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    )


def save_weights(
    data: StateDict, path: PathLike, *, meta: dict[str, str] | None = None
):
    r"""
    Save weights.
    """
    data_path = _parse_path(path)
    data_meta = {"framework": "pt", "timestamp": datetime.now().isoformat()}
    if meta is not None:
        data_meta.update(meta)
    safetensors.torch.save_file(data, data_path, data_meta)


def load_meta(path: PathLike) -> dict[str, str]:
    path = _parse_path(path)
    with safetensors.safe_open(path, framework="pt", device="cpu") as st:
        return st.metadata()


def save_meta(path: PathLike, meta: dict[str, str]) -> None:
    path = _parse_path(path)
    weights = load_weights(path, device="cpu")
    save_weights(weights, path, meta=meta)


def _parse_path(path: PathLike) -> pathlib.Path:
    return pathlib.Path(path).resolve()
