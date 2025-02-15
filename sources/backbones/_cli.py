r"""
CLI
===
A command line interface for the backbones library.
"""

import argparse
import inspect
import json
import pathlib
from collections.abc import Callable
from typing import Any, cast


class cli:
    """Decorator for CLI commands with automatic argument binding."""

    registry: dict[str, Callable[..., Any]] = {}

    def __new__(cls, fn: Callable[..., Any]) -> Callable[..., Any]:
        cls.registry[fn.__name__.replace("_", "-")] = fn
        return fn

    @classmethod
    def main(cls) -> None:
        """Configure argparse and execute registered commands."""
        parser = argparse.ArgumentParser(description="Backbones CLI")
        subparsers = parser.add_subparsers(title="commands", required=True)

        for name, func in cls.registry.items():
            cmd_parser = subparsers.add_parser(name, help=func.__doc__)
            cls._add_parser(cmd_parser, func)
            cmd_parser.set_defaults(_command=func)

        args = parser.parse_args()
        if hasattr(args, "_command"):
            cmd_args, cmd_kwargs = cls._bind_arguments(args)
            args._command(*cmd_args, **cmd_kwargs)

    @classmethod
    def _get_arg_name(cls, param) -> str:
        return param.name.replace("_", "-")

    @classmethod
    def _add_parser(
        cls, parser: argparse.ArgumentParser, func: Callable[..., Any]
    ) -> None:
        """Add arguments to parser based on function signature."""
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            arg_required = param.default is param.empty
            match param.kind:
                case param.POSITIONAL_ONLY:
                    parser.add_argument(
                        cls._get_arg_name(param),
                        default=param.default
                        if param.default is not param.empty
                        else None,
                        help=f"{param.annotation.__name__}",
                    )
                case param.KEYWORD_ONLY:
                    parser.add_argument(
                        f"--{cls._get_arg_name(param)}",
                        type=param.annotation
                        if param.annotation != param.empty
                        else str,
                        required=param.default is param.empty,
                        default=param.default
                        if param.default is not param.empty
                        else None,
                        help=f"{param.annotation.__name__}"
                        if param.annotation != param.empty
                        else "",
                    )
                case param.VAR_KEYWORD:
                    parser.add_argument(
                        cls._get_arg_name(param),
                        nargs=argparse.REMAINDER,
                        help="key=value",
                    )
                case unsupported_kind:
                    msg = f"Cannot add {param.name} ({unsupported_kind}) to argument parser: {param}"
                    raise NotImplementedError(msg)

    @classmethod
    def _bind_arguments(
        cls, args: argparse.Namespace
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Extract relevant arguments from namespace."""

        args_pos = []
        args_key = {}
        for param in inspect.signature(args._command).parameters.values():
            key = cls._get_arg_name(param)
            value = getattr(args, key)
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                args_pos.append(value)
                continue
            if param.kind in (param.KEYWORD_ONLY,):
                assert key not in args_key, args_key.keys()
                args_key[key] = value
                continue
            if param.kind in (param.VAR_KEYWORD,):
                assert key not in args_key, args_key.keys()

                type_as = (
                    param.annotation
                    if callable(param.annotation) and param.annotation != param.empty
                    else lambda x: x
                )

                args_key[key] = {
                    var_key: type_as(var_value)
                    for var_key, var_value in value.split("=")
                }
                continue

            msg = f"Unknown parameter kind {param.kind}"
            raise RuntimeError(msg)

        return args_pos, args_key


@cli
def version() -> None:
    """Print the version of the backbones library."""
    from . import __version__

    print(f"backbones v{__version__}")


@cli
def meta(path: pathlib.Path, /) -> None:
    """Read metadata of a weights file."""
    import safetensors.torch as st

    with st.safe_open(path, "torch") as data:  # type: ignore[arg-type]
        meta = cast(dict[str, str], data.metadata())  # type: ignore[attr-defined]
        print(json.dumps(meta, indent=4))


@cli
def extract() -> None:
    r"""
    Extract features from a pre-trained network.
    """

    msg = "Extracting features from a pre-trained network is not yet implemented."
    raise NotImplementedError(msg)


@cli
def export() -> None:
    r"""
    Export a pre-trained network using `torch.export`.
    """

    msg = "Exporting a pre-trained network is not yet implemented."
    raise NotImplementedError(msg)


@cli
def list() -> None:
    r"""
    List all available backbones.
    """

    msg = "Listing all available backbones is not yet implemented."
    raise NotImplementedError(msg)
