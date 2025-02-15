r"""
CLI
===
A command line interface for the backbones library.
"""

import argparse
import inspect
import json
import pathlib
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

from backbones._io import load_data, load_meta, save_weights


class cli:
    """Decorator for CLI commands with automatic argument binding."""

    registry: dict[str, Callable[..., Any]] = {}

    def __new__(cls, fn: Callable[..., Any]) -> Callable[[], int]:
        command = fn.__name__.replace("_", "-")
        cls.registry[command] = fn
        return partial(cls.main, command)

    @classmethod
    def main(cls, command: str | None = None) -> int:
        """Configure argparse and execute registered commands."""
        parser = argparse.ArgumentParser(description="Backbones CLI")
        subparsers = parser.add_subparsers(title="commands", required=True)

        for name, func in cls.registry.items():
            if command is not None and command != name:
                continue
            cmd_parser = subparsers.add_parser(name, help=func.__doc__)
            cls._add_parser(cmd_parser, func)
            cmd_parser.set_defaults(_command=func)

        if command is not None:
            sys.argv.insert(0, command)

        args = parser.parse_args()
        if hasattr(args, "_command"):
            cmd_args, cmd_kwargs = cls._bind_arguments(args)
            args._command(*cmd_args, **cmd_kwargs)

        return 0

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
                    if isinstance(param.annotation, bool):
                        parser.add_argument(
                            f"--{cls._get_arg_name(param)}",
                            action="store_false"
                            if param.default is True
                            else "store_false",
                            default=param.default
                            if param.default is not param.empty
                            else None,
                        )
                    else:
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
                if len(value) == 0:
                    continue
                type_as = (
                    param.annotation
                    if callable(param.annotation) and param.annotation != param.empty
                    else lambda x: x
                )

                for var_key, var_value in (vs.split("=") for vs in value):
                    args_key[var_key] = type_as(var_value)
                continue

            msg = f"Unknown parameter kind {param.kind}"
            raise RuntimeError(msg)

        return args_pos, args_key

    @staticmethod
    def query_bool(question, *, default: bool | None = None) -> bool:
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        match default:
            case True:
                prompt = " [Y/n] "
            case False:
                prompt = " [y/N] "
            case _:
                prompt = " [y/n] "

        while True:
            sys.stdout.write("\n" + question + prompt)
            choice = input().lower()
            if default is not None and choice == "":
                return default
            if choice in valid:
                return valid[choice]


@cli
def version() -> None:
    """Print the version of the backbones library."""
    from . import __version__

    print(f"backbones v{__version__}")


@cli
def meta(path: pathlib.Path, /, *, yes: bool = False, **overrides) -> None:
    """Read metadata of a weights file."""
    meta = load_meta(path, missing_ok=True)
    for k, v in overrides.items():
        if (isinstance(v, str) and v == "") or v is None:
            del meta[k]
        else:
            meta[k] = v

    json.dump(meta, sys.stdout, indent=4)
    sys.stdout.write("\n")

    if len(overrides) == 0:
        return
    if not yes:
        yes = cli.query_bool("Save modified metadata to weights file?", default=False)
    if not yes:
        return
    data = load_data(path, device="cpu")  # cannot write meta only...
    save_weights((data, meta), path)


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
def available() -> None:
    r"""
    List all available backbones.
    """

    msg = "Listing all available backbones is not yet implemented."
    raise NotImplementedError(msg)
