r"""
CLI
===
A command line interface for the backbones library.
"""

import argparse
import inspect
import json
import logging
import pathlib
import sys
import warnings
from collections.abc import Callable
from functools import partial
from pprint import pformat
from typing import Any

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _get_arg_name(param) -> str:
        return param.name.replace("_", "-")

    @staticmethod
    def _get_arg_type(param) -> Any:
        arg_type = param.annotation if param.annotation != param.empty else str
        if not callable(arg_type):
            warnings.warn(f"Invalid type annotation: {arg_type}", stacklevel=2)
            arg_type = str
        if not isinstance(arg_type, type):
            warnings.warn(f"Invalid type annotation: {arg_type}", stacklevel=2)
            arg_type = str
        return arg_type

    @staticmethod
    def _get_arg_default(param) -> Any:
        return param.default if param.default is not param.empty else None

    @classmethod
    def _add_parser(
        cls, parser: argparse.ArgumentParser, func: Callable[..., Any]
    ) -> None:
        """Add arguments to parser based on function signature."""
        sig = inspect.signature(func)
        args_pos = []
        args_flag = []
        for param in sig.parameters.values():
            arg_required = param.default is param.empty
            arg_type = cls._get_arg_type(param)
            match param.kind:
                case param.POSITIONAL_ONLY:
                    args_pos.append(
                        partial(
                            parser.add_argument,
                            param.name,
                            metavar=param.name.upper(),
                            default=param.default
                            if param.default is not param.empty
                            else None,
                            help=f"{param.name} {param.annotation.__name__.lower()}",
                        )
                    )
                case param.VAR_POSITIONAL:
                    args_pos.append(
                        partial(
                            parser.add_argument,
                            "--" + cls._get_arg_name(param),
                            dest=param.name,
                            metavar=arg_type.__name__.upper(),
                            type=arg_type,
                            nargs="*",
                            help=f"{param.annotation.__name__} (varargs)",
                        )
                    )
                case param.KEYWORD_ONLY:
                    if isinstance(param.annotation, bool):
                        args_flag.append(
                            partial(
                                parser.add_argument,
                                f"--{cls._get_arg_name(param)}",
                                dest=param.name,
                                action="store_false"
                                if param.default is True
                                else "store_false",
                                default=param.default
                                if param.default is not param.empty
                                else None,
                            )
                        )
                    else:
                        args_flag.append(
                            partial(
                                parser.add_argument,
                                f"--{cls._get_arg_name(param)}",
                                type=arg_type,
                                dest=param.name,
                                metavar=param.name.upper(),
                                required=arg_required,
                                default=param.default
                                if param.default is not param.empty
                                else None,
                                help=f"{param.annotation.__name__}"
                                if param.annotation != param.empty
                                else "",
                            )
                        )
                case param.VAR_KEYWORD:
                    args_pos.append(
                        partial(
                            parser.add_argument,
                            param.name,
                            metavar="KEY=VALUE",
                            nargs=argparse.REMAINDER,
                            help=param.name
                            + " "
                            + arg_type.__name__.lower()
                            + " keywords",
                        )
                    )
                case unsupported_kind:
                    msg = f"Cannot add {param.name} ({unsupported_kind}) to parser."
                    raise NotImplementedError(msg)

        # First add flags, then positionals
        for arg in args_flag:
            arg()
        for arg in args_pos:
            arg()

    @classmethod
    def _bind_arguments(
        cls, args: argparse.Namespace
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Extract relevant arguments from namespace."""

        args_pos = []
        args_key = {}
        for param in inspect.signature(args._command).parameters.values():
            arg_type = cls._get_arg_type(param)
            arg_value = getattr(args, param.name)
            match param.kind:
                case param.POSITIONAL_ONLY | param.POSITIONAL_OR_KEYWORD:
                    args_pos.append(arg_type(arg_value))
                case param.KEYWORD_ONLY:
                    args_key[param.name] = arg_type(arg_value)
                case param.VAR_POSITIONAL:
                    if arg_value is None or len(arg_value) == 0:
                        continue
                    args_pos.extend(arg_type(v) for v in arg_value)
                case param.VAR_KEYWORD:
                    if arg_value is None or len(arg_value) == 0:
                        continue

                    for var_key, var_value in (vs.split("=") for vs in arg_value):
                        args_key[var_key] = arg_type(var_value)
                case unknown_kind:
                    msg = f"Unknown parameter kind {unknown_kind}"
                    raise RuntimeError(msg)
        return tuple(args_pos), args_key

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
    from ._io import load_meta, save_meta

    meta = load_meta(path)
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
    save_meta(path, meta)


@cli
def keys(path: pathlib.Path, /) -> None:
    """Read metadata of a weights file."""
    from ._io import load_weights

    data = load_weights(path, device="cpu")  # cannot write meta only...
    for key in sorted(data.keys()):
        print(key)


@cli
def extract() -> None:
    r"""
    Extract features from a pre-trained network.
    """

    msg = "Extracting features from a pre-trained network is not yet implemented."
    raise NotImplementedError(msg)


@cli
def export(
    path: pathlib.Path,
    /,
    *features_list: str,
    device: str = "cpu",
    unsafe: bool = False,
    **features_map: str,
) -> None:
    r"""
    Export a pre-trained network using `torch.export`.
    """

    import torch.export
    import torch.nn

    from ._export import export
    from ._features import extract_features
    from ._io import load_model

    # Sanitization: features to export
    if len(features_list) == 0 and len(features_map) == 0:
        msg = "No features to export."
        raise ValueError(msg)

    if set(features_list) & set(features_map.keys()):
        msg = "Cannot export the same feature multiple times."
        raise ValueError(msg)

    # Load model
    model = load_model(path, device=device, unsafe=unsafe)

    # Extract features
    features = {**dict.fromkeys(features_list, features_list), **features_map}
    logger.info("Extracting features: %s", pformat(features, indent=4))

    model = extract_features(model, features)

    # Export model
    logger.info("Exporting model...")
    model_export = export(model)

    output = path.with_suffix(".pt2")
    torch.export.save(model_export, output)

    sys.stdout.write(str(output.resolve()) + "\n")


@cli
def available() -> None:
    r"""
    List all available backbones.
    """

    msg = "Listing all available backbones is not yet implemented."
    raise NotImplementedError(msg)
