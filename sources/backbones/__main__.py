r"""
CLI
===
A command line interface for the backbones library.
"""

import typing
import argparse


class command:
    registry: dict[str, typing.Any] = {}

    def __new__[_FN: typing.Callable[[], None]](cls, fn: _FN) -> _FN:
        cls.registry[fn.__name__.replace("_", "-")] = fn

        return fn

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description="Backbones CLI")
        parser.add_argument(
            "command", type=str, choices=cls.registry.keys(), help="The command to run."
        )
        parser.parse_args()


@command
def version() -> None:
    r"""
    Print the version of the backbones library.
    """
    from . import __version__

    print(f"backbones v{__version__}")


@command
def extract() -> None:
    r"""
    Extract features from a pre-trained network.
    """

    msg = "Extracting features from a pre-trained network is not yet implemented."
    raise NotImplementedError(msg)


@command
def export() -> None:
    r"""
    Export a pre-trained network using `torch.export`.
    """

    msg = "Exporting a pre-trained network is not yet implemented."
    raise NotImplementedError(msg)


@command
def list() -> None:
    r"""
    List all available backbones.
    """

    msg = "Listing all available backbones is not yet implemented."
    raise NotImplementedError(msg)


command.main()
