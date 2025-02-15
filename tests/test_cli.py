r"""
Tests for ``backbones.__main__``
"""

import backbones._cli as commands


def test_cli_version():
    assert commands.version() == 0


def test_cli_metadata():
    pass


def test_cli_available():
    assert commands.available() == 0
