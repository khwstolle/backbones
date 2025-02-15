import pathlib

import pytest
from backbones import load_model

WEIGHTS_ROOT = pathlib.Path(__file__).parent.parent / "weights"
WEIGHT_PATHS = list(WEIGHTS_ROOT.glob("**/*.safetensors"))
WEIGHT_IDS = [wt.relative_to(WEIGHTS_ROOT).as_posix() for wt in WEIGHT_PATHS]


@pytest.mark.parametrize("path", WEIGHT_PATHS, ids=WEIGHT_IDS)
def test_weights_load_model(path: pathlib.Path):
    model = load_model(path, device="cpu")
    assert model is not None
