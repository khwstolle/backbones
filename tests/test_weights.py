import pytest

import pathlib
import gc

from backbones import load_statedict, load_metadata

WEIGHTS_ROOT = pathlib.Path(__file__).parent.parent / "weights"
WEIGHT_PATHS = list(WEIGHTS_ROOT.glob("**/*.safetensors"))
WEIGHT_IDS = [wt.relative_to(WEIGHTS_ROOT).as_posix() for wt in WEIGHT_PATHS]


@pytest.mark.parametrize("path", WEIGHT_PATHS, ids=WEIGHT_IDS)
def test_weights_statedict(path: pathlib.Path):
    data = load_statedict(path, device="cpu")
    assert data is not None

    del data

    gc.collect()


@pytest.mark.parametrize("path", WEIGHT_PATHS, ids=WEIGHT_IDS)
def test_weights_metadata(path: pathlib.Path):
    data = load_metadata(path, missing_ok=False)
    assert data is not None
