from pathlib import Path

import pytest
from backbones._io import load_meta, load_model
from torch import nn

WEIGHTS_ROOT = Path(__file__).parent.parent / "weights"
WEIGHT_FILES = list(WEIGHTS_ROOT.glob("**/*.safetensors"))


@pytest.mark.parametrize(
    "path", WEIGHT_FILES, ids=lambda x: str(x.relative_to(WEIGHTS_ROOT))
)
def test_load_meta(path: str):
    meta = load_meta(path)
    assert isinstance(meta, dict), type(meta)


@pytest.mark.parametrize(
    "path", WEIGHT_FILES, ids=lambda x: str(x.relative_to(WEIGHTS_ROOT))
)
def test_load_model(path: str):
    model = load_model(path)
    assert isinstance(model, nn.Module), type(model)
    assert len(list(model.parameters())) > 0, "Model has no parameters."
