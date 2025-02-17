from pathlib import Path

import pytest
from backbones import load_weights, resnet
from unipercept.config.lazy import instantiate

WEIGHTS_ROOT = Path(__file__).parent.parent / "weights"


def test_resnet_module():
    pass


@pytest.mark.parametrize(
    ("path", "config"),
    [
        (path, resnet.configs.RESNET_50)
        for path in WEIGHTS_ROOT.glob("resnet/50/*.safetensors")
    ],
)
def test_resnet_weights(path, config):
    model = instantiate(config)
    assert isinstance(model, resnet.ResNet), type(model)
    state = load_weights(path, model, device="cpu")
    assert state is None, "Expected model to be loaded in-place"
