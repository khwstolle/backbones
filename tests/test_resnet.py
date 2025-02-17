
import pytest
import unipercept.config.lazy
from backbones import resnet


@pytest.mark.parametrize("name", resnet.configs.__all__)
def test_resnet_configs(name):
    config = getattr(resnet.configs, name)
    assert config is not None
    model = unipercept.config.lazy.instantiate(config)
    assert isinstance(model, resnet.ResNet), type(model)
