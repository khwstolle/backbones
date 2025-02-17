def test_package_version():
    import backbones

    assert backbones.__version__, "Package version is not defined."
    assert backbones.__version__ != "unknown", "Package version is unknown."
