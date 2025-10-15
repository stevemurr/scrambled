"""Basic import tests to verify package structure."""


def test_import_main_package():
    """Test that main package can be imported."""
    import scrambled_hacks
    assert hasattr(scrambled_hacks, "__version__")
    assert scrambled_hacks.__version__ == "0.1.0"


def test_import_subpackages():
    """Test that all subpackages can be imported."""
    from scrambled_hacks import embeddings
    from scrambled_hacks import segmentation
    from scrambled_hacks import database
    from scrambled_hacks import reconstruction

    # If imports work, test passes
    assert True
