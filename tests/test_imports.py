"""Basic import tests for streamdiffusion-rt."""

import pytest


def test_core_imports():
    """Test that core classes can be imported."""
    from streamdiffusion_rt import StreamDiffusion
    assert StreamDiffusion is not None


def test_wrapper_import():
    """Test that wrapper can be imported."""
    from streamdiffusion_rt import StreamDiffusionWrapper
    assert StreamDiffusionWrapper is not None


def test_image_utils_import():
    """Test that image utilities can be imported."""
    from streamdiffusion_rt import postprocess_image
    assert postprocess_image is not None


def test_image_filter_import():
    """Test that image filter can be imported."""
    from streamdiffusion_rt import SimilarImageFilter
    assert SimilarImageFilter is not None


def test_tensorrt_imports():
    """Test that TensorRT module can be imported (if available)."""
    try:
        from streamdiffusion_rt.acceleration.tensorrt import (
            accelerate_with_tensorrt,
            compile_unet,
            compile_vae_decoder,
            compile_vae_encoder,
        )
        assert accelerate_with_tensorrt is not None
    except ImportError:
        pytest.skip("TensorRT not installed")


def test_version():
    """Test that version is set correctly."""
    import streamdiffusion_rt
    assert streamdiffusion_rt.__version__ == "0.1.0"
