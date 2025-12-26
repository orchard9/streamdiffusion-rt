"""
StreamDiffusion-RT: Real-time optimized fork of StreamDiffusion.

This is a streamlined derivative of StreamDiffusion focused on GPU streaming pipelines.
See https://github.com/orchard9/streamdiffusion-rt for details.
"""

from streamdiffusion_rt.pipeline import StreamDiffusion
from streamdiffusion_rt.wrapper import StreamDiffusionWrapper
from streamdiffusion_rt.image_utils import postprocess_image
from streamdiffusion_rt.image_filter import SimilarImageFilter

__version__ = "0.1.0"
__all__ = [
    "StreamDiffusion",
    "StreamDiffusionWrapper",
    "postprocess_image",
    "SimilarImageFilter",
]
