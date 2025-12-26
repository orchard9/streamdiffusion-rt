# StreamDiffusion-RT

Real-time optimized fork of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) for GPU streaming pipelines.

## Why Fork?

StreamDiffusion is excellent real-time diffusion framework. This fork exists because:

1. **Minimal footprint**: Stripped demos, examples, CLI tools (~60% size reduction)
2. **Modern packaging**: pyproject.toml instead of setup.py
3. **Dependency updates**: protobuf 4.25.0+ (compatibility with other GPU libraries)
4. **Integration focus**: Optimized for embedding in larger GPU pipelines

This is NOT a competitor to StreamDiffusion - it's a streamlined derivative for specific use cases.

## What's Different?

| Aspect | Upstream | This Fork |
|--------|----------|-----------|
| LOC | ~5,116 | ~2,059 |
| Package name | `streamdiffusion` | `streamdiffusion_rt` |
| protobuf | 3.20.2 | 4.25.0+ |
| Demos | Included | Removed |
| CLI tools | Included (fire, colored) | Removed |
| Packaging | setup.py | pyproject.toml |

## Installation

```bash
# Basic install
pip install streamdiffusion-rt

# With TensorRT support (recommended for production)
pip install "streamdiffusion-rt[tensorrt]"
```

## Usage

```python
from streamdiffusion_rt import StreamDiffusionWrapper

# Initialize with SDXL-Turbo
stream = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sdxl-turbo",
    t_index_list=[0, 16, 32, 45],
    use_lcm_lora=True,
)

# TensorRT acceleration (4-5x faster)
from streamdiffusion_rt.acceleration.tensorrt import accelerate_with_tensorrt
stream = accelerate_with_tensorrt(stream, engine_dir="/path/to/cache")

# Prepare prompt
stream.prepare(
    prompt="professional portrait, studio lighting",
    negative_prompt="blurry, low quality",
    guidance_scale=1.0,
)

# Real-time inference (img2img)
for frame in video_stream:
    output = stream.img2img(frame)
    yield output
```

## TensorRT Performance

On RTX 4090 with SDXL-Turbo (4-step):

| Resolution | TensorRT FP16 | PyTorch FP16 |
|------------|---------------|--------------|
| 512x512 | ~6ms | ~20ms |
| 768x768 | ~11ms | ~35ms |
| 1024x1024 | ~22ms | ~60ms |

## License

Apache 2.0 (same as upstream)

## Attribution

This project is a derivative work of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
by Akio Kodaira et al. We gratefully acknowledge their pioneering work on real-time diffusion.

Original paper: [StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation](https://arxiv.org/abs/2312.12491)
