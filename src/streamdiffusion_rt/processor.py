"""M40-compatible wrapper for StreamDiffusion real-time inference.

This module provides StreamDiffusionProcessor, which implements the M40 FrameProcessor
protocol for integration with Masquerade's faceswap-worker pipeline.

Key features:
- Implements FrameProcessor protocol (initialize, configure, process_frame, etc.)
- Supports TensorRT acceleration for low-latency inference
- Maintains temporal consistency via x_t_latent buffer
- Declares capabilities for resource budgeting

Example:
    from streamdiffusion_rt import StreamDiffusionProcessor

    processor = StreamDiffusionProcessor(model_id="stabilityai/sdxl-turbo")
    processor.initialize()
    processor.configure(prompt="professional portrait, studio lighting")

    for frame in video_stream:
        output = processor.process_frame(frame, context)
        yield output

    processor.cleanup()
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from streamdiffusion_rt.pipeline import StreamDiffusion
from streamdiffusion_rt.wrapper import StreamDiffusionWrapper


@dataclass
class ProcessorCapabilities:
    """Declares what a processor can do and needs.

    This is a local copy for standalone use. When used with masquerade.processors,
    the import from there will be used instead.
    """

    transforms_face: bool = False
    transforms_background: bool = False
    transforms_full_frame: bool = False
    requires_face_detection: bool = False
    requires_landmarks: bool = False
    requires_embedding: bool = False
    requires_text_prompt: bool = False
    preferred_resolution: Optional[tuple] = None
    maintains_input_resolution: bool = True
    tensor_format: Literal["cupy", "torch"] = "cupy"
    is_stateful: bool = False
    requires_frame_history: int = 0
    supports_batch: bool = False
    typical_latency_ms: float = 0.0
    vram_usage_mb: float = 0.0
    exclusive_gpu: bool = False


@dataclass
class ProcessingContext:
    """Shared context passed through processor chain.

    This is a minimal local copy for standalone use. When used with masquerade.processors,
    the import from there will be used instead.
    """

    original_frame: Optional[Any] = None
    detected_faces: Optional[List[Any]] = None
    face_landmarks: Optional[Dict[int, Any]] = None
    source_embedding: Optional[Any] = None
    text_embeddings: Optional[Any] = None
    negative_embeddings: Optional[Any] = None
    guidance_scale: float = 7.5
    occlusion_mask: Optional[Any] = None
    face_mask: Optional[Any] = None
    stream_key: Optional[str] = None
    frame_index: int = 0
    processor_state: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class WarmupConfig:
    """Configuration for processor warmup."""

    prompts: Optional[List[str]] = None
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    timeout_seconds: float = 60.0


@dataclass
class WarmupResult:
    """Result of processor warmup operation."""

    success: bool
    duration_seconds: float
    stages_completed: List[str] = field(default_factory=list)
    error: Optional[str] = None


class StreamDiffusionProcessor:
    """M40-compatible wrapper for StreamDiffusion real-time inference.

    Implements the FrameProcessor protocol for integration with Masquerade's
    faceswap-worker pipeline. Uses StreamDiffusion for real-time diffusion
    with temporal consistency.

    Args:
        model_id: HuggingFace model ID (default: stabilityai/sdxl-turbo).
        t_index_list: Timestep indices for denoising (default: [0, 16, 32, 45]).
        use_tensorrt: Whether to use TensorRT acceleration (default: True).
        engine_cache_dir: Directory for cached TensorRT engines.
        width: Output width (default: 768).
        height: Output height (default: 768).

    Example:
        >>> processor = StreamDiffusionProcessor()
        >>> processor.initialize()
        >>> processor.configure(prompt="portrait, studio lighting")
        >>> result = processor.process_frame(frame, context)
    """

    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        t_index_list: Optional[List[int]] = None,
        use_tensorrt: bool = True,
        engine_cache_dir: str = "/opt/masq/models/tensorrt_cache/diffusion",
        width: int = 768,
        height: int = 768,
    ):
        self._model_id = model_id
        self._t_index_list = t_index_list or [0, 16, 32, 45]
        self._use_tensorrt = use_tensorrt
        self._engine_cache_dir = engine_cache_dir
        self._width = width
        self._height = height

        self._stream: Optional[StreamDiffusionWrapper] = None
        self._prompt: Optional[str] = None
        self._negative_prompt: Optional[str] = None
        self._is_ready = False

        self._capabilities = ProcessorCapabilities(
            transforms_full_frame=True,
            requires_text_prompt=True,
            is_stateful=True,
            tensor_format="torch",
            preferred_resolution=(width, height),
            maintains_input_resolution=False,
            typical_latency_ms=50.0,
            vram_usage_mb=8000.0,
            exclusive_gpu=True,
        )

    @property
    def name(self) -> str:
        """Unique processor identifier."""
        return "diffusion_streamdiffusion"

    @property
    def capabilities(self) -> ProcessorCapabilities:
        """Processor capabilities declaration."""
        return self._capabilities

    @property
    def is_ready(self) -> bool:
        """Check if processor is initialized and ready for inference."""
        return self._is_ready

    @property
    def tensor_format(self) -> Literal["cupy", "torch"]:
        """Preferred tensor format for this processor."""
        return "torch"

    def initialize(self) -> None:
        """Initialize processor resources.

        Loads SDXL-Turbo model and optionally builds TensorRT engines.
        This can take several minutes for first-time TensorRT compilation.
        """
        if self._is_ready:
            return

        acceleration = "tensorrt" if self._use_tensorrt else "xformers"

        self._stream = StreamDiffusionWrapper(
            model_id_or_path=self._model_id,
            t_index_list=self._t_index_list,
            use_lcm_lora=True,
            output_type="pt",
            width=self._width,
            height=self._height,
            acceleration=acceleration,
            engine_dir=self._engine_cache_dir,
        )

        self._is_ready = True

    def configure(self, **kwargs: Any) -> None:
        """Configure processor for current stream.

        Args:
            prompt: Text prompt for generation.
            negative_prompt: Negative prompt (default: "blurry, low quality").
            guidance_scale: CFG scale (default: 1.0 for SDXL-Turbo).
        """
        if self._stream is None:
            raise RuntimeError("Must call initialize() before configure()")

        self._prompt = kwargs.get("prompt", "high quality portrait, professional lighting")
        self._negative_prompt = kwargs.get("negative_prompt", "blurry, low quality, deformed")
        guidance_scale = kwargs.get("guidance_scale", 1.0)

        self._stream.prepare(
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            guidance_scale=guidance_scale,
        )

    def process_frame(
        self,
        frame: Any,
        context: ProcessingContext,
    ) -> Any:
        """Process a single frame through diffusion pipeline.

        Accepts both CuPy and PyTorch input tensors. Converts CuPy to PyTorch
        internally (zero-copy via DLPack when possible), processes through
        diffusion, then converts back to input format.

        Args:
            frame: Input frame tensor (BCHW format, float16, 0-1 range).
                   Can be torch.Tensor or cupy.ndarray.
            context: Shared processing context with temporal state.

        Returns:
            Processed frame tensor in same format as input.
        """
        if not self._is_ready:
            raise RuntimeError("Processor not initialized")

        # Track input format for output conversion
        input_is_cupy = hasattr(frame, "__cuda_array_interface__") and not isinstance(frame, torch.Tensor)

        t0 = time.perf_counter()

        # Convert CuPy to PyTorch if needed (zero-copy via DLPack)
        if input_is_cupy:
            t_convert_start = time.perf_counter()
            frame_torch = self._cupy_to_torch(frame)
            context.stage_timings["diffusion_input_convert"] = (
                time.perf_counter() - t_convert_start
            ) * 1000
        else:
            frame_torch = frame

        # Ensure correct format [B, C, H, W] and dtype
        if frame_torch.dim() == 3:
            frame_torch = frame_torch.unsqueeze(0)
        frame_torch = frame_torch.half()

        # Get temporal state from context (unused currently - StreamDiffusion manages internally)
        # x_t_latent = context.processor_state.get("diffusion_x_t_latent")

        # Run img2img diffusion
        output = self._stream.stream.img2img(frame_torch)

        # Store temporal state for next frame
        if hasattr(self._stream.stream, "x_t_latent_buffer"):
            context.processor_state["diffusion_x_t_latent"] = (
                self._stream.stream.x_t_latent_buffer
            )

        t1 = time.perf_counter()
        context.stage_timings["diffusion_inference"] = (t1 - t0) * 1000

        # Convert output back to CuPy if input was CuPy
        if input_is_cupy:
            t_convert_start = time.perf_counter()
            output = self._torch_to_cupy(output)
            context.stage_timings["diffusion_output_convert"] = (
                time.perf_counter() - t_convert_start
            ) * 1000

        return output

    def _cupy_to_torch(self, arr: Any) -> torch.Tensor:
        """Convert CuPy array to PyTorch tensor (zero-copy via DLPack).

        Args:
            arr: CuPy array on GPU.

        Returns:
            PyTorch tensor sharing memory (zero-copy when contiguous).
        """
        # Use DLPack for zero-copy conversion
        try:
            return torch.from_dlpack(arr.toDlpack())
        except Exception:
            # Fallback: make contiguous first
            import cupy as cp
            arr_contig = cp.ascontiguousarray(arr)
            return torch.from_dlpack(arr_contig.toDlpack())

    def _torch_to_cupy(self, tensor: torch.Tensor) -> Any:
        """Convert PyTorch tensor to CuPy array (zero-copy via DLPack).

        Args:
            tensor: PyTorch tensor on CUDA.

        Returns:
            CuPy array sharing memory (zero-copy when contiguous).
        """
        import cupy as cp

        # Make contiguous if needed
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return cp.from_dlpack(tensor)

    def warmup(self, config: Optional[WarmupConfig] = None) -> WarmupResult:
        """Warm up processor for low-latency inference.

        Runs dummy inference passes to prime GPU kernels and TensorRT engines.
        Critical for stable low-latency performance.

        Args:
            config: Optional warmup configuration.

        Returns:
            WarmupResult with success status and timing.
        """
        if not self._is_ready:
            self.initialize()

        config = config or WarmupConfig()
        stages = []
        start = time.time()

        try:
            # Configure with default or provided prompts
            prompts = config.prompts or ["high quality portrait, professional lighting"]
            self.configure(prompt=prompts[0])
            stages.append("configure")

            # Run warmup frames
            dummy = torch.zeros(
                1, 3, self._height, self._width,
                device="cuda",
                dtype=torch.float16,
            )
            context = ProcessingContext()

            for i in range(5):
                _ = self.process_frame(dummy, context)
            stages.append("warmup_inference")

            # Sync and measure
            torch.cuda.synchronize()
            stages.append("cuda_sync")

            duration = time.time() - start
            return WarmupResult(
                success=True,
                duration_seconds=duration,
                stages_completed=stages,
            )

        except Exception as e:
            return WarmupResult(
                success=False,
                duration_seconds=time.time() - start,
                stages_completed=stages,
                error=str(e),
            )

    def cleanup(self) -> None:
        """Release processor resources."""
        if self._stream is not None:
            del self._stream
            self._stream = None
        self._is_ready = False
        torch.cuda.empty_cache()

    def reset_state(self) -> None:
        """Clear temporal state for new stream.

        Resets the diffusion denoising state (x_t_latent buffer) for a new stream.
        Call this when starting a new video stream.
        """
        if self._stream is not None and hasattr(self._stream, "stream"):
            if hasattr(self._stream.stream, "x_t_latent_buffer"):
                self._stream.stream.x_t_latent_buffer = None


# Export for package-level import
__all__ = [
    "StreamDiffusionProcessor",
    "ProcessorCapabilities",
    "ProcessingContext",
    "WarmupConfig",
    "WarmupResult",
]
