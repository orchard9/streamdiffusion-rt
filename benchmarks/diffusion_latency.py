#!/usr/bin/env python3
"""Benchmark script for StreamDiffusionProcessor latency.

Measures P50/P95/P99 latency and VRAM usage for real-time diffusion processing.

Usage:
    python benchmarks/diffusion_latency.py --num-frames 100 --resolution 768
"""

import argparse
import time
from typing import Optional

import numpy as np
import torch

from streamdiffusion_rt import StreamDiffusionProcessor, ProcessingContext


def benchmark_diffusion(
    num_frames: int = 100,
    resolution: int = 768,
    use_tensorrt: bool = True,
    warmup_frames: int = 10,
) -> dict:
    """Run latency benchmark for StreamDiffusionProcessor.

    Args:
        num_frames: Number of frames to benchmark.
        resolution: Square resolution (width = height).
        use_tensorrt: Whether to use TensorRT acceleration.
        warmup_frames: Number of warmup frames before timing.

    Returns:
        Dictionary with latency statistics and VRAM usage.
    """
    print(f"\n{'='*60}")
    print(f"StreamDiffusion Latency Benchmark")
    print(f"{'='*60}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"TensorRT: {use_tensorrt}")
    print(f"Frames: {num_frames}")
    print(f"Warmup: {warmup_frames}")

    # Initialize processor
    print("\nInitializing processor...")
    processor = StreamDiffusionProcessor(
        use_tensorrt=use_tensorrt,
        width=resolution,
        height=resolution,
    )
    processor.initialize()
    processor.configure(prompt="professional portrait, studio lighting, high quality")

    # Create dummy frame
    dummy = torch.zeros(
        1, 3, resolution, resolution,
        device="cuda",
        dtype=torch.float16,
    )
    context = ProcessingContext()

    # Warmup
    print(f"\nWarming up ({warmup_frames} frames)...")
    for i in range(warmup_frames):
        _ = processor.process_frame(dummy, context)
    torch.cuda.synchronize()

    # Benchmark
    print(f"\nRunning benchmark ({num_frames} frames)...")
    latencies = []

    for i in range(num_frames):
        # Create random frame each iteration
        frame = torch.randn(
            1, 3, resolution, resolution,
            device="cuda",
            dtype=torch.float16,
        )
        context = ProcessingContext()

        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = processor.process_frame(frame, context)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        latencies.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1}/{num_frames}: {elapsed:.2f}ms")

    processor.cleanup()

    # Calculate statistics
    latencies = np.array(latencies)
    stats = {
        "resolution": f"{resolution}x{resolution}",
        "tensorrt": use_tensorrt,
        "num_frames": num_frames,
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "vram_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Mean:  {stats['mean_ms']:.2f}ms (± {stats['std_ms']:.2f}ms)")
    print(f"P50:   {stats['p50_ms']:.2f}ms")
    print(f"P95:   {stats['p95_ms']:.2f}ms")
    print(f"P99:   {stats['p99_ms']:.2f}ms")
    print(f"Min:   {stats['min_ms']:.2f}ms")
    print(f"Max:   {stats['max_ms']:.2f}ms")
    print(f"\nVRAM Allocated: {stats['vram_allocated_gb']:.2f} GB")
    print(f"VRAM Reserved:  {stats['vram_reserved_gb']:.2f} GB")

    # Check target
    target_ms = 50.0  # 24fps requires <41.6ms, aim for <50ms
    if stats["mean_ms"] < target_ms:
        print(f"\n✅ PASS: Mean latency {stats['mean_ms']:.2f}ms < {target_ms}ms target")
    else:
        print(f"\n❌ FAIL: Mean latency {stats['mean_ms']:.2f}ms > {target_ms}ms target")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark StreamDiffusionProcessor")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames")
    parser.add_argument("--resolution", type=int, default=768, help="Square resolution")
    parser.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames")
    args = parser.parse_args()

    # Run benchmark
    stats = benchmark_diffusion(
        num_frames=args.num_frames,
        resolution=args.resolution,
        use_tensorrt=not args.no_tensorrt,
        warmup_frames=args.warmup,
    )

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
