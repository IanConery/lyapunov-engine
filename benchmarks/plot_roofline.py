import argparse
import numpy as np


def generate_roofline_data(peak_tflops: float = 120.0, peak_bandwidth_gbs: float = 1000.0):
    """Compute theoretical roofline curve: Achievable Performance = min(Peak FLOPS, Operational Intensity * Peak Bandwidth)."""
    arithmetic_intensities = np.logspace(-2, 3, 100) # FLOPs / Byte
    achievable_tflops = np.minimum(peak_tflops, arithmetic_intensities * (peak_bandwidth_gbs / 1000.0))

    ridge_point = peak_tflops / (peak_bandwidth_gbs / 1000.0)
    print("=" * 60)
    print("GPU Theoretical Roofline Parameters")
    print("=" * 60)
    print(f"Peak Compute Performance: {peak_tflops:.1f} TFLOP/s")
    print(f"Peak Memory Bandwidth:   {peak_bandwidth_gbs:.1f} GB/s")
    print(f"Roofline Ridge Point:     {ridge_point:.2f} FLOPs/Byte")
    print("-" * 60)
    print(f"Memory-Bound Regime:      Operational Intensity < {ridge_point:.2f} FLOPs/Byte")
    print(f"Compute-Bound Regime:     Operational Intensity >= {ridge_point:.2f} FLOPs/Byte")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-tflops", type=float, default=120.0)
    parser.add_argument("--peak-bandwidth", type=float, default=1000.0)
    args = parser.parse_args()

    generate_roofline_data(args.peak_tflops, args.peak_bandwidth)
