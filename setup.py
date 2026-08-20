import os
import sys
from setuptools import find_packages, setup

# Check if CUDA extension build is requested and supported
build_cuda_ext = os.environ.get("BUILD_CUDA_EXT", "1") not in ("0", "false", "False", "no")
no_cuda = os.environ.get("NO_CUDA", "0") in ("1", "true", "True", "yes")

ext_modules = []
cmdclass = {}

if build_cuda_ext and not no_cuda:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

        if CUDA_HOME is not None or os.environ.get("FORCE_CUDA", "0") == "1":
            # Allow minor CUDA toolkit version mismatch between host nvcc and prebuilt PyTorch
            os.environ["TORCH_ALLOW_CUDA_VERSION_MISMATCH"] = "1"

            # Target compute capabilities (Ampere sm_86, Ada sm_89, Hopper sm_90)
            if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;8.9;9.0"

            this_dir = os.path.dirname(os.path.abspath(__file__))

            sources = [
                "csrc/bindings.cpp",
                "csrc/kernels/fused_rmsnorm_swiglu.cu",
                "csrc/kernels/flash_attn_v2.cu",
                "csrc/kernels/flash_decoding.cu",
                "csrc/kernels/paged_attention.cu",
                "csrc/kernels/quant_ops.cu",
                "csrc/kernels/fused_rope_paged.cu",
                "csrc/kernels/speculative.cu",
                "csrc/engine/block_manager.cpp",
                "csrc/engine/scheduler.cpp",
                "csrc/engine/cuda_graph.cpp",
            ]

            include_dirs = [
                os.path.join(this_dir, "csrc", "include"),
            ]

            extra_compile_args = {
                "cxx": ["-O3", "-std=c++20", "-Wall", "-Wextra"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    "-D__CUDA_NO_HALF_OPERATORS__=0",
                    "-D__CUDA_NO_HALF_CONVERSIONS__=0",
                ],
            }

            ext_modules = [
                CUDAExtension(
                    name="lyapunov_engine._C",
                    sources=sources,
                    include_dirs=include_dirs,
                    extra_compile_args=extra_compile_args,
                )
            ]
            cmdclass = {"build_ext": BuildExtension}
        else:
            print("CUDA_HOME not found. Building pure Python package without CUDA extension.")
    except Exception as exc:
        print(f"CUDA extension setup failed: {exc}. Falling back to pure Python package.")
        ext_modules = []
        cmdclass = {}

setup(
    name="lyapunov-engine",
    version="0.1.0",
    description="Custom CUDA LLM Inference Engine with Dynamic Trajectory & Uncertainty Analysis",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
