import os

from setuptools import find_packages, setup

# Allow minor CUDA toolkit version mismatch between host nvcc and prebuilt PyTorch
os.environ["TORCH_ALLOW_CUDA_VERSION_MISMATCH"] = "1"

# Target compute capabilities (Ampere sm_86, Ada sm_89, Hopper sm_90)
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;8.9;9.0"

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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

setup(
    name="lyapunov-engine",
    version="0.1.0",
    description="Custom CUDA LLM Inference Engine with Dynamic Trajectory & Uncertainty Analysis",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
