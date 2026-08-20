#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "engine/block_manager.hpp"
#include "engine/scheduler.hpp"
#include "engine/sequence.hpp"
#include "kernels/flash_attn.cuh"
#include "kernels/flash_decoding.cuh"
#include "kernels/fused_ops.cuh"
#include "kernels/fused_rope.cuh"
#include "kernels/paged_attn.cuh"
#include "kernels/quant_ops.cuh"
#include "kernels/speculative.cuh"

namespace py = pybind11;
using namespace lyapunov::kernels;
using namespace lyapunov::engine;

// ----------------------------------------------------------------------------
// Kernel PyTorch Bindings
// ----------------------------------------------------------------------------

std::vector<torch::Tensor>
rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps,
                std::optional<torch::Tensor> residual = std::nullopt) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int hidden_dim = static_cast<int>(input.size(-1));
  int num_tokens = static_cast<int>(input.numel() / hidden_dim);

  auto out = torch::empty_like(input);
  torch::Tensor out_res;
  void *res_ptr = nullptr;
  void *out_res_ptr = nullptr;

  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    out_res = torch::empty_like(*residual);
    res_ptr = residual->data_ptr();
    out_res_ptr = out_res.data_ptr();
  }

  cudaDataType_t dtype;
  if (input.scalar_type() == at::ScalarType::Float) {
    dtype = CUDA_R_32F;
  } else if (input.scalar_type() == at::ScalarType::Half) {
    dtype = CUDA_R_16F;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported data type for rmsnorm: only Float and Half are supported");
  }

  launch_rmsnorm(out.data_ptr(), out_res_ptr, input.data_ptr(), res_ptr,
                 weight.data_ptr(), static_cast<float>(eps), num_tokens,
                 hidden_dim, dtype, stream);

  if (residual.has_value()) {
    return {out, out_res};
  }
  return {out};
}

torch::Tensor swiglu_forward(torch::Tensor gate_up) {
  TORCH_CHECK(gate_up.is_cuda(), "gate_up must be a CUDA tensor");
  TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int last_dim = static_cast<int>(gate_up.size(-1));
  TORCH_CHECK(last_dim % 2 == 0, "Last dimension of gate_up must be even");
  int intermediate_dim = last_dim / 2;
  int num_tokens = static_cast<int>(gate_up.numel() / last_dim);

  auto sizes = gate_up.sizes().vec();
  sizes.back() = intermediate_dim;
  auto out = torch::empty(sizes, gate_up.options());

  cudaDataType_t dtype;
  if (gate_up.scalar_type() == at::ScalarType::Float) {
    dtype = CUDA_R_32F;
  } else if (gate_up.scalar_type() == at::ScalarType::Half) {
    dtype = CUDA_R_16F;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported data type for swiglu: only Float and Half are supported");
  }

  launch_swiglu(out.data_ptr(), gate_up.data_ptr(), num_tokens,
                intermediate_dim, dtype, stream);

  return out;
}

torch::Tensor flash_attn_v2_fwd_tensor(torch::Tensor q, torch::Tensor k,
                                       torch::Tensor v,
                                       std::optional<double> sm_scale,
                                       bool is_causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "All tensors must be on CUDA");
  TORCH_CHECK(q.dim() == 4,
              "q must have shape [batch, num_heads, seqlen, head_dim]");
  TORCH_CHECK(k.dim() == 4,
              "k must have shape [batch, num_kv_heads, seqlen, head_dim]");
  TORCH_CHECK(v.dim() == 4,
              "v must have shape [batch, num_kv_heads, seqlen, head_dim]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = static_cast<int>(q.size(0));
  int num_heads = static_cast<int>(q.size(1));
  int q_seqlen = static_cast<int>(q.size(2));
  int head_dim = static_cast<int>(q.size(3));

  int num_kv_heads = static_cast<int>(k.size(1));
  int kv_seqlen = static_cast<int>(k.size(2));

  auto out = torch::empty_like(q);

  float scale = sm_scale.has_value()
                    ? static_cast<float>(*sm_scale)
                    : 1.0f / std::sqrt(static_cast<float>(head_dim));

  FlashAttnParams params;
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.out_ptr = out.data_ptr();
  params.lse_ptr = nullptr;

  params.batch_size = batch_size;
  params.num_heads = num_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim = head_dim;
  params.q_seqlen = q_seqlen;
  params.kv_seqlen = kv_seqlen;

  params.q_batch_stride = q.stride(0);
  params.q_head_stride = q.stride(1);
  params.q_seq_stride = q.stride(2);

  params.k_batch_stride = k.stride(0);
  params.k_head_stride = k.stride(1);
  params.k_seq_stride = k.stride(2);

  params.v_batch_stride = v.stride(0);
  params.v_head_stride = v.stride(1);
  params.v_seq_stride = v.stride(2);

  params.out_batch_stride = out.stride(0);
  params.out_head_stride = out.stride(1);
  params.out_seq_stride = out.stride(2);

  params.sm_scale = scale;
  params.is_causal = is_causal;

  if (q.scalar_type() == at::ScalarType::Float) {
    params.dtype = CUDA_R_32F;
  } else if (q.scalar_type() == at::ScalarType::Half) {
    params.dtype = CUDA_R_16F;
  } else {
    TORCH_CHECK(false, "Unsupported data type for flash_attn_v2");
  }

  launch_flash_attn_v2(params, stream);
  return out;
}

torch::Tensor flash_decoding_tensor(torch::Tensor q, torch::Tensor k,
                                    torch::Tensor v, int num_partitions,
                                    std::optional<double> sm_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "All tensors must be on CUDA");
  TORCH_CHECK(q.dim() == 4 && q.size(2) == 1,
              "q must have shape [batch, num_heads, 1, head_dim]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = static_cast<int>(q.size(0));
  int num_heads = static_cast<int>(q.size(1));
  int head_dim = static_cast<int>(q.size(3));
  int num_kv_heads = static_cast<int>(k.size(1));
  int kv_seqlen = static_cast<int>(k.size(2));

  auto out = torch::empty_like(q);
  auto mid_out = torch::empty({batch_size, num_heads, num_partitions, head_dim},
                              q.options());
  auto mid_lse = torch::empty({batch_size, num_heads, num_partitions},
                              q.options().dtype(torch::kFloat32));

  float scale = sm_scale.has_value()
                    ? static_cast<float>(*sm_scale)
                    : 1.0f / std::sqrt(static_cast<float>(head_dim));

  FlashDecodingParams params;
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.out_ptr = out.data_ptr();
  params.mid_out_ptr = mid_out.data_ptr();
  params.mid_lse_ptr = static_cast<float *>(mid_lse.data_ptr());

  params.batch_size = batch_size;
  params.num_heads = num_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim = head_dim;
  params.kv_seqlen = kv_seqlen;
  params.num_partitions = num_partitions;

  params.q_batch_stride = q.stride(0);
  params.q_head_stride = q.stride(1);

  params.k_batch_stride = k.stride(0);
  params.k_head_stride = k.stride(1);
  params.k_seq_stride = k.stride(2);

  params.v_batch_stride = v.stride(0);
  params.v_head_stride = v.stride(1);
  params.v_seq_stride = v.stride(2);

  params.out_batch_stride = out.stride(0);
  params.out_head_stride = out.stride(1);

  params.sm_scale = scale;

  if (q.scalar_type() == at::ScalarType::Float) {
    params.dtype = CUDA_R_32F;
  } else if (q.scalar_type() == at::ScalarType::Half) {
    params.dtype = CUDA_R_16F;
  } else {
    TORCH_CHECK(false, "Unsupported data type for flash_decoding");
  }

  launch_flash_decoding(params, stream);
  return out;
}

torch::Tensor paged_attention_v1(torch::Tensor q, torch::Tensor k_cache,
                                 torch::Tensor v_cache,
                                 torch::Tensor block_tables,
                                 torch::Tensor context_lens,
                                 std::optional<double> sm_scale) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
              "All tensors must be on CUDA");
  TORCH_CHECK(block_tables.is_cuda() && context_lens.is_cuda(),
              "Metadata tensors must be on CUDA");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = static_cast<int>(q.size(0));
  int num_heads = static_cast<int>(q.size(1));
  int head_dim = static_cast<int>(q.size(2));

  int num_blocks = static_cast<int>(k_cache.size(0));
  int num_kv_heads = static_cast<int>(k_cache.size(1));
  int block_size = static_cast<int>(k_cache.size(2));
  int max_num_blocks_per_seq = static_cast<int>(block_tables.size(1));

  auto out = torch::empty_like(q);
  float scale = sm_scale.has_value()
                    ? static_cast<float>(*sm_scale)
                    : 1.0f / std::sqrt(static_cast<float>(head_dim));

  PagedAttentionParams params;
  params.out_ptr = out.data_ptr();
  params.q_ptr = q.data_ptr();
  params.k_cache_ptr = k_cache.data_ptr();
  params.v_cache_ptr = v_cache.data_ptr();
  params.block_tables = block_tables.data_ptr<int32_t>();
  params.context_lens = context_lens.data_ptr<int32_t>();

  params.batch_size = batch_size;
  params.num_heads = num_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim = head_dim;
  params.block_size = block_size;
  params.max_num_blocks_per_seq = max_num_blocks_per_seq;

  params.q_batch_stride = q.stride(0);
  params.q_head_stride = q.stride(1);

  params.k_block_stride = k_cache.stride(0);
  params.k_head_stride = k_cache.stride(1);
  params.k_token_stride = k_cache.stride(2);

  params.v_block_stride = v_cache.stride(0);
  params.v_head_stride = v_cache.stride(1);
  params.v_token_stride = v_cache.stride(2);

  params.out_batch_stride = out.stride(0);
  params.out_head_stride = out.stride(1);

  params.sm_scale = scale;

  if (q.scalar_type() == at::ScalarType::Float) {
    params.dtype = CUDA_R_32F;
  } else if (q.scalar_type() == at::ScalarType::Half) {
    params.dtype = CUDA_R_16F;
  } else {
    TORCH_CHECK(false, "Unsupported data type for paged_attention");
  }

  launch_paged_attention(params, stream);
  return out;
}

// ----------------------------------------------------------------------------
// Quantization Kernel Wrappers
// ----------------------------------------------------------------------------

torch::Tensor dequantize_q4_0_tensor(torch::Tensor src, int64_t num_elements) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(src));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto out = torch::empty({num_elements}, src.options().dtype(torch::kFloat32));
  launch_dequantize_q4_0(src.data_ptr(), out.data_ptr<float>(), num_elements,
                         stream);
  return out;
}

torch::Tensor dequantize_q8_0_tensor(torch::Tensor src, int64_t num_elements) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(src));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto out = torch::empty({num_elements}, src.options().dtype(torch::kFloat32));
  launch_dequantize_q8_0(src.data_ptr(), out.data_ptr<float>(), num_elements,
                         stream);
  return out;
}

torch::Tensor dequantize_q4_k_tensor(torch::Tensor src, int64_t num_elements) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(src));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto out = torch::empty({num_elements}, src.options().dtype(torch::kFloat32));
  launch_dequantize_q4_k(src.data_ptr(), out.data_ptr<float>(), num_elements,
                         stream);
  return out;
}

torch::Tensor quant_gemv_tensor(torch::Tensor x, torch::Tensor qweight,
                                std::optional<torch::Tensor> scales,
                                int64_t in_features, int64_t out_features,
                                int64_t qtype_enum) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = x.size(0);
  auto y = torch::empty({batch_size, out_features},
                        x.options().dtype(torch::kFloat32));

  const float *scales_ptr =
      scales.has_value() ? scales.value().data_ptr<float>() : nullptr;
  QuantType qtype = static_cast<QuantType>(qtype_enum);

  launch_quant_gemv(x.data_ptr<float>(), qweight.data_ptr(), scales_ptr,
                    y.data_ptr<float>(), batch_size, in_features, out_features,
                    qtype, stream);

  return y;
}

torch::Tensor fp8_gemm_tensor(torch::Tensor x_fp8, torch::Tensor w_fp8,
                              double scale_x = 1.0, double scale_w = 1.0) {
  TORCH_CHECK(x_fp8.is_cuda(), "x_fp8 must be a CUDA tensor");
  TORCH_CHECK(w_fp8.is_cuda(), "w_fp8 must be a CUDA tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x_fp8));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int m = x_fp8.size(0);
  int k = x_fp8.size(1);
  int n = w_fp8.size(0);

  auto y = torch::empty(
      {m, n},
      torch::TensorOptions().dtype(torch::kFloat32).device(x_fp8.device()));
  float sx = static_cast<float>(scale_x);
  float sw = static_cast<float>(scale_w);

  launch_fp8_gemm(x_fp8.data_ptr(), w_fp8.data_ptr(), &sx, &sw,
                  y.data_ptr<float>(), m, k, n, stream);

  return y;
}

torch::Tensor marlin_gemm_tensor(torch::Tensor x, torch::Tensor qweight,
                                 torch::Tensor scales) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int m = x.size(0);
  int in_features = x.size(1);
  int out_features = qweight.size(0);

  auto y = torch::empty({m, out_features}, x.options().dtype(torch::kFloat32));

  launch_marlin_gemm(x.data_ptr<float>(), qweight.data_ptr<int32_t>(),
                     scales.data_ptr<float>(), y.data_ptr<float>(), m,
                     in_features, out_features, stream);

  return y;
}

void fused_rope_paged_tensor(
    torch::Tensor q_out, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor q_in, torch::Tensor k_in, torch::Tensor v_in,
    torch::Tensor cos_sin_cache,
    std::optional<torch::Tensor> block_tables = std::nullopt,
    std::optional<torch::Tensor> context_lens = std::nullopt) {
  TORCH_CHECK(q_in.is_cuda(), "Inputs must be CUDA tensors");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_in));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  FusedRoPEParams params;
  params.q_out = q_out.data_ptr();
  params.k_cache = k_cache.data_ptr();
  params.v_cache = v_cache.data_ptr();
  params.q_in = q_in.data_ptr();
  params.k_in = k_in.data_ptr();
  params.v_in = v_in.data_ptr();
  params.cos_sin_cache = cos_sin_cache.data_ptr<float>();

  params.block_tables = block_tables.has_value()
                            ? block_tables.value().data_ptr<int32_t>()
                            : nullptr;
  params.context_lens = context_lens.has_value()
                            ? context_lens.value().data_ptr<int32_t>()
                            : nullptr;
  params.seq_idx_map = nullptr;

  params.num_tokens = q_in.size(0);
  params.num_heads = q_in.size(1);
  params.num_kv_heads = k_in.size(1);
  params.head_dim = q_in.size(2);
  params.block_size = k_cache.size(2);
  params.max_blocks_per_seq =
      block_tables.has_value() ? block_tables.value().size(1) : 0;
  params.dtype =
      (q_in.scalar_type() == at::ScalarType::Half) ? CUDA_R_16F : CUDA_R_32F;

  launch_fused_rope_paged(params, stream);
}

std::tuple<torch::Tensor, torch::Tensor> speculative_verification_tensor(
    torch::Tensor target_probs, torch::Tensor draft_probs,
    torch::Tensor draft_tokens, torch::Tensor rand_uniform) {
  TORCH_CHECK(target_probs.is_cuda(), "Inputs must be CUDA tensors");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(target_probs));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int batch_size = target_probs.size(0);
  int num_draft = target_probs.size(1);
  int vocab_size = target_probs.size(2);

  auto accepted_tokens = torch::zeros({batch_size, num_draft + 1},
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(target_probs.device()));
  auto num_accepted =
      torch::zeros({batch_size}, torch::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(target_probs.device()));

  launch_speculative_verification(
      target_probs.data_ptr<float>(), draft_probs.data_ptr<float>(),
      draft_tokens.data_ptr<int32_t>(), rand_uniform.data_ptr<float>(),
      accepted_tokens.data_ptr<int32_t>(), num_accepted.data_ptr<int32_t>(),
      batch_size, num_draft, vocab_size, stream);

  return std::make_tuple(accepted_tokens, num_accepted);
}

// ----------------------------------------------------------------------------
// PYBIND11 Module Registration
// ----------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "lyapunov-engine custom CUDA kernel library and serving runtime";

  // Custom Kernels
  m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm forward kernel",
        py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-5,
        py::arg("residual") = py::none());
  m.def("swiglu_forward", &swiglu_forward, "Fused SwiGLU forward kernel",
        py::arg("gate_up"));
  m.def("flash_attn_v2_fwd", &flash_attn_v2_fwd_tensor,
        "FlashAttention-2 forward kernel", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("sm_scale") = py::none(),
        py::arg("is_causal") = true);
  m.def("flash_decoding", &flash_decoding_tensor,
        "FlashDecoding split-KV kernel", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("num_partitions") = 4,
        py::arg("sm_scale") = py::none());
  m.def("paged_attention_v1", &paged_attention_v1,
        "PagedAttention v1 decoding kernel", py::arg("q"), py::arg("k_cache"),
        py::arg("v_cache"), py::arg("block_tables"), py::arg("context_lens"),
        py::arg("sm_scale") = py::none());
  m.def("fused_rope_paged", &fused_rope_paged_tensor,
        "Fused RoPE and paged KV write kernel", py::arg("q_out"),
        py::arg("k_cache"), py::arg("v_cache"), py::arg("q_in"),
        py::arg("k_in"), py::arg("v_in"), py::arg("cos_sin_cache"),
        py::arg("block_tables") = py::none(),
        py::arg("context_lens") = py::none());
  m.def("speculative_verification", &speculative_verification_tensor,
        "Speculative decoding parallel verification", py::arg("target_probs"),
        py::arg("draft_probs"), py::arg("draft_tokens"),
        py::arg("rand_uniform"));

  // Quantization Kernels
  m.def("dequantize_q4_0", &dequantize_q4_0_tensor,
        "Dequantize Q4_0 block tensor to FP32", py::arg("src"),
        py::arg("num_elements"));
  m.def("dequantize_q8_0", &dequantize_q8_0_tensor,
        "Dequantize Q8_0 block tensor to FP32", py::arg("src"),
        py::arg("num_elements"));
  m.def("dequantize_q4_k", &dequantize_q4_k_tensor,
        "Dequantize Q4_K super-block tensor to FP32", py::arg("src"),
        py::arg("num_elements"));
  m.def("quant_gemv", &quant_gemv_tensor,
        "Quantized GEMV matrix multiplication", py::arg("x"),
        py::arg("qweight"), py::arg("scales") = py::none(),
        py::arg("in_features"), py::arg("out_features"), py::arg("qtype_enum"));
  m.def("fp8_gemm", &fp8_gemm_tensor,
        "FP8 E4M3 matrix multiplication with scaling", py::arg("x_fp8"),
        py::arg("w_fp8"), py::arg("scale_x") = 1.0, py::arg("scale_w") = 1.0);
  m.def("marlin_gemm", &marlin_gemm_tensor,
        "Marlin W4A16 matrix multiplication", py::arg("x"), py::arg("qweight"),
        py::arg("scales"));

  // C++ Engine Types
  py::enum_<SequenceStatus>(m, "SequenceStatus")
      .value("WAITING", SequenceStatus::WAITING)
      .value("RUNNING", SequenceStatus::RUNNING)
      .value("SWAPPED", SequenceStatus::SWAPPED)
      .value("FINISHED_STOPPED", SequenceStatus::FINISHED_STOPPED)
      .value("FINISHED_LENGTH_CAPPED", SequenceStatus::FINISHED_LENGTH_CAPPED)
      .value("FINISHED_ABORTED", SequenceStatus::FINISHED_ABORTED);

  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init<>())
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_p", &SamplingParams::top_p)
      .def_readwrite("top_k", &SamplingParams::top_k)
      .def_readwrite("max_tokens", &SamplingParams::max_tokens)
      .def_readwrite("ignore_eos", &SamplingParams::ignore_eos);

  py::class_<Sequence, SequencePtr>(m, "Sequence")
      .def(py::init<int64_t, const std::vector<int32_t> &,
                    const SamplingParams &>())
      .def("get_seq_id", &Sequence::get_seq_id)
      .def("get_status", &Sequence::get_status)
      .def("get_prompt_tokens", &Sequence::get_prompt_tokens)
      .def("get_output_tokens", &Sequence::get_output_tokens)
      .def("get_total_len", &Sequence::get_total_len)
      .def("get_last_token_id", &Sequence::get_last_token_id)
      .def("is_finished", &Sequence::is_finished)
      .def("get_block_table", &Sequence::get_block_table);

  py::class_<BlockSpaceManager, std::shared_ptr<BlockSpaceManager>>(
      m, "BlockSpaceManager")
      .def(py::init<int, int, bool>(), py::arg("num_blocks"),
           py::arg("block_size") = 16, py::arg("enable_prefix_caching") = true)
      .def("get_num_blocks", &BlockSpaceManager::get_num_blocks)
      .def("get_block_size", &BlockSpaceManager::get_block_size)
      .def("get_num_free_blocks", &BlockSpaceManager::get_num_free_blocks)
      .def("get_num_used_blocks", &BlockSpaceManager::get_num_used_blocks)
      .def("can_allocate", &BlockSpaceManager::can_allocate)
      .def("allocate", &BlockSpaceManager::allocate)
      .def("can_append_slot", &BlockSpaceManager::can_append_slot)
      .def("append_slot", &BlockSpaceManager::append_slot)
      .def("free", &BlockSpaceManager::free)
      .def("reset", &BlockSpaceManager::reset);

  py::class_<SchedulerConfig>(m, "SchedulerConfig")
      .def(py::init<>())
      .def_readwrite("max_num_seqs", &SchedulerConfig::max_num_seqs)
      .def_readwrite("max_num_batched_tokens",
                     &SchedulerConfig::max_num_batched_tokens)
      .def_readwrite("max_model_len", &SchedulerConfig::max_model_len)
      .def_readwrite("eos_token_id", &SchedulerConfig::eos_token_id);

  py::class_<SchedulerOutputs>(m, "SchedulerOutputs")
      .def_readonly("scheduled_seqs", &SchedulerOutputs::scheduled_seqs)
      .def_readonly("is_prefill", &SchedulerOutputs::is_prefill)
      .def_readonly("num_batched_tokens", &SchedulerOutputs::num_batched_tokens)
      .def_readonly("finished_seqs", &SchedulerOutputs::finished_seqs);

  py::class_<ContinuousScheduler>(m, "ContinuousScheduler")
      .def(py::init<const SchedulerConfig &,
                    std::shared_ptr<BlockSpaceManager>>())
      .def("add_sequence", &ContinuousScheduler::add_sequence)
      .def("abort_sequence", &ContinuousScheduler::abort_sequence)
      .def("has_unfinished_sequences",
           &ContinuousScheduler::has_unfinished_sequences)
      .def("get_num_waiting_sequences",
           &ContinuousScheduler::get_num_waiting_sequences)
      .def("get_num_running_sequences",
           &ContinuousScheduler::get_num_running_sequences)
      .def("schedule", &ContinuousScheduler::schedule)
      .def("post_step", &ContinuousScheduler::post_step);
}
