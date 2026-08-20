#pragma once

#include "utils/cuda_utils.cuh"
#include <functional>
#include <memory>
#include <unordered_map>

namespace lyapunov {
namespace engine {

class CUDAGraphRunner {
public:
  CUDAGraphRunner();
  ~CUDAGraphRunner();

  bool is_captured(int batch_size) const;
  void capture(int batch_size,
               std::function<void(cudaStream_t)> model_step_func,
               cudaStream_t stream = nullptr);
  void replay(int batch_size, cudaStream_t stream = nullptr);
  void reset();

private:
  std::unordered_map<int, cudaGraph_t> graphs_;
  std::unordered_map<int, cudaGraphExec_t> graph_execs_;
};

} // namespace engine
} // namespace lyapunov
