#include "engine/cuda_graph.hpp"
#include "utils/logging.hpp"

namespace lyapunov {
namespace engine {

CUDAGraphRunner::CUDAGraphRunner() {}

CUDAGraphRunner::~CUDAGraphRunner() { reset(); }

bool CUDAGraphRunner::is_captured(int batch_size) const {
  return graph_execs_.find(batch_size) != graph_execs_.end();
}

void CUDAGraphRunner::capture(int batch_size,
                              std::function<void(cudaStream_t)> model_step_func,
                              cudaStream_t stream) {
  if (is_captured(batch_size)) {
    return;
  }

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  // Warmup step prior to capture
  model_step_func(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Begin capture
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  model_step_func(stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  // Instantiate executable graph
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  graphs_[batch_size] = graph;
  graph_execs_[batch_size] = graph_exec;

  LYAPUNOV_LOG_INFO("Successfully captured CUDA Graph for batch size "
                    << batch_size);
}

void CUDAGraphRunner::replay(int batch_size, cudaStream_t stream) {
  auto it = graph_execs_.find(batch_size);
  if (it == graph_execs_.end()) {
    throw std::runtime_error("CUDA Graph not captured for batch size " +
                             std::to_string(batch_size));
  }

  CUDA_CHECK(cudaGraphLaunch(it->second, stream));
}

void CUDAGraphRunner::reset() {
  for (auto &pair : graph_execs_) {
    if (pair.second) {
      cudaGraphExecDestroy(pair.second);
    }
  }
  graph_execs_.clear();

  for (auto &pair : graphs_) {
    if (pair.second) {
      cudaGraphDestroy(pair.second);
    }
  }
  graphs_.clear();
}

} // namespace engine
} // namespace lyapunov
