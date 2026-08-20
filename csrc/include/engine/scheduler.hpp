#pragma once

#include "block_manager.hpp"
#include "sequence.hpp"
#include <deque>
#include <memory>
#include <vector>

namespace lyapunov {
namespace engine {

struct SchedulerConfig {
  int max_num_seqs{256};
  int max_num_batched_tokens{4096};
  int max_model_len{4096};
  int eos_token_id{128001}; // Default Llama-3 EOS token ID
};

struct SchedulerOutputs {
  std::vector<SequencePtr> scheduled_seqs;
  bool is_prefill{false};
  int num_batched_tokens{0};
  std::vector<SequencePtr> finished_seqs;
};

class ContinuousScheduler {
public:
  ContinuousScheduler(const SchedulerConfig &config,
                      std::shared_ptr<BlockSpaceManager> block_manager);

  void add_sequence(SequencePtr seq);
  void abort_sequence(int64_t seq_id);

  bool has_unfinished_sequences() const;
  int get_num_waiting_sequences() const {
    return static_cast<int>(waiting_.size());
  }
  int get_num_running_sequences() const {
    return static_cast<int>(running_.size());
  }

  SchedulerOutputs schedule();
  void post_step(const std::vector<SequencePtr> &seqs,
                 const std::vector<int32_t> &next_tokens);

private:
  SchedulerConfig config_;
  std::shared_ptr<BlockSpaceManager> block_manager_;

  std::deque<SequencePtr> waiting_;
  std::vector<SequencePtr> running_;
  std::deque<SequencePtr> swapped_;
};

} // namespace engine
} // namespace lyapunov
