#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lyapunov {
namespace engine {

enum class SequenceStatus {
  WAITING,
  RUNNING,
  SWAPPED,
  FINISHED_STOPPED,
  FINISHED_LENGTH_CAPPED,
  FINISHED_ABORTED
};

struct SamplingParams {
  float temperature{1.0f};
  float top_p{1.0f};
  int top_k{-1};
  int max_tokens{128};
  bool ignore_eos{false};
};

class Sequence {
public:
  Sequence(int64_t seq_id, const std::vector<int32_t> &prompt_tokens,
           const SamplingParams &sampling_params)
      : seq_id_(seq_id), prompt_token_ids_(prompt_tokens), output_token_ids_(),
        sampling_params_(sampling_params), status_(SequenceStatus::WAITING),
        num_computed_tokens_(0) {}

  int64_t get_seq_id() const { return seq_id_; }
  SequenceStatus get_status() const { return status_; }
  void set_status(SequenceStatus status) { status_ = status; }

  const std::vector<int32_t> &get_prompt_tokens() const {
    return prompt_token_ids_;
  }
  const std::vector<int32_t> &get_output_tokens() const {
    return output_token_ids_;
  }
  const SamplingParams &get_sampling_params() const { return sampling_params_; }

  size_t get_prompt_len() const { return prompt_token_ids_.size(); }
  size_t get_output_len() const { return output_token_ids_.size(); }
  size_t get_total_len() const {
    return prompt_token_ids_.size() + output_token_ids_.size();
  }

  int32_t get_last_token_id() const {
    if (!output_token_ids_.empty()) {
      return output_token_ids_.back();
    }
    return prompt_token_ids_.back();
  }

  void append_token_id(int32_t token_id) {
    output_token_ids_.push_back(token_id);
  }

  bool is_finished() const {
    return status_ == SequenceStatus::FINISHED_STOPPED ||
           status_ == SequenceStatus::FINISHED_LENGTH_CAPPED ||
           status_ == SequenceStatus::FINISHED_ABORTED;
  }

  size_t get_num_computed_tokens() const { return num_computed_tokens_; }
  void set_num_computed_tokens(size_t num) { num_computed_tokens_ = num; }

  const std::vector<int32_t> &get_block_table() const { return block_table_; }
  std::vector<int32_t> &get_mutable_block_table() { return block_table_; }
  void set_block_table(const std::vector<int32_t> &table) {
    block_table_ = table;
  }

private:
  int64_t seq_id_;
  std::vector<int32_t> prompt_token_ids_;
  std::vector<int32_t> output_token_ids_;
  SamplingParams sampling_params_;
  SequenceStatus status_;
  size_t num_computed_tokens_;
  std::vector<int32_t> block_table_;
};

using SequencePtr = std::shared_ptr<Sequence>;

} // namespace engine
} // namespace lyapunov
