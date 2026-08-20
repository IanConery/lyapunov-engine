#pragma once

#include "sequence.hpp"
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lyapunov {
namespace engine {

class BlockSpaceManager {
public:
  BlockSpaceManager(int num_blocks, int block_size = 16,
                    bool enable_prefix_caching = true);

  int get_num_blocks() const { return num_blocks_; }
  int get_block_size() const { return block_size_; }
  int get_num_free_blocks() const {
    return static_cast<int>(free_blocks_.size());
  }
  int get_num_used_blocks() const {
    return num_blocks_ - get_num_free_blocks();
  }

  bool can_allocate(const SequencePtr &seq) const;
  void allocate(const SequencePtr &seq);
  bool can_append_slot(const SequencePtr &seq) const;
  void append_slot(const SequencePtr &seq);
  void free(const SequencePtr &seq);

  void reset();

private:
  int num_blocks_;
  int block_size_;
  bool enable_prefix_caching_;

  std::deque<int32_t> free_blocks_;
  std::vector<int32_t> ref_counts_;
  std::unordered_map<int64_t, std::vector<int32_t>> seq_to_blocks_;
};

} // namespace engine
} // namespace lyapunov
