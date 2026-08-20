#include "engine/block_manager.hpp"
#include <algorithm>

namespace lyapunov {
namespace engine {

BlockSpaceManager::BlockSpaceManager(int num_blocks, int block_size, bool enable_prefix_caching)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      enable_prefix_caching_(enable_prefix_caching),
      ref_counts_(num_blocks, 0) {
    reset();
}

void BlockSpaceManager::reset() {
    free_blocks_.clear();
    std::fill(ref_counts_.begin(), ref_counts_.end(), 0);
    seq_to_blocks_.clear();

    for (int i = 0; i < num_blocks_; ++i) {
        free_blocks_.push_back(i);
    }
}

bool BlockSpaceManager::can_allocate(const SequencePtr& seq) const {
    size_t total_tokens = seq->get_total_len();
    int needed_blocks = static_cast<int>((total_tokens + block_size_ - 1) / block_size_);
    return static_cast<int>(free_blocks_.size()) >= needed_blocks;
}

void BlockSpaceManager::allocate(const SequencePtr& seq) {
    size_t total_tokens = seq->get_total_len();
    int needed_blocks = static_cast<int>((total_tokens + block_size_ - 1) / block_size_);

    if (static_cast<int>(free_blocks_.size()) < needed_blocks) {
        throw std::runtime_error("Out of memory: Insufficient free blocks in BlockSpaceManager");
    }

    std::vector<int32_t> allocated;
    allocated.reserve(needed_blocks);

    for (int i = 0; i < needed_blocks; ++i) {
        int32_t block_id = free_blocks_.front();
        free_blocks_.pop_front();
        ref_counts_[block_id] = 1;
        allocated.push_back(block_id);
    }

    seq->set_block_table(allocated);
    seq_to_blocks_[seq->get_seq_id()] = allocated;
}

bool BlockSpaceManager::can_append_slot(const SequencePtr& seq) const {
    size_t total_tokens = seq->get_total_len();
    if (total_tokens % block_size_ == 1) {
        return !free_blocks_.empty();
    }
    return true;
}

void BlockSpaceManager::append_slot(const SequencePtr& seq) {
    size_t total_tokens = seq->get_total_len();
    if (total_tokens % block_size_ == 1) {
        if (free_blocks_.empty()) {
            throw std::runtime_error("Out of memory: No free blocks available to append slot");
        }
        int32_t new_block_id = free_blocks_.front();
        free_blocks_.pop_front();
        ref_counts_[new_block_id] = 1;

        std::vector<int32_t>& table = seq->get_mutable_block_table();
        table.push_back(new_block_id);
        seq_to_blocks_[seq->get_seq_id()] = table;
    }
}

void BlockSpaceManager::free(const SequencePtr& seq) {
    int64_t seq_id = seq->get_seq_id();
    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return;
    }

    const std::vector<int32_t>& blocks = it->second;
    for (int32_t block_id : blocks) {
        if (block_id >= 0 && block_id < num_blocks_) {
            ref_counts_[block_id]--;
            if (ref_counts_[block_id] <= 0) {
                ref_counts_[block_id] = 0;
                free_blocks_.push_back(block_id);
            }
        }
    }

    seq_to_blocks_.erase(it);
    seq->set_block_table({});
}

} // namespace engine
} // namespace lyapunov
