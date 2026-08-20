#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <string>

namespace lyapunov {
namespace engine {

struct PrefixTreeNode {
    int physical_block_id;
    int ref_count;
    std::unordered_map<int32_t, std::shared_ptr<PrefixTreeNode>> children;

    PrefixTreeNode(int block_id = -1) : physical_block_id(block_id), ref_count(1) {}
};

class RadixPrefixTree {
public:
    RadixPrefixTree(int block_size = 16) : block_size_(block_size) {
        root_ = std::make_shared<PrefixTreeNode>(-1);
    }

    // Find cached physical block IDs matching token prefix
    std::vector<int> match_prefix(const std::vector<int32_t>& prompt_tokens) const {
        std::vector<int> matched_blocks;
        auto curr = root_;

        size_t num_full_blocks = prompt_tokens.size() / block_size_;
        for (size_t b = 0; b < num_full_blocks; ++b) {
            // Traverse block tokens
            bool matched_block = true;
            for (int i = 0; i < block_size_; ++i) {
                int32_t tok = prompt_tokens[b * block_size_ + i];
                auto it = curr->children.find(tok);
                if (it == curr->children.end()) {
                    matched_block = false;
                    break;
                }
                curr = it->second;
            }

            if (!matched_block || curr->physical_block_id < 0) {
                break;
            }
            matched_blocks.push_back(curr->physical_block_id);
        }

        return matched_blocks;
    }

    // Insert allocated block mapping for prefix reuse
    void insert_prefix(
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int>& physical_blocks
    ) {
        size_t num_blocks = std::min(prompt_tokens.size() / block_size_, physical_blocks.size());
        auto curr = root_;

        for (size_t b = 0; b < num_blocks; ++b) {
            for (int i = 0; i < block_size_; ++i) {
                int32_t tok = prompt_tokens[b * block_size_ + i];
                auto it = curr->children.find(tok);
                if (it == curr->children.end()) {
                    auto new_node = std::make_shared<PrefixTreeNode>(-1);
                    curr->children[tok] = new_node;
                    curr = new_node;
                } else {
                    curr = it->second;
                }
            }
            curr->physical_block_id = physical_blocks[b];
            curr->ref_count++;
        }
    }

private:
    int block_size_;
    std::shared_ptr<PrefixTreeNode> root_;
};

} // namespace engine
} // namespace lyapunov
