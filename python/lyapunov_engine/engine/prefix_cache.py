from typing import List, Dict, Optional, Tuple


class PrefixNode:
    def __init__(self, block_id: int = -1):
        self.block_id = block_id
        self.ref_count = 1
        self.children: Dict[int, "PrefixNode"] = {}


class RadixPrefixCache:
    """Radix-Tree based prefix cache for zero-copy KV cache sharing."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = PrefixNode(-1)
        self.cached_blocks_count = 0

    def match_prefix(self, prompt_tokens: List[int]) -> List[int]:
        """Find cached physical block IDs matching the prompt prefix."""
        matched_blocks = []
        curr = self.root
        num_full_blocks = len(prompt_tokens) // self.block_size

        for b in range(num_full_blocks):
            matched = True
            for i in range(self.block_size):
                tok = prompt_tokens[b * self.block_size + i]
                if tok not in curr.children:
                    matched = False
                    break
                curr = curr.children[tok]

            if not matched or curr.block_id < 0:
                break
            matched_blocks.append(curr.block_id)

        return matched_blocks

    def insert_prefix(self, prompt_tokens: List[int], block_ids: List[int]):
        """Cache block mappings for future multi-turn / system prompt reuse."""
        num_blocks = min(len(prompt_tokens) // self.block_size, len(block_ids))
        curr = self.root

        for b in range(num_blocks):
            for i in range(self.block_size):
                tok = prompt_tokens[b * self.block_size + i]
                if tok not in curr.children:
                    curr.children[tok] = PrefixNode(-1)
                curr = curr.children[tok]
            if curr.block_id == -1:
                self.cached_blocks_count += 1
            curr.block_id = block_ids[b]
            curr.ref_count += 1
