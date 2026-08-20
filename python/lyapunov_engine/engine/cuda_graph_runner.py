from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn


class CUDAGraphRunner:
    """Manages static CUDA Graph captures across discrete batch powers for zero-overhead decoding."""

    def __init__(
        self,
        model: nn.Module,
        batch_buckets: List[int] = [1, 2, 4, 8, 16, 32, 64],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16
    ):
        self.model = model.eval()
        self.batch_buckets = sorted(batch_buckets)
        self.device = torch.device(device)
        self.dtype = dtype

        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_outputs: Dict[int, torch.Tensor] = {}

    def capture_graphs(self, max_bucket: int = 64):
        """Capture CUDA graphs for all supported batch buckets up to max_bucket."""
        if not torch.cuda.is_available():
            return

        for b in self.batch_buckets:
            if b > max_bucket:
                break

            # Create static dummy inputs
            static_tokens = torch.zeros((b, 1), dtype=torch.long, device=self.device)

            # Warmup runs
            s = torch.cuda.Stream(device=self.device)
            s.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = self.model(static_tokens)
            torch.cuda.current_stream(device=self.device).wait_stream(s)

            # Graph Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                static_out = self.model(static_tokens)

            self.graphs[b] = graph
            self.static_inputs[b] = static_tokens
            self.static_outputs[b] = static_out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic batch bucketing and graph replay."""
        batch_size = tokens.size(0)

        # Find smallest bucket >= batch_size
        bucket = None
        for b in self.batch_buckets:
            if b >= batch_size and b in self.graphs:
                bucket = b
                break

        if bucket is None or not tokens.is_cuda:
            # Fallback to standard eager execution
            return self.model(tokens)

        # Copy into static memory buffer and pad if necessary
        static_in = self.static_inputs[bucket]
        static_in[:batch_size].copy_(tokens)
        if batch_size < bucket:
            static_in[batch_size:].zero_()

        # Replay graph
        self.graphs[bucket].replay()

        # Slice and return valid output tokens
        return self.static_outputs[bucket][:batch_size]
