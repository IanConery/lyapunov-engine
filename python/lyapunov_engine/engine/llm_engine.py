from typing import List, Optional, Dict, Union
import torch
import torch.nn as nn

try:
    from lyapunov_engine import _C
    HAS_CUDA_EXT = True
except ImportError:
    _C = None
    HAS_CUDA_EXT = False

from lyapunov_engine.engine.sampling import SamplingParams, sample_next_tokens
from lyapunov_engine.models.llama import LlamaForCausalLM, LlamaConfig


class EngineConfig:
    """Runtime configuration parameters for LLMEngine."""
    def __init__(
        self,
        model_config=None,
        num_blocks: int = 1024,
        num_gpu_blocks: Optional[int] = None,
        block_size: int = 16,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 2048,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda:0"
    ):
        self.model_config = model_config
        self.num_blocks = num_gpu_blocks if num_gpu_blocks is not None else num_blocks
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.dtype = dtype
        self.device = device


class RequestOutput:
    def __init__(self, request_id: Union[int, str], prompt_tokens: List[int], output_tokens: List[int], finished: bool, finish_reason: Optional[str] = "stop"):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.finished = finished
        self.is_finished = finished
        self.finish_reason = finish_reason


class PythonSequence:
    def __init__(self, seq_id: int, prompt_tokens: List[int], sampling_params: SamplingParams):
        self.seq_id = seq_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = []
        self.sampling_params = sampling_params
        self.block_table = []
        self.status = "WAITING"

    def get_seq_id(self) -> int:
        return self.seq_id

    def get_prompt_tokens(self) -> List[int]:
        return self.prompt_tokens

    def get_output_tokens(self) -> List[int]:
        return self.output_tokens

    def get_total_len(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    def get_last_token_id(self) -> int:
        if self.output_tokens:
            return self.output_tokens[-1]
        return self.prompt_tokens[-1]

    def get_block_table(self) -> List[int]:
        return self.block_table

    def is_finished(self) -> bool:
        return self.status.startswith("FINISHED")


class LLMEngine:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EngineConfig] = None,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 2048,
        device: str = "cuda:0"
    ):
        if config is not None:
            self.device = torch.device(config.device)
            self.block_size = config.block_size
            self.num_blocks = config.num_blocks
            self.max_num_seqs = config.max_num_seqs
            self.max_num_batched_tokens = config.max_num_batched_tokens
            self.dtype = config.dtype
        else:
            self.device = torch.device(device)
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.max_num_seqs = max_num_seqs
            self.max_num_batched_tokens = max_num_batched_tokens
            self.dtype = torch.float16

        self.model = model.to(self.device).eval()

        # Initialize C++ BlockManager and Scheduler if available, else Python fallback
        if HAS_CUDA_EXT:
            self.block_manager = _C.BlockSpaceManager(self.num_blocks, self.block_size, True)
            sched_cfg = _C.SchedulerConfig()
            sched_cfg.max_num_seqs = self.max_num_seqs
            sched_cfg.max_num_batched_tokens = self.max_num_batched_tokens
            self.scheduler = _C.ContinuousScheduler(sched_cfg, self.block_manager)
            self.use_cpp_runtime = True
        else:
            self.use_cpp_runtime = False
            self.waiting_queue: List[PythonSequence] = []
            self.running_queue: List[PythonSequence] = []
            self.free_blocks = list(range(self.num_blocks))

        # Allocate KV Cache memory pools on GPU
        model_config = getattr(model, "config", LlamaConfig())
        num_layers = model_config.num_hidden_layers
        num_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.head_dim

        self.k_caches = [
            torch.zeros((self.num_blocks, num_kv_heads, self.block_size, head_dim), dtype=self.dtype, device=self.device)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros((self.num_blocks, num_kv_heads, self.block_size, head_dim), dtype=self.dtype, device=self.device)
            for _ in range(num_layers)
        ]

        self.req_counter = 0
        self.active_sequences: Dict[Union[int, str], object] = {}

    def add_request(
        self,
        arg1: Union[int, str, List[int]],
        arg2: Optional[Union[List[int], SamplingParams]] = None,
        arg3: Optional[SamplingParams] = None
    ) -> Union[int, str]:
        """Add a request. Supports:
        - add_request(prompt_tokens, sampling_params)
        - add_request(request_id, prompt_tokens, sampling_params)
        """
        if isinstance(arg1, (int, str)) and isinstance(arg2, list):
            request_id = arg1
            prompt_tokens = arg2
            sampling_params = arg3 or SamplingParams()
        else:
            request_id = self.req_counter
            self.req_counter += 1
            prompt_tokens = arg1
            sampling_params = arg2 or SamplingParams()

        seq_id = self.req_counter
        self.req_counter += 1

        if self.use_cpp_runtime:
            c_sampling = _C.SamplingParams()
            c_sampling.temperature = sampling_params.temperature
            c_sampling.top_p = sampling_params.top_p
            c_sampling.top_k = sampling_params.top_k
            c_sampling.max_tokens = sampling_params.max_tokens
            c_sampling.ignore_eos = sampling_params.ignore_eos

            seq = _C.Sequence(seq_id, prompt_tokens, c_sampling)
            self.scheduler.add_sequence(seq)
            self.active_sequences[seq_id] = (seq, sampling_params, request_id)
        else:
            seq = PythonSequence(seq_id, prompt_tokens, sampling_params)
            self.waiting_queue.append(seq)
            self.active_sequences[seq_id] = (seq, sampling_params, request_id)

        return request_id

    def has_unfinished_requests(self) -> bool:
        if self.use_cpp_runtime:
            return self.scheduler.has_unfinished_sequences()
        return len(self.waiting_queue) > 0 or len(self.running_queue) > 0

    def step(self) -> List[RequestOutput]:
        if not self.has_unfinished_requests():
            return []

        if self.use_cpp_runtime:
            sched_outputs = self.scheduler.schedule()
            scheduled_seqs = sched_outputs.scheduled_seqs
            if not scheduled_seqs:
                return []

            is_prefill = sched_outputs.is_prefill
            batch_size = len(scheduled_seqs)

            # Build metadata tensors
            context_lens_list = [s.get_total_len() for s in scheduled_seqs]
            max_blocks = max(len(s.get_block_table()) for s in scheduled_seqs) if scheduled_seqs else 1
            if max_blocks == 0:
                max_blocks = 1

            block_tables_tensor = torch.zeros((batch_size, max_blocks), dtype=torch.int32, device=self.device)
            for i, s in enumerate(scheduled_seqs):
                tbl = s.get_block_table()
                if tbl:
                    block_tables_tensor[i, :len(tbl)] = torch.tensor(tbl, dtype=torch.int32, device=self.device)

            context_lens_tensor = torch.tensor(context_lens_list, dtype=torch.int32, device=self.device)

            if is_prefill:
                logits_list = []
                for i, seq in enumerate(scheduled_seqs):
                    tokens = torch.tensor([seq.get_prompt_tokens()], dtype=torch.long, device=self.device)
                    b_table = block_tables_tensor[i : i + 1]
                    c_lens = context_lens_tensor[i : i + 1]
                    with torch.no_grad():
                        l = self.model(
                            tokens,
                            k_caches=self.k_caches,
                            v_caches=self.v_caches,
                            block_tables=b_table,
                            context_lens=c_lens,
                            is_prefill=True
                        )[:, -1, :]
                        logits_list.append(l)
                logits = torch.cat(logits_list, dim=0)
            else:
                tokens = torch.tensor([[s.get_last_token_id()] for s in scheduled_seqs], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.model(
                        tokens,
                        k_caches=self.k_caches,
                        v_caches=self.v_caches,
                        block_tables=block_tables_tensor,
                        context_lens=context_lens_tensor,
                        is_prefill=False
                    )[:, -1, :]

            sp_list = [self.active_sequences[s.get_seq_id()][1] for s in scheduled_seqs]
            next_tokens = sample_next_tokens(logits, sp_list)
            self.scheduler.post_step(scheduled_seqs, next_tokens)

            outputs = []
            for s in scheduled_seqs:
                req_id = self.active_sequences[s.get_seq_id()][2]
                outputs.append(RequestOutput(
                    request_id=req_id,
                    prompt_tokens=s.get_prompt_tokens(),
                    output_tokens=s.get_output_tokens(),
                    finished=s.is_finished(),
                    finish_reason="stop" if s.is_finished() else None
                ))
            return outputs
        else:
            return []
