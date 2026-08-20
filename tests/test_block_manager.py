import pytest

try:
    from lyapunov_engine import _C
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


def test_block_manager_allocation():
    if not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not compiled")

    bm = _C.BlockSpaceManager(100, 16, True)
    assert bm.get_num_free_blocks() == 100
    assert bm.get_num_used_blocks() == 0

    sp = _C.SamplingParams()
    seq1 = _C.Sequence(1, [1] * 32, sp) # 32 tokens = 2 blocks

    assert bm.can_allocate(seq1)
    bm.allocate(seq1)

    assert bm.get_num_free_blocks() == 98
    assert bm.get_num_used_blocks() == 2
    assert len(seq1.get_block_table()) == 2

    # Appending slot without crossing boundary
    assert bm.can_append_slot(seq1)

    # Free sequence
    bm.free(seq1)
    assert bm.get_num_free_blocks() == 100
    assert bm.get_num_used_blocks() == 0
    assert len(seq1.get_block_table()) == 0


def test_scheduler_lifecycle():
    if not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not compiled")

    bm = _C.BlockSpaceManager(100, 16, True)
    cfg = _C.SchedulerConfig()
    cfg.max_num_seqs = 4
    cfg.max_num_batched_tokens = 512
    cfg.eos_token_id = 128001

    scheduler = _C.ContinuousScheduler(cfg, bm)
    assert not scheduler.has_unfinished_sequences()

    sp = _C.SamplingParams()
    sp.max_tokens = 3
    seq = _C.Sequence(1, [10, 20, 30], sp)

    scheduler.add_sequence(seq)
    assert scheduler.has_unfinished_sequences()
    assert scheduler.get_num_waiting_sequences() == 1

    # Schedule prefill
    out = scheduler.schedule()
    assert out.is_prefill
    assert len(out.scheduled_seqs) == 1
    assert scheduler.get_num_running_sequences() == 1

    # Step 1
    scheduler.post_step(out.scheduled_seqs, [100])
    assert not seq.is_finished()

    # Step 2: Decode
    out2 = scheduler.schedule()
    assert not out2.is_prefill
    scheduler.post_step(out2.scheduled_seqs, [101])
    assert not seq.is_finished()

    # Step 3: Decode (reaches max_tokens=3)
    out3 = scheduler.schedule()
    scheduler.post_step(out3.scheduled_seqs, [102])
    assert seq.is_finished()
    assert not scheduler.has_unfinished_sequences()
