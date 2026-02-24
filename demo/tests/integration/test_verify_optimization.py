import pytest
import torch

from src.execution.continuous_batch_engine import CBExecutor
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.scheduling.cache_strategy import LRUCacheStrategy


def _make_expert_cache():
    return ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_verify_logits_match_prefill(qwen3_runner, dummy_parameter_loader, small_config):
    prefill_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    verify_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )

    prefill_exec = CBExecutor(
        model_runner=qwen3_runner,
        kv_cache=prefill_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        prefetcher=None,
        draft_scheduler=None,
        acceptance_strategy=None,
        max_draft_tokens=2,
    )
    verify_exec = CBExecutor(
        model_runner=qwen3_runner,
        kv_cache=verify_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        prefetcher=None,
        draft_scheduler=None,
        acceptance_strategy=None,
        max_draft_tokens=2,
    )

    seq_id = 0
    prompt = [1]
    draft_tokens = [2, 3]
    full_ids = torch.tensor(prompt + draft_tokens, device="cuda", dtype=torch.long)
    positions = torch.arange(len(full_ids), device="cuda", dtype=torch.long)

    prefill_cache.add_sequence(seq_id, prompt_len=len(full_ids))
    prefill_logits = prefill_exec._forward(full_ids, positions, [seq_id], is_prefill=True)
    if prefill_logits.dim() == 3:
        prefill_logits = prefill_logits[0]

    verify_cache.add_sequence(seq_id, prompt_len=len(prompt))
    verify_exec._forward(full_ids[:1], positions[:1], [seq_id], is_prefill=True)
    verify_cache.start_draft(seq_id)
    verify_cache.append_token(seq_id)
    verify_cache.append_token(seq_id)

    verify_input_ids = torch.tensor([prompt[-1]] + draft_tokens, device="cuda", dtype=torch.long)
    verify_positions = torch.arange(len(prompt) - 1, len(prompt) + len(draft_tokens), device="cuda", dtype=torch.long)
    verify_logits = verify_exec._forward_verify(verify_input_ids, verify_positions, [seq_id])
    if verify_logits.dim() == 3:
        verify_logits = verify_logits[0]

    expected = prefill_logits[len(prompt) - 1: len(prompt) - 1 + len(draft_tokens)]
    torch.testing.assert_close(verify_logits[: len(draft_tokens)], expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_replace_draft_with_verify_consistency(small_config):
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=4,
        dtype=small_config.get_dtype(),
    )
    seq_id = 0
    verify_seq_id = 1
    kv_cache.add_sequence(seq_id, prompt_len=4)
    kv_cache.add_sequence(verify_seq_id, prompt_len=4)
    kv_cache.start_draft(seq_id)
    kv_cache.append_token(seq_id)
    kv_cache.append_token(seq_id)
    kv_cache.append_token(seq_id)
    kv_cache.append_token(verify_seq_id)
    kv_cache.append_token(verify_seq_id)
    kv_cache.append_token(verify_seq_id)

    kv_cache.replace_draft_with_verify(seq_id, verify_seq_id, num_accepted_tokens=2)
    draft_state = kv_cache.sequences[seq_id]
    verify_state = kv_cache.sequences[verify_seq_id]
    assert len(draft_state.block_table) == len(verify_state.block_table)
    assert draft_state.num_tokens == draft_state.draft_start_num_tokens + 2
