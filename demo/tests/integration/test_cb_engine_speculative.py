import pytest
import torch

from src.core.types import InferenceRequest, GenerationConfig, InferenceMode
from src.execution.acceptance_strategy import StandardAcceptanceStrategy, AcceptanceStrategy
from src.execution.continuous_batch_engine import ContinuousBatchEngine, DecodeMode, Sequence
from src.execution.orchestrator import EnhancedInferenceOrchestrator
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.scheduling.cache_strategy import LRUCacheStrategy
from src.scheduling.draft_schduler import SimpleDraftScheduler


class _FixedBlocksKVCache(PagedKVCache):
    def __init__(self, config, block_size: int, num_blocks: int, dtype: torch.dtype):
        self._fixed_num_blocks = num_blocks
        super().__init__(config, block_size=block_size, gpu_memory_utilization=0.01, dtype=dtype)

    def _calculate_num_blocks(self, gpu_memory_utilization: float) -> int:
        return self._fixed_num_blocks


class _SelectiveAppendKVCache(PagedKVCache):
    def __init__(self, config, block_size: int, num_blocks: int, dtype: torch.dtype, blocked_ids: set[int]):
        self._fixed_num_blocks = num_blocks
        self._blocked_ids = blocked_ids
        super().__init__(config, block_size=block_size, gpu_memory_utilization=0.01, dtype=dtype)

    def _calculate_num_blocks(self, gpu_memory_utilization: float) -> int:
        return self._fixed_num_blocks

    def can_append_token(self, seq_id: int) -> bool:
        if seq_id in self._blocked_ids:
            return False
        return super().can_append_token(seq_id)


class _RejectThenAccept(AcceptanceStrategy):
    def __init__(self):
        self.calls = 0

    def accept(self, draft_token_ids: torch.Tensor, verify_logits: torch.Tensor, temperature: float = 1.0):
        self.calls += 1
        if self.calls == 1:
            return {
                "num_accepted": 0,
                "accepted_tokens": draft_token_ids[:0],
                "rejection_position": 0,
            }
        return {
            "num_accepted": len(draft_token_ids),
            "accepted_tokens": draft_token_ids,
            "rejection_position": -1,
        }


def _make_expert_cache():
    return ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_speculative_single_acceptance(qwen3_runner, dummy_parameter_loader, small_config):
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = ContinuousBatchEngine(
        model_runner=qwen3_runner,
        kv_cache=kv_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        decode_mode=DecodeMode.SPECULATIVE,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        max_draft_tokens=2,
    )
    torch.manual_seed(0)
    results = engine.generate([[1, 2, 3]], max_new_tokens=4, top_k=0, top_p=1.0)
    assert len(results[0]["output_token_ids"]) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_speculative_matches_orchestrator(qwen3_runner, dummy_parameter_loader, small_config):
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = ContinuousBatchEngine(
        model_runner=qwen3_runner,
        kv_cache=kv_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        decode_mode=DecodeMode.SPECULATIVE,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        max_draft_tokens=2,
    )

    orchestrator = EnhancedInferenceOrchestrator(
        config=small_config,
        parameter_loader=dummy_parameter_loader,
        expert_cache=_make_expert_cache(),
        prefetcher=None,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        model_runner=qwen3_runner,
        default_mode=InferenceMode.SPECULATIVE,
    )

    torch.manual_seed(0)
    cb_out = engine.generate([[5, 6, 7]], max_new_tokens=3, top_k=0, top_p=1.0)[0]["output_token_ids"]

    request = InferenceRequest(
        request_id="req",
        input_ids=torch.tensor([5, 6, 7], device="cuda", dtype=torch.long),
        max_new_tokens=3,
        generation_config=GenerationConfig(max_new_tokens=3, top_k=0, top_p=1.0, temperature=1.0, do_sample=True),
    )
    torch.manual_seed(0)
    orchestrator_out = orchestrator._generate_speculative(request).tolist()
    assert cb_out == orchestrator_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_speculative_multi_seq(qwen3_runner, dummy_parameter_loader, small_config):
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = ContinuousBatchEngine(
        model_runner=qwen3_runner,
        kv_cache=kv_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        decode_mode=DecodeMode.SPECULATIVE,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        max_draft_tokens=2,
    )
    torch.manual_seed(0)
    results = engine.generate([[1, 2], [3, 4, 5]], max_new_tokens=3, top_k=0, top_p=1.0)
    assert len(results) == 2
    assert all(len(r["output_token_ids"]) == 3 for r in results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_speculative_partial_error(qwen3_runner, dummy_parameter_loader, small_config):
    seq1 = Sequence([1, 2], max_new_tokens=2, top_k=0, top_p=1.0)
    seq2 = Sequence([3, 4, 5], max_new_tokens=2, top_k=0, top_p=1.0)
    kv_cache = _SelectiveAppendKVCache(
        config=small_config,
        block_size=4,
        num_blocks=4,
        dtype=small_config.get_dtype(),
        blocked_ids={seq2.seq_id},
    )
    engine = ContinuousBatchEngine(
        model_runner=qwen3_runner,
        kv_cache=kv_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        decode_mode=DecodeMode.SPECULATIVE,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=StandardAcceptanceStrategy(acceptance_threshold=0.0),
        max_draft_tokens=2,
    )
    engine.add_sequences([seq1, seq2])
    while not engine.scheduler.is_finished():
        engine.step()
    assert seq1.error_msg is None
    assert seq2.error_msg is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_speculative_full_rejection_then_continue(qwen3_runner, dummy_parameter_loader, small_config):
    kv_cache = PagedKVCache(
        config=small_config,
        block_size=256,
        dtype=small_config.get_dtype(),
    )
    engine = ContinuousBatchEngine(
        model_runner=qwen3_runner,
        kv_cache=kv_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        decode_mode=DecodeMode.SPECULATIVE,
        draft_scheduler=SimpleDraftScheduler(),
        acceptance_strategy=_RejectThenAccept(),
        max_draft_tokens=2,
    )
    torch.manual_seed(0)
    results = engine.generate([[9, 10]], max_new_tokens=2, top_k=0, top_p=1.0)
    assert len(results[0]["output_token_ids"]) == 2
