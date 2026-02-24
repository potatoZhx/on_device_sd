import pytest
import torch

from src.core.types import BatchedRequest, InferenceRequest, GenerationConfig
from src.execution.continuous_batch_engine import ContinuousBatchEngine
from src.execution.standard_engine import StandardDecodeEngine
from src.memory.expert_cache import ExpertCache
from src.memory.paged_kv_cache import PagedKVCache
from src.scheduling.cache_strategy import LRUCacheStrategy


class _FixedBlocksKVCache(PagedKVCache):
    def __init__(self, config, block_size: int, num_blocks: int, dtype: torch.dtype):
        self._fixed_num_blocks = num_blocks
        super().__init__(config, block_size=block_size, gpu_memory_utilization=0.01, dtype=dtype)

    def _calculate_num_blocks(self, gpu_memory_utilization: float) -> int:
        return self._fixed_num_blocks


def _make_expert_cache():
    return ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_standard_single_and_eos(qwen3_runner, dummy_parameter_loader, small_config):
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
    )

    torch.manual_seed(0)
    result = engine.generate([[1, 2, 3]], max_new_tokens=1, top_k=1, top_p=1.0)
    first_token = result[0]["output_token_ids"][0]

    result_eos = engine.generate([[1, 2, 3]], max_new_tokens=5, top_k=1, top_p=1.0, eos_token_id=first_token)
    assert len(result_eos[0]["output_token_ids"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_standard_multi_seq_batch(qwen3_runner, dummy_parameter_loader, small_config):
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
    )
    torch.manual_seed(0)
    results = engine.generate([[1, 2], [3, 4, 5, 6]], max_new_tokens=3, top_k=1, top_p=1.0)
    assert len(results) == 2
    assert len(results[0]["output_token_ids"]) == 3
    assert len(results[1]["output_token_ids"]) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_standard_matches_standard_engine(qwen3_runner, dummy_parameter_loader, small_config):
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
    )
    standard_engine = StandardDecodeEngine(
        model_runner=qwen3_runner,
        parameter_loader=dummy_parameter_loader,
        expert_cache=_make_expert_cache(),
    )

    torch.manual_seed(0)
    prompt = [7, 8, 9]
    cb_out = engine.generate([prompt], max_new_tokens=4, top_k=1, top_p=1.0)[0]["output_token_ids"]

    gen_config = GenerationConfig(max_new_tokens=4, top_k=1, top_p=1.0, temperature=1.0, do_sample=True)
    request = InferenceRequest(
        request_id="req",
        input_ids=torch.tensor(prompt, device="cuda", dtype=torch.long),
        max_new_tokens=4,
        generation_config=gen_config,
    )
    batch = BatchedRequest(
        batch_id="batch",
        requests=[request],
        input_ids=request.input_ids.unsqueeze(0),
        attention_mask=torch.ones(1, len(prompt), device="cuda", dtype=torch.long),
        position_ids=torch.arange(len(prompt), device="cuda", dtype=torch.long).unsqueeze(0),
        current_lengths=[len(prompt)],
        finished=[False],
        max_batch_seq_len=len(prompt),
    )
    standard_out = standard_engine.generate_batch(batch)["generated_sequences"][0].tolist()
    assert cb_out == standard_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_standard_handles_kv_cache_exhaustion(qwen3_runner, dummy_parameter_loader, small_config):
    kv_cache = _FixedBlocksKVCache(
        config=small_config,
        block_size=4,
        num_blocks=2,
        dtype=small_config.get_dtype(),
    )
    engine = ContinuousBatchEngine(
        model_runner=qwen3_runner,
        kv_cache=kv_cache,
        expert_cache=_make_expert_cache(),
        parameter_loader=dummy_parameter_loader,
        max_num_batched_tokens=16,
    )
    torch.manual_seed(0)
    results = engine.generate([[1, 2, 3], [4, 5, 6, 7]], max_new_tokens=1, top_k=1, top_p=1.0)
    errors = [r["error"] for r in results]
    assert any(err is None for err in errors)
    assert any(err is not None for err in errors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_engine_standard_respects_max_num_seqs(qwen3_runner, dummy_parameter_loader, small_config):
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
        max_num_seqs=1,
    )
    torch.manual_seed(0)
    results = engine.generate([[1, 2], [3, 4]], max_new_tokens=2, top_k=1, top_p=1.0)
    assert len(results) == 2
    assert all(len(res["output_token_ids"]) == 2 for res in results)
