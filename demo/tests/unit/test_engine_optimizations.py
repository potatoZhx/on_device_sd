import pytest
import torch

from src.core.model import MoEConfig
from src.core.model_runner import RoutingResult, ExpertPlacement
from src.core.types import ExpertID, ExpertActivation, LayerExpertActivations
from src.execution.continuous_batch_engine import Sequence, SequenceStatus, CBScheduler, select_experts_to_prefetch
from src.layers.attention import Qwen3Attention, FLASH_ATTN_AVAILABLE
from src.memory.expert_cache import ExpertCache, AsyncExpertTransfer
from src.memory.paged_kv_cache import PagedKVCache
from src.model.qwen3_runner import Qwen3ModelRunner
from src.scheduling.cache_strategy import LRUCacheStrategy


def _make_config() -> MoEConfig:
    return MoEConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=16,
        vocab_size=32,
        num_experts=4,
        num_experts_per_token=1,
        moe_intermediate_size=16,
        max_position_embeddings=32,
        torch_dtype="float16",
    )


def _make_kv_cache() -> PagedKVCache:
    config = _make_config()
    return PagedKVCache(config, block_size=256, gpu_memory_utilization=0.01, dtype=torch.float16)


def _make_params(hidden_size: int, intermediate_size: int, device: str) -> dict:
    return {
        "gate_proj": torch.randn(intermediate_size, hidden_size, device=device),
        "up_proj": torch.randn(intermediate_size, hidden_size, device=device),
        "down_proj": torch.randn(hidden_size, intermediate_size, device=device),
    }


def _sequential_execute(hidden_states: torch.Tensor, placement: ExpertPlacement) -> torch.Tensor:
    routing = placement.routing_result
    if routing is None:
        return torch.zeros_like(hidden_states)

    if hidden_states.dim() == 3:
        b, s, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        need_reshape = True
    else:
        flat = hidden_states
        h = flat.shape[-1]
        need_reshape = False

    topk_indices = routing.topk_indices
    topk_scores = routing.topk_scores
    sub_map = placement.substitution_map
    final_output = torch.zeros(flat.shape[0], h, device=flat.device, dtype=flat.dtype)

    def _prepare_params(params: dict, device: torch.device, dtype: torch.dtype) -> dict:
        if params["gate_proj"].device == device and params["gate_proj"].dtype == dtype:
            return params
        return {k: v.to(device=device, dtype=dtype) for k, v in params.items()}

    for expert_id in routing.activated_expert_ids:
        expert_idx = expert_id.expert_idx
        expert_mask = (topk_indices == expert_idx)
        token_expert_pairs = torch.where(expert_mask)
        token_indices = token_expert_pairs[0]
        k_indices = token_expert_pairs[1]

        if len(token_indices) == 0:
            continue

        weights = topk_scores[token_indices, k_indices]
        expert_input = flat[token_indices]

        if expert_idx in placement.gpu_expert_params:
            params = _prepare_params(placement.gpu_expert_params[expert_idx], flat.device, flat.dtype)
            expert_output = Qwen3ModelRunner._execute_moe_with_placement.__globals__["expert_forward_with_weights"](
                expert_input,
                params["gate_proj"], params["up_proj"], params["down_proj"],
            )
            if expert_output.dtype != flat.dtype:
                expert_output = expert_output.to(flat.dtype)
        elif expert_idx in placement.cpu_expert_params:
            params = placement.cpu_expert_params[expert_idx]
            expert_input_cpu = expert_input.to("cpu")
            if expert_input_cpu.dtype != params["gate_proj"].dtype:
                expert_input_cpu = expert_input_cpu.to(params["gate_proj"].dtype)
            expert_output_cpu = Qwen3ModelRunner._execute_moe_with_placement.__globals__["expert_forward_with_weights"](
                expert_input_cpu,
                params["gate_proj"], params["up_proj"], params["down_proj"],
            )
            expert_output = expert_output_cpu.to(flat.device, dtype=flat.dtype)
        elif expert_idx in sub_map:
            sub_idx = sub_map[expert_idx]
            if sub_idx in placement.gpu_expert_params:
                params = _prepare_params(placement.gpu_expert_params[sub_idx], flat.device, flat.dtype)
                expert_output = Qwen3ModelRunner._execute_moe_with_placement.__globals__["expert_forward_with_weights"](
                    expert_input,
                    params["gate_proj"], params["up_proj"], params["down_proj"],
                )
                if expert_output.dtype != flat.dtype:
                    expert_output = expert_output.to(flat.dtype)
            elif sub_idx in placement.cpu_expert_params:
                params = placement.cpu_expert_params[sub_idx]
                expert_input_cpu = expert_input.to("cpu")
                if expert_input_cpu.dtype != params["gate_proj"].dtype:
                    expert_input_cpu = expert_input_cpu.to(params["gate_proj"].dtype)
                expert_output_cpu = Qwen3ModelRunner._execute_moe_with_placement.__globals__["expert_forward_with_weights"](
                    expert_input_cpu,
                    params["gate_proj"], params["up_proj"], params["down_proj"],
                )
                expert_output = expert_output_cpu.to(flat.device, dtype=flat.dtype)
            else:
                continue
        else:
            continue

        final_output[token_indices] += expert_output * weights.unsqueeze(1)

    if need_reshape:
        final_output = final_output.view(b, s, h)
    return final_output


def test_sequence_state_transitions():
    seq = Sequence([1, 2], max_new_tokens=2, eos_token_id=3)
    assert seq.status == SequenceStatus.WAITING
    assert seq.prompt_len == 2
    seq.append_token(5)
    assert not seq.check_finished()
    seq.append_token(3)
    assert seq.check_finished()
    assert seq.status == SequenceStatus.FINISHED

    seq = Sequence([1], max_new_tokens=4, eos_token_id=3)
    seq.start_draft()
    seq.append_draft_token(7)
    seq.accept_draft(0)
    assert seq.output_token_ids == []
    assert seq.status == SequenceStatus.RUNNING

    seq.start_draft()
    seq.append_draft_token(9)
    seq.accept_draft(1)
    assert seq.output_token_ids == [9]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cb_scheduler_prefill_and_decode():
    kv_cache = _make_kv_cache()
    scheduler = CBScheduler(kv_cache, max_num_seqs=2, max_num_batched_tokens=6)
    seq1 = Sequence([1, 2, 3])
    seq2 = Sequence([4, 5, 6])
    seq3 = Sequence([7, 8, 9])
    scheduler.add(seq1)
    scheduler.add(seq2)
    scheduler.add(seq3)

    result = scheduler.schedule()
    assert result.is_prefill is True
    assert len(result.sequences) == 2

    result = scheduler.schedule()
    assert result.is_prefill is False
    assert len(result.sequences) == 2

    seq1.status = SequenceStatus.FINISHED
    scheduler.postprocess([seq1], [seq1.seq_id])
    assert seq1.seq_id not in kv_cache.sequences


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_paged_kv_cache_verify_context_and_accept_draft():
    kv_cache = _make_kv_cache()
    seq_id = 0
    kv_cache.add_sequence(seq_id, prompt_len=3)
    kv_cache.start_draft(seq_id)
    kv_cache.append_token(seq_id)
    kv_cache.append_token(seq_id)

    ctx = kv_cache.get_verify_context([seq_id])
    seq_state = kv_cache.sequences[seq_id]
    expected = seq_state.get_slot_mapping(seq_state.draft_start_num_tokens, 2)
    assert ctx["slot_mapping"].tolist() == expected
    assert ctx["context_lens"].tolist() == [seq_state.num_tokens]

    kv_cache.accept_draft(seq_id, 1)
    assert kv_cache.sequences[seq_id].num_tokens == 4


@pytest.mark.skipif(
    not torch.cuda.is_available() or not FLASH_ATTN_AVAILABLE,
    reason="CUDA or flash_attn not available",
)
def test_attention_verify_matches_prefill_for_draft_tokens():
    torch.manual_seed(0)
    config = _make_config()
    attn = Qwen3Attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        layer_idx=0,
    ).cuda().half()

    hidden_states = torch.randn(3, config.hidden_size, device="cuda", dtype=torch.float16)
    positions = torch.arange(3, device="cuda", dtype=torch.long)

    kv_cache_prefill = _make_kv_cache()
    kv_cache_prefill.add_sequence(0, prompt_len=3)
    prefill_out = attn(hidden_states, positions, kv_cache_prefill, [0], is_prefill=True, is_verify=False)

    kv_cache_verify = _make_kv_cache()
    kv_cache_verify.add_sequence(0, prompt_len=1)
    attn(hidden_states[:1], positions[:1], kv_cache_verify, [0], is_prefill=True, is_verify=False)
    kv_cache_verify.start_draft(0)
    kv_cache_verify.append_token(0)
    kv_cache_verify.append_token(0)

    verify_out = attn(hidden_states, positions, kv_cache_verify, [0], is_prefill=False, is_verify=True)
    torch.testing.assert_close(verify_out[-2:], prefill_out[-2:], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_async_expert_transfer():
    cache = ExpertCache(
        max_cache_size_gb=0.01,
        expert_size_mb=0.001,
        replacement_strategy=LRUCacheStrategy(),
    )
    transfer = AsyncExpertTransfer(max_concurrent=1)
    expert_id = ExpertID(0, 0)
    cpu_params = _make_params(4, 8, "cpu")
    transfer.start_transfer(expert_id, cpu_params, cache)
    transfer.wait_all(cache)
    assert cache.is_cached(expert_id)
    cached = cache.get(expert_id)
    assert cached is not None
    assert all(param.is_cuda for param in cached.values())


def test_select_experts_to_prefetch():
    activations = [
        ExpertActivation(
            expert_id=ExpertID(0, i),
            token_indices=torch.tensor([0]),
            scores=torch.tensor([1.0 - i * 0.1]),
            top_k_rank=0,
        )
        for i in range(4)
    ]
    layer_acts = LayerExpertActivations(
        layer_idx=0,
        activations=activations,
        routing_scores=torch.randn(1, 4),
    )
    result = select_experts_to_prefetch(
        current_step=0,
        max_draft_tokens=8,
        step_activations=[layer_acts],
        cached_experts={ExpertID(0, 0)},
        pending_transfers=set(),
        cache_capacity=3,
        num_experts_per_layer=4,
        max_prefetch_per_step=2,
    )
    assert ExpertID(0, 0) not in result
    assert len(result) == 2
    assert result[0] == ExpertID(0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_parallel_execution_matches_sequential():
    torch.manual_seed(0)
    hidden_size = 4
    intermediate = 8
    hidden_states = torch.randn(2, hidden_size, device="cuda", dtype=torch.float16)

    topk_indices = torch.tensor([[0], [1]], device="cuda")
    topk_scores = torch.tensor([[1.0], [1.0]], device="cuda", dtype=torch.float16)
    activated = {ExpertID(0, 0), ExpertID(0, 1)}
    routing = RoutingResult(
        layer_idx=0,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        activated_expert_ids=activated,
    )

    gpu_params = _make_params(hidden_size, intermediate, "cuda")
    cpu_params = _make_params(hidden_size, intermediate, "cpu")

    placement = ExpertPlacement(
        gpu_expert_params={0: gpu_params},
        cpu_expert_params={1: cpu_params},
        routing_result=routing,
    )

    runner = Qwen3ModelRunner.__new__(Qwen3ModelRunner)
    parallel_out = runner._execute_moe_with_placement(hidden_states, placement)
    sequential_out = _sequential_execute(hidden_states, placement)
    torch.testing.assert_close(parallel_out, sequential_out, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_parallel_execution_with_substitution():
    torch.manual_seed(0)
    hidden_size = 4
    intermediate = 8
    hidden_states = torch.randn(2, hidden_size, device="cuda", dtype=torch.float16)

    topk_indices = torch.tensor([[2], [2]], device="cuda")
    topk_scores = torch.tensor([[1.0], [1.0]], device="cuda", dtype=torch.float16)
    activated = {ExpertID(0, 2)}
    routing = RoutingResult(
        layer_idx=0,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        activated_expert_ids=activated,
    )

    gpu_params = _make_params(hidden_size, intermediate, "cuda")

    placement = ExpertPlacement(
        gpu_expert_params={0: gpu_params},
        cpu_expert_params={},
        substitution_map={2: 0},
        routing_result=routing,
    )

    runner = Qwen3ModelRunner.__new__(Qwen3ModelRunner)
    parallel_out = runner._execute_moe_with_placement(hidden_states, placement)
    sequential_out = _sequential_execute(hidden_states, placement)
    torch.testing.assert_close(parallel_out, sequential_out, rtol=1e-3, atol=1e-3)
