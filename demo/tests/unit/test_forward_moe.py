import torch

from src.core.model_runner import AttentionOutput, ExpertPlacement, RoutingResult
from src.core.types import ExpertID, DeviceType


def _make_routing(layer_idx: int, topk_indices: torch.Tensor, topk_scores: torch.Tensor):
    activated = {ExpertID(layer_idx, idx) for idx in topk_indices.flatten().unique().tolist()}
    return RoutingResult(
        layer_idx=layer_idx,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        activated_expert_ids=activated,
    )


def test_forward_moe_output_shape(qwen3_runner, dummy_parameter_loader):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 2, config.hidden_size, device="cuda", dtype=config.get_dtype())
    attn_output = AttentionOutput(
        hidden_states=hidden,
        post_attn_normed=hidden,
        residual=hidden,
    )

    topk_indices = torch.tensor([[0, 1], [1, 0]], device="cuda")
    topk_scores = torch.full_like(topk_indices, 0.5, dtype=torch.float32)
    routing = _make_routing(0, topk_indices, topk_scores)

    placement = ExpertPlacement(
        gpu_expert_params={
            0: dummy_parameter_loader.get_expert_params(ExpertID(0, 0), device=DeviceType.GPU),
            1: dummy_parameter_loader.get_expert_params(ExpertID(0, 1), device=DeviceType.GPU),
        },
        cpu_expert_params={},
        routing_result=routing,
    )

    out = qwen3_runner.forward_moe(0, attn_output, placement)
    assert out.shape == hidden.shape
    assert out.abs().sum() > 0


def test_cpu_vs_gpu_execution_close(qwen3_runner, dummy_parameter_loader):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 2, config.hidden_size, device="cuda", dtype=config.get_dtype())
    attn_output = AttentionOutput(
        hidden_states=hidden,
        post_attn_normed=hidden,
        residual=hidden,
    )

    topk_indices = torch.tensor([[0, 0]], device="cuda")
    topk_scores = torch.ones_like(topk_indices, dtype=torch.float32)
    routing = _make_routing(0, topk_indices, topk_scores)

    gpu_params = dummy_parameter_loader.get_expert_params(ExpertID(0, 0), device=DeviceType.GPU)
    cpu_params = dummy_parameter_loader.get_expert_params(ExpertID(0, 0), device=DeviceType.CPU)

    gpu_placement = ExpertPlacement(gpu_expert_params={0: gpu_params}, cpu_expert_params={}, routing_result=routing)
    cpu_placement = ExpertPlacement(gpu_expert_params={}, cpu_expert_params={0: cpu_params}, routing_result=routing)

    gpu_out = qwen3_runner.forward_moe(0, attn_output, gpu_placement)
    cpu_out = qwen3_runner.forward_moe(0, attn_output, cpu_placement)

    assert torch.allclose(gpu_out, cpu_out, atol=1e-3)


def test_substitution_uses_substitute_params(qwen3_runner, dummy_parameter_loader):
    config = qwen3_runner.get_config()
    hidden = torch.randn(1, 2, config.hidden_size, device="cuda", dtype=config.get_dtype())
    attn_output = AttentionOutput(
        hidden_states=hidden,
        post_attn_normed=hidden,
        residual=hidden,
    )

    topk_indices = torch.tensor([[1, 1]], device="cuda")
    topk_scores = torch.ones_like(topk_indices, dtype=torch.float32)
    routing = _make_routing(0, topk_indices, topk_scores)

    placement = ExpertPlacement(
        gpu_expert_params={
            0: dummy_parameter_loader.get_expert_params(ExpertID(0, 0), device=DeviceType.GPU)
        },
        cpu_expert_params={},
        substitution_map={1: 0},
        routing_result=routing,
    )

    out = qwen3_runner.forward_moe(0, attn_output, placement)
    assert out.abs().sum() > 0
