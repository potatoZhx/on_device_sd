import torch

from src.core.model_runner import ModelRunner, RoutingResult, ExpertPlacement, AttentionOutput
from src.core.model import MoEConfig
from src.execution.prefill_engine import PrefillEngine
from src.utils.metrics import MetricsCollector


class MockModelRunner(ModelRunner):
    def __init__(self):
        self._config = MoEConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=16,
            vocab_size=32,
            num_experts=2,
            num_experts_per_token=1,
            num_shared_experts=0,
            moe_intermediate_size=8,
            max_position_embeddings=16,
            rope_theta=10000.0,
            torch_dtype="float16",
            model_type="mock",
        )

    def get_config(self) -> MoEConfig:
        return self._config

    def get_num_layers(self) -> int:
        return 1

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_ids.shape + (self._config.hidden_size,), device=input_ids.device)

    def route_experts(self, layer_idx: int, hidden_states: torch.Tensor) -> RoutingResult:
        topk_indices = torch.zeros((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.long)
        topk_scores = torch.ones_like(topk_indices, dtype=torch.float32)
        return RoutingResult(layer_idx=layer_idx, topk_indices=topk_indices, topk_scores=topk_scores, activated_expert_ids=set())

    def forward_attention(self, layer_idx: int, hidden_states: torch.Tensor, kv_cache, positions: torch.Tensor, *, seq_ids=None, is_prefill=False) -> AttentionOutput:
        return AttentionOutput(hidden_states=hidden_states, post_attn_normed=hidden_states, residual=hidden_states)

    def forward_moe(self, layer_idx: int, attn_output: AttentionOutput, expert_placement: ExpertPlacement) -> torch.Tensor:
        return attn_output.hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden_states.shape[0], hidden_states.shape[1], self._config.vocab_size, device=hidden_states.device)


def test_engine_accepts_any_model_runner():
    engine = PrefillEngine(
        model_runner=MockModelRunner(),
        parameter_loader=None,
        expert_cache=None,
        prefetcher=None,
        metrics=MetricsCollector(),
    )
    assert engine.model_runner is not None
