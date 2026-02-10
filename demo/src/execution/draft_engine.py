from typing import Dict, List
import torch

from ..core.model_runner import ModelRunner
from ..core.model import MoEConfig
from ..core.types import DraftMetrics
from ..memory.expert_cache import ExpertCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.draft_schduler import DraftSchedulingStrategy
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector
from .model_runner_utils import build_layer_activations, build_draft_placement

logger = get_logger(__name__)


class DraftEngine:
    """
    Draft phase execution engine using ModelRunner.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        draft_scheduler: DraftSchedulingStrategy,
        metrics: MetricsCollector,
    ):
        self.model_runner = model_runner
        self.config: MoEConfig = model_runner.get_config()
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.draft_scheduler = draft_scheduler
        self.metrics = metrics

        self.draft_activations: List = []
        self.cache_hits = 0
        self.cache_misses = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache,
        max_draft_tokens: int,
        temperature: float = 1.0,
        seq_ids: List[int] | None = None,
    ) -> Dict:
        self.metrics.start_phase('draft')

        drafted_tokens: List[int] = []
        current_token = input_ids.cuda()

        self.cache_hits = 0
        self.cache_misses = 0
        self.draft_activations = []

        for step in range(max_draft_tokens):
            logger.debug(f"Draft step {step + 1}/{max_draft_tokens}")
            output = self._draft_forward_pass(
                input_ids=current_token,
                kv_cache=kv_cache,
                seq_ids=seq_ids,
            )

            logits = output['logits'][:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            drafted_tokens.append(next_token.item())
            current_token = next_token

        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        perplexity = self._calculate_perplexity(output['logits'])

        metrics = DraftMetrics(
            num_drafted_tokens=len(drafted_tokens),
            perplexity=perplexity,
            expert_hit_rate=cache_hit_rate,
            cpu_compute_ratio=self.cache_misses / max(self.cache_hits + self.cache_misses, 1),
        )

        self._schedule_expert_transfers()

        self.metrics.end_phase('draft')

        logger.info(
            f"Drafted {len(drafted_tokens)} tokens, "
            f"cache hit rate: {cache_hit_rate:.2%}, "
            f"perplexity: {perplexity:.3f}"
        )

        return {
            'drafted_tokens': drafted_tokens,
            'metrics': metrics,
            'activations': self.draft_activations,
        }

    def _draft_forward_pass(self, input_ids: torch.Tensor, kv_cache, seq_ids: List[int] | None) -> Dict:
        if seq_ids is None:
            seq_ids = [0]

        if hasattr(kv_cache, "sequences") and hasattr(kv_cache, "append_token"):
            for seq_id in seq_ids:
                if seq_id not in kv_cache.sequences:
                    kv_cache.add_sequence(seq_id, prompt_len=1)
                else:
                    kv_cache.append_token(seq_id)

        hidden_states = self.model_runner.embed(input_ids)

        for layer_idx in range(self.model_runner.get_num_layers()):
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                positions=None,
                seq_ids=seq_ids,
                is_prefill=False,
            )

            routing_result = self.model_runner.route_experts(
                layer_idx=layer_idx,
                hidden_states=attn_output.post_attn_normed,
            )

            placement = build_draft_placement(
                routing_result=routing_result,
                expert_cache=self.expert_cache,
                parameter_loader=self.parameter_loader,
                draft_scheduler=self.draft_scheduler,
                top_c=self.config.draft_top_c,
                num_experts=self.config.num_experts,
            )

            self.cache_hits += len(placement.gpu_expert_params)
            self.cache_misses += len(placement.cpu_expert_params)

            layer_acts = build_layer_activations(routing_result, self.config.num_experts)
            self.draft_activations.append(layer_acts)

            hidden_states = self.model_runner.forward_moe(
                layer_idx=layer_idx,
                attn_output=attn_output,
                expert_placement=placement,
            )

        logits = self.model_runner.compute_logits(hidden_states)
        return {'logits': logits, 'hidden_states': hidden_states}

    def _schedule_expert_transfers(self) -> None:
        if not self.draft_activations:
            return

        cached_experts = set(self.expert_cache.cached_experts.keys())
        experts_to_transfer = self.draft_scheduler.select_experts_to_transfer(
            recent_activations=self.draft_activations,
            cached_experts=cached_experts,
            cache_capacity=self.expert_cache.max_experts,
        )

        if experts_to_transfer:
            logger.info(f"Scheduling {len(experts_to_transfer)} expert transfers")
            for expert_id in experts_to_transfer:
                cpu_params = self.parameter_loader.get_expert_params(expert_id)
                if cpu_params:
                    self.expert_cache.put(expert_id, cpu_params)

    def _calculate_perplexity(self, logits: torch.Tensor) -> float:
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=-1))
        return torch.exp(entropy).item()
