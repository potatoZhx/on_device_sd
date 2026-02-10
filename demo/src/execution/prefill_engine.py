from typing import Dict, List, Optional
import torch

from ..core.model_runner import ModelRunner
from ..core.model import MoEConfig
from ..memory.expert_cache import ExpertCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.prefetcher import ExpertPrefetcher
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector
from .model_runner_utils import build_layer_activations, build_prefill_placement

logger = get_logger(__name__)


class PrefillEngine:
    """
    Prefill phase execution engine using ModelRunner.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: Optional[ExpertPrefetcher],
        metrics: MetricsCollector,
    ):
        self.model_runner = model_runner
        self.config: MoEConfig = model_runner.get_config()
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.metrics = metrics
        self.activation_history: List = []

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache,
        positions: Optional[torch.Tensor] = None,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = True,
    ) -> Dict:
        self.metrics.start_phase('prefill')

        input_ids_gpu = input_ids.cuda()
        if input_ids_gpu.dim() == 1:
            input_ids_gpu = input_ids_gpu.unsqueeze(0)
        batch_size, seq_len = input_ids_gpu.shape

        if positions is None:
            positions = torch.arange(seq_len, device=input_ids_gpu.device).unsqueeze(0).expand(batch_size, -1)

        if seq_ids is None:
            seq_ids = list(range(batch_size))

        hidden_states = self.model_runner.embed(input_ids_gpu)

        for layer_idx in range(self.model_runner.get_num_layers()):
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                positions=positions,
                seq_ids=seq_ids,
                is_prefill=is_prefill,
            )

            routing_result = self.model_runner.route_experts(
                layer_idx=layer_idx,
                hidden_states=attn_output.post_attn_normed,
            )

            placement = build_prefill_placement(
                routing_result=routing_result,
                expert_cache=self.expert_cache,
                parameter_loader=self.parameter_loader,
            )

            if self.prefetcher is not None:
                layer_acts = build_layer_activations(routing_result, self.config.num_experts)
                self.prefetcher.prefetch_for_next_layer(
                    current_layer_idx=layer_idx,
                    current_activations=layer_acts,
                    history=self.activation_history,
                    expert_cache=self.expert_cache,
                    parameter_loader=self.parameter_loader,
                )
                self.activation_history.append(layer_acts)
                if len(self.activation_history) > 20:
                    self.activation_history.pop(0)

            hidden_states = self.model_runner.forward_moe(
                layer_idx=layer_idx,
                attn_output=attn_output,
                expert_placement=placement,
            )

        logits = self.model_runner.compute_logits(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        self.metrics.end_phase('prefill')

        return {
            'logits': logits,
            'next_token_id': next_token_id,
            'hidden_states': hidden_states,
        }
