from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from ..core.model_runner import ModelRunner
from ..core.types import BatchedRequest, GenerationConfig
from ..core.model import MoEConfig
from ..memory.expert_cache import ExpertCache
from ..memory.parameter_loader import ParameterLoader
from ..memory.paged_kv_cache import PagedKVCache
from ..scheduling.prefetcher import ExpertPrefetcher
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector
from .model_runner_utils import build_prefill_placement
from .prefill_engine import PrefillEngine

logger = get_logger(__name__)


class StandardDecodeEngine:
    """
    Standard autoregressive decoding engine using ModelRunner.
    Uses per-request generation (sequential) for simplicity.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: Optional[ExpertPrefetcher] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.model_runner = model_runner
        self.config: MoEConfig = model_runner.get_config()
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.metrics = metrics or MetricsCollector()

        self.prefill_engine = PrefillEngine(
            model_runner=model_runner,
            parameter_loader=parameter_loader,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=self.metrics,
        )

        logger.info("StandardDecodeEngine initialized")

    def generate_batch(self, batch: BatchedRequest) -> Dict:
        self.metrics.start_phase('standard_generation')

        generated_sequences = []
        for request in batch.requests:
            generated_sequences.append(self._generate_single(request))

        self.metrics.end_phase('standard_generation')

        return {
            'generated_sequences': generated_sequences,
            'statistics': {},
        }

    def _generate_single(self, request) -> torch.Tensor:
        kv_cache = PagedKVCache(
            config=self.config,
            block_size=256,
            dtype=self.config.get_dtype(),
        )

        prefill_output = self.prefill_engine.forward(
            input_ids=request.input_ids,
            kv_cache=kv_cache,
            is_prefill=True,
        )

        generated_ids = [prefill_output['next_token_id'].item()]
        current_token = prefill_output['next_token_id'].view(1, 1)

        max_new_tokens = request.generation_config.max_new_tokens

        while len(generated_ids) < max_new_tokens:
            next_token = self._decode_step(
                current_token=current_token,
                kv_cache=kv_cache,
                generation_config=request.generation_config,
            )
            generated_ids.append(next_token.item())
            current_token = next_token.view(1, 1)

        return torch.tensor(generated_ids, dtype=torch.long)

    def _decode_step(
        self,
        current_token: torch.Tensor,
        kv_cache: PagedKVCache,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        if hasattr(kv_cache, "sequences") and hasattr(kv_cache, "append_token"):
            if 0 not in kv_cache.sequences:
                kv_cache.add_sequence(0, prompt_len=1)
            else:
                kv_cache.append_token(0)

        hidden_states = self.model_runner.embed(current_token.cuda())

        for layer_idx in range(self.model_runner.get_num_layers()):
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                positions=None,
                seq_ids=[0],
                is_prefill=False,
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

            hidden_states = self.model_runner.forward_moe(
                layer_idx=layer_idx,
                attn_output=attn_output,
                expert_placement=placement,
            )

        logits = self.model_runner.compute_logits(hidden_states)
        next_token = self._sample_token(logits[:, -1, :], generation_config)
        return next_token

    def _sample_token(self, logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        logits = logits / config.temperature

        if config.do_sample:
            if config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, config.top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)

            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token.squeeze()
