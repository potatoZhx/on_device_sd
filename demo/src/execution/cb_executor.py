from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..core.model_runner import ModelRunner
from ..core.types import DeviceType
from ..core.sequence import Sequence, SequenceStatus
from ..memory.expert_cache import ExpertCache
from ..memory.paged_kv_cache import PagedKVCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.draft_schduler import DraftSchedulingStrategy
from ..scheduling.prefetcher import ExpertPrefetcher
from .acceptance_strategy import AcceptanceStrategy
from .model_runner_utils import build_draft_placement, build_layer_activations, build_prefill_placement
from .prefetch_selector import select_experts_to_prefetch


class CBExecutor:
    def __init__(
        self,
        model_runner: ModelRunner,
        kv_cache: PagedKVCache,
        expert_cache: ExpertCache,
        parameter_loader: ParameterLoader,
        prefetcher: Optional[ExpertPrefetcher] = None,
        draft_scheduler: Optional[DraftSchedulingStrategy] = None,
        acceptance_strategy: Optional[AcceptanceStrategy] = None,
        max_draft_tokens: int = 8,
    ):
        self.model_runner = model_runner
        self.kv_cache = kv_cache
        self.expert_cache = expert_cache
        self.parameter_loader = parameter_loader
        self.prefetcher = prefetcher
        self.draft_scheduler = draft_scheduler
        self.acceptance_strategy = acceptance_strategy
        self.max_draft_tokens = max_draft_tokens
        self.draft_activations: List = []
        cfg = self.model_runner.get_config()
        total_routed_experts = cfg.num_hidden_layers * cfg.num_experts
        if hasattr(self.parameter_loader, "get_gpu_expert_count"):
            self._all_routed_experts_on_gpu = (
                self.parameter_loader.get_gpu_expert_count() >= total_routed_experts
            )
        else:
            self._all_routed_experts_on_gpu = False

    def execute_prefill(self, seqs: List[Sequence]) -> List[int]:
        seq_ids = [seq.seq_id for seq in seqs]

        input_ids_list = []
        positions_list = []
        for seq in seqs:
            prompt = seq.prompt_token_ids
            input_ids_list.extend(prompt)
            positions_list.extend(range(len(prompt)))

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device="cuda")
        positions = torch.tensor(positions_list, dtype=torch.long, device="cuda")

        logits = self._forward(input_ids, positions, seq_ids, is_prefill=True)

        last_indices = []
        offset = 0
        for seq in seqs:
            last_indices.append(offset + seq.prompt_len - 1)
            offset += seq.prompt_len
        last_logits = logits[last_indices]

        token_ids = self._sample(last_logits, seqs)

        finished_seq_ids = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if not self.kv_cache.append_token(seq.seq_id):
                seq.mark_error("KV cache exhausted during prefill append")
                finished_seq_ids.append(seq.seq_id)
                continue
            if seq.check_finished():
                finished_seq_ids.append(seq.seq_id)

        return finished_seq_ids

    def execute_decode_standard(self, seqs: List[Sequence]) -> List[int]:
        seq_ids = [seq.seq_id for seq in seqs]

        for seq_id in seq_ids:
            if not self.kv_cache.append_token(seq_id):
                seq = next(s for s in seqs if s.seq_id == seq_id)
                seq.mark_error("KV cache exhausted during decode")

        input_ids = torch.tensor(
            [seq.last_token_id for seq in seqs], dtype=torch.long, device="cuda"
        ).unsqueeze(1)
        positions = torch.tensor(
            [seq.total_len - 1 for seq in seqs], dtype=torch.long, device="cuda"
        ).unsqueeze(1)

        logits = self._forward(input_ids, positions, seq_ids, is_prefill=False)
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        token_ids = self._sample(logits, seqs)

        finished_seq_ids = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if seq.check_finished():
                finished_seq_ids.append(seq.seq_id)

        return finished_seq_ids

    def execute_decode_speculative(self, seqs: List[Sequence]) -> List[int]:
        draft_seqs = [seq for seq in seqs if not seq.is_finished]
        if not draft_seqs:
            return []

        self._execute_draft(draft_seqs)
        verify_logits_map = self._execute_verify(draft_seqs)
        finished_seq_ids = self._execute_accept(draft_seqs, verify_logits_map)
        return finished_seq_ids

    def _execute_draft(self, seqs: List[Sequence]) -> None:
        self.draft_activations = []

        for seq in seqs:
            seq.start_draft()
            self.kv_cache.start_draft(seq.seq_id)

        for _ in range(self.max_draft_tokens):
            active_seqs = [
                s for s in seqs
                if s.status == SequenceStatus.DRAFTING
                and s.num_draft_tokens < max(0, s.max_new_tokens - s.num_generated)
            ]
            if not active_seqs:
                break
            active_seq_ids = [s.seq_id for s in active_seqs]

            for seq_id in active_seq_ids:
                if not self.kv_cache.can_append_token(seq_id):
                    seq = next(s for s in active_seqs if s.seq_id == seq_id)
                    seq.mark_error("KV cache exhausted during draft")
                    continue
                self.kv_cache.append_token(seq_id)

            active_seqs = [s for s in seqs if s.status == SequenceStatus.DRAFTING]
            active_seqs = [
                s for s in active_seqs
                if s.num_draft_tokens < max(0, s.max_new_tokens - s.num_generated)
            ]
            if not active_seqs:
                break
            active_seq_ids = [s.seq_id for s in active_seqs]

            input_ids = torch.tensor(
                [s.last_draft_token_id for s in active_seqs],
                dtype=torch.long, device="cuda"
            ).unsqueeze(1)
            positions = torch.tensor(
                [s.num_tokens_before_draft + s.num_draft_tokens - 1 for s in active_seqs],
                dtype=torch.long, device="cuda"
            ).unsqueeze(1)

            logits, step_activations, _, _ = self._forward_draft(
                input_ids, positions, active_seq_ids
            )
            self.draft_activations.extend(step_activations)

            if logits.dim() == 3:
                logits = logits[:, -1, :]

            token_ids = self._sample(logits, active_seqs)

            for seq, token_id in zip(active_seqs, token_ids):
                seq.append_draft_token(token_id)

            self._schedule_expert_transfers(step_activations)
            self.expert_cache.complete_ready_transfers()

        self._schedule_expert_transfers()
        self.expert_cache.complete_ready_transfers()

    def _schedule_expert_transfers(self, recent_activations: Optional[List] = None) -> None:
        activations = recent_activations or self.draft_activations
        if not activations:
            return

        cached_experts = set(self.expert_cache.cached_experts.keys())
        pending_transfers = set(self.expert_cache.pending_transfers.keys())
        experts_to_transfer = select_experts_to_prefetch(
            current_step=max(0, len(activations) - 1),
            max_draft_tokens=self.max_draft_tokens,
            step_activations=activations,
            cached_experts=cached_experts,
            pending_transfers=pending_transfers,
            cache_capacity=self.expert_cache.max_experts,
            num_experts_per_layer=self.model_runner.get_config().num_experts,
        )

        if not experts_to_transfer:
            return

        source_params = {}
        for expert_id in experts_to_transfer:
            if self.parameter_loader.get_expert_params(expert_id, device=None) is None:
                continue
            if self.parameter_loader.get_expert_params(expert_id, device=DeviceType.GPU) is not None:
                continue
            cpu_params = self.parameter_loader.get_expert_params(expert_id, device=DeviceType.CPU)
            if cpu_params:
                source_params[expert_id] = cpu_params

        if source_params:
            self.expert_cache.prefetch_async(list(source_params.keys()), source_params)

    def _execute_verify(self, seqs: List[Sequence]) -> Dict[int, torch.Tensor]:
        verify_logits_map: Dict[int, torch.Tensor] = {}

        verify_seqs = [
            seq for seq in seqs
            if (not seq.is_finished) and seq.num_draft_tokens > 0
        ]
        if not verify_seqs:
            return verify_logits_map

        batched_input_ids: List[int] = []
        batched_positions: List[int] = []
        seq_ids: List[int] = []
        verify_lengths: List[int] = []

        for seq in verify_seqs:
            num_draft = seq.num_draft_tokens
            verify_input_ids = [seq.last_token_id] + seq.draft_token_ids
            draft_start_pos = seq.num_tokens_before_draft
            verify_positions = list(range(draft_start_pos - 1, draft_start_pos + num_draft))

            batched_input_ids.extend(verify_input_ids)
            batched_positions.extend(verify_positions)
            seq_ids.append(seq.seq_id)
            verify_lengths.append(num_draft + 1)

        input_ids = torch.tensor(batched_input_ids, dtype=torch.long, device="cuda")
        positions = torch.tensor(batched_positions, dtype=torch.long, device="cuda")

        logits = self._forward_verify(input_ids, positions, seq_ids)
        if logits.dim() == 3:
            logits = logits[0]

        offset = 0
        for seq, verify_len in zip(verify_seqs, verify_lengths):
            num_draft = seq.num_draft_tokens
            seq_logits = logits[offset: offset + verify_len]
            verify_logits_map[seq.seq_id] = seq_logits[:num_draft]
            offset += verify_len

        return verify_logits_map

    def _execute_accept(
        self,
        seqs: List[Sequence],
        verify_logits_map: Dict[int, torch.Tensor],
    ) -> List[int]:
        finished_seq_ids = []

        for seq in seqs:
            if seq.is_finished:
                finished_seq_ids.append(seq.seq_id)
                continue

            if seq.seq_id not in verify_logits_map:
                seq.status = SequenceStatus.RUNNING
                continue

            verify_logits = verify_logits_map[seq.seq_id]
            draft_token_ids = torch.tensor(seq.draft_token_ids, dtype=torch.long, device="cuda")

            result = self.acceptance_strategy.accept(
                draft_token_ids=draft_token_ids,
                verify_logits=verify_logits,
                temperature=seq.temperature,
            )

            remaining_budget = max(0, seq.max_new_tokens - seq.num_generated)
            num_accepted = min(int(result["num_accepted"]), remaining_budget)
            seq.accept_draft(num_accepted)

            try:
                self.kv_cache.accept_draft(seq.seq_id, num_accepted)
            except Exception as e:
                seq.mark_error(f"KV cache reconcile failed: {e}")
                finished_seq_ids.append(seq.seq_id)
                continue

            if seq.check_finished():
                finished_seq_ids.append(seq.seq_id)

        return finished_seq_ids

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: List[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        hidden_states = self.model_runner.embed(input_ids)
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        for layer_idx in range(self.model_runner.get_num_layers()):
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=self.kv_cache,
                positions=positions,
                seq_ids=seq_ids,
                is_prefill=is_prefill,
            )

            routing_result = self.model_runner.route_experts(
                layer_idx=layer_idx,
                hidden_states=attn_output.post_attn_normed,
            )

            if self.prefetcher is not None:
                activations = build_layer_activations(
                    routing_result, self.model_runner.get_config().num_experts
                )
                self.prefetcher.on_layer_complete(layer_idx, activations)

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

        return self.model_runner.compute_logits(hidden_states)

    @torch.inference_mode()
    def _forward_draft(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: List[int],
    ) -> Tuple[torch.Tensor, List, int, int]:
        hidden_states = self.model_runner.embed(input_ids)
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        config = self.model_runner.get_config()
        layer_activations = []
        cache_hits = 0
        cache_misses = 0

        for layer_idx in range(self.model_runner.get_num_layers()):
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=self.kv_cache,
                positions=positions,
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
                top_c=config.draft_top_c,
                num_experts=config.num_experts,
            )

            cache_hits += len(placement.gpu_expert_params)
            cache_misses += len(placement.cpu_expert_params)

            if not self._all_routed_experts_on_gpu:
                acts = build_layer_activations(routing_result, config.num_experts)
                layer_activations.append(acts)

            hidden_states = self.model_runner.forward_moe(
                layer_idx=layer_idx,
                attn_output=attn_output,
                expert_placement=placement,
            )

        logits = self.model_runner.compute_logits(hidden_states)
        return logits, layer_activations, cache_hits, cache_misses

    @torch.inference_mode()
    def _forward_verify(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: List[int],
    ) -> torch.Tensor:
        hidden_states = self.model_runner.embed(input_ids)
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        for layer_idx in range(self.model_runner.get_num_layers()):
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=self.kv_cache,
                positions=positions,
                seq_ids=seq_ids,
                is_prefill=False,
                is_verify=True,
            )

            routing_result = self.model_runner.route_experts(
                layer_idx=layer_idx,
                hidden_states=attn_output.post_attn_normed,
            )

            if self.prefetcher is not None:
                activations = build_layer_activations(
                    routing_result, self.model_runner.get_config().num_experts
                )
                self.prefetcher.on_layer_complete(layer_idx, activations)

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

        return self.model_runner.compute_logits(hidden_states)

    def _sample(self, logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
        token_ids = []
        for idx, seq in enumerate(seqs):
            logit = logits[idx]
            token_ids.append(self._sample_one(logit, seq))
        return token_ids

    def _sample_one(self, logits: torch.Tensor, seq: Sequence) -> int:
        if not seq.do_sample:
            return int(torch.argmax(logits, dim=-1).item())

        temperature = seq.temperature
        top_k = seq.top_k
        top_p = seq.top_p

        logits = logits / temperature

        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            mask_logits = torch.full_like(logits, float("-inf"))
            mask_logits.scatter_(-1, top_k_indices, top_k_logits)
            logits = mask_logits

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.clone()
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return int(token_id.item())
