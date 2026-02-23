from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
from itertools import count
from typing import Dict, List, Optional, Tuple
import torch

from ..core.model_runner import ModelRunner
from ..core.types import ExpertID, LayerExpertActivations
from ..memory.paged_kv_cache import PagedKVCache
from ..memory.expert_cache import ExpertCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.prefetcher import ExpertPrefetcher
from ..scheduling.draft_schduler import DraftSchedulingStrategy
from .acceptance_strategy import AcceptanceStrategy
from .model_runner_utils import build_prefill_placement, build_draft_placement, build_layer_activations


def select_experts_to_prefetch(
    current_step: int,
    max_draft_tokens: int,
    step_activations: List[LayerExpertActivations],
    cached_experts: set[ExpertID],
    pending_transfers: set[ExpertID],
    cache_capacity: int,
    num_experts_per_layer: int,
    *,
    max_prefetch_per_step: int = 4,
) -> List[ExpertID]:
    if not step_activations:
        return []

    expert_stats: Dict[ExpertID, Tuple[int, float]] = {}
    for layer_acts in step_activations:
        for act in layer_acts.activations:
            eid = act.expert_id
            count, total_score = expert_stats.get(eid, (0, 0.0))
            expert_stats[eid] = (count + 1, total_score + act.scores.max().item())

    already_available = cached_experts | pending_transfers
    candidates = {
        eid: (count, total_score)
        for eid, (count, total_score) in expert_stats.items()
        if eid not in already_available
    }

    if not candidates:
        return []

    available_slots = cache_capacity - len(cached_experts) - len(pending_transfers)
    if available_slots <= 0:
        return []

    scored = [
        (eid, count * (total_score / count))
        for eid, (count, total_score) in candidates.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    num_to_prefetch = min(max_prefetch_per_step, available_slots, len(scored))
    return [eid for eid, _ in scored[:num_to_prefetch]]


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    DRAFTING = auto()
    FINISHED = auto()
    ERROR = auto()


class Sequence:
    _counter = count()

    def __init__(
        self,
        token_ids: List[int],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ):
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING

        self.prompt_token_ids = list(token_ids)
        self.output_token_ids: List[int] = []

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.eos_token_id = eos_token_id

        self.draft_token_ids: List[int] = []
        self.num_tokens_before_draft = 0
        self.error_msg: Optional[str] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ERROR)

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_finished(self) -> bool:
        if self.status == SequenceStatus.FINISHED:
            return False
        if self.num_generated >= self.max_new_tokens:
            self.status = SequenceStatus.FINISHED
            return True
        if self.eos_token_id is not None and self.last_token_id == self.eos_token_id:
            self.status = SequenceStatus.FINISHED
            return True
        return False

    def mark_error(self, msg: str) -> None:
        self.status = SequenceStatus.ERROR
        self.error_msg = msg

    def start_draft(self) -> None:
        self.status = SequenceStatus.DRAFTING
        self.draft_token_ids = []
        self.num_tokens_before_draft = self.total_len

    def append_draft_token(self, token_id: int) -> None:
        self.draft_token_ids.append(token_id)

    def accept_draft(self, num_accepted: int) -> None:
        accepted = self.draft_token_ids[:num_accepted]
        self.output_token_ids.extend(accepted)
        self.draft_token_ids = []
        self.num_tokens_before_draft = 0
        self.status = SequenceStatus.RUNNING

    @property
    def num_draft_tokens(self) -> int:
        return len(self.draft_token_ids)

    @property
    def last_draft_token_id(self) -> int:
        if self.draft_token_ids:
            return self.draft_token_ids[-1]
        return self.last_token_id


class DecodeMode(Enum):
    STANDARD = "standard"
    SPECULATIVE = "speculative"


@dataclass
class ScheduleResult:
    sequences: List[Sequence]
    is_prefill: bool


class CBScheduler:
    def __init__(
        self,
        kv_cache: PagedKVCache,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
    ):
        self.kv_cache = kv_cache
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> ScheduleResult:
        scheduled: List[Sequence] = []
        num_seqs = 0
        num_tokens = 0
        if not self.running:
            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                prompt_len = seq.prompt_len

                if num_tokens + prompt_len > self.max_num_batched_tokens:
                    break

                num_blocks_needed = (prompt_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
                if not self.kv_cache.block_manager.can_allocate(num_blocks_needed):
                    break

                success = self.kv_cache.add_sequence(seq.seq_id, prompt_len=prompt_len)
                if not success:
                    break

                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled.append(seq)
                num_seqs += 1
                num_tokens += prompt_len

            if scheduled:
                return ScheduleResult(sequences=scheduled, is_prefill=True)

        remaining_running: deque[Sequence] = deque()

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            if not self.kv_cache.can_append_token(seq.seq_id):
                seq.mark_error("KV cache exhausted, cannot append token")
                self.kv_cache.remove_sequence(seq.seq_id)
                continue

            scheduled.append(seq)
            remaining_running.append(seq)
            num_seqs += 1

        remaining_running.extend(self.running)
        self.running = remaining_running

        if not scheduled:
            raise RuntimeError(
                "Scheduler: no sequences can be scheduled. "
                f"waiting={len(self.waiting)}, running={len(self.running)}"
            )

        return ScheduleResult(sequences=scheduled, is_prefill=False)

    def postprocess(self, seqs: List[Sequence], finished_seq_ids: List[int]) -> None:
        for seq_id in finished_seq_ids:
            self.kv_cache.remove_sequence(seq_id)

        self.running = deque(seq for seq in self.running if not seq.is_finished)


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
        seq_ids = [seq.seq_id for seq in seqs]
        self.draft_activations = []

        for seq in seqs:
            seq.start_draft()
            self.kv_cache.start_draft(seq.seq_id)

        for _ in range(self.max_draft_tokens):
            active_seqs = [s for s in seqs if s.status == SequenceStatus.DRAFTING]
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
        if not activations or self.draft_scheduler is None:
            return

        cached_experts = set(self.expert_cache.cached_experts.keys())
        experts_to_transfer = self.draft_scheduler.select_experts_to_transfer(
            recent_activations=activations,
            cached_experts=cached_experts,
            cache_capacity=self.expert_cache.max_experts,
        )

        if not experts_to_transfer:
            return

        source_params = {}
        for expert_id in experts_to_transfer:
            cpu_params = self.parameter_loader.get_expert_params(expert_id)
            if cpu_params:
                source_params[expert_id] = cpu_params

        if source_params:
            self.expert_cache.prefetch_async(list(source_params.keys()), source_params)

    def _execute_verify(self, seqs: List[Sequence]) -> Dict[int, torch.Tensor]:
        verify_logits_map: Dict[int, torch.Tensor] = {}

        for seq in seqs:
            if seq.is_finished or seq.num_draft_tokens == 0:
                continue

            num_draft = seq.num_draft_tokens
            verify_input_ids = [seq.last_token_id] + seq.draft_token_ids
            input_ids = torch.tensor(verify_input_ids, dtype=torch.long, device="cuda")

            draft_start_pos = seq.num_tokens_before_draft
            positions = torch.arange(
                draft_start_pos - 1,
                draft_start_pos + num_draft,
                dtype=torch.long, device="cuda",
            )

            logits = self._forward_verify(input_ids, positions, [seq.seq_id])
            if logits.dim() == 3:
                logits = logits[0]
            verify_logits = logits[:num_draft]
            verify_logits_map[seq.seq_id] = verify_logits

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

            num_accepted = result["num_accepted"]
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


class ContinuousBatchEngine:
    def __init__(
        self,
        model_runner: ModelRunner,
        kv_cache: PagedKVCache,
        expert_cache: ExpertCache,
        parameter_loader: ParameterLoader,
        prefetcher: Optional[ExpertPrefetcher] = None,
        draft_scheduler: Optional[DraftSchedulingStrategy] = None,
        acceptance_strategy: Optional[AcceptanceStrategy] = None,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        decode_mode: DecodeMode = DecodeMode.STANDARD,
        max_draft_tokens: int = 8,
    ):
        self.decode_mode = decode_mode

        self.scheduler = CBScheduler(
            kv_cache=kv_cache,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        self.executor = CBExecutor(
            model_runner=model_runner,
            kv_cache=kv_cache,
            expert_cache=expert_cache,
            parameter_loader=parameter_loader,
            prefetcher=prefetcher,
            draft_scheduler=draft_scheduler,
            acceptance_strategy=acceptance_strategy,
            max_draft_tokens=max_draft_tokens,
        )

    def add_sequences(self, seqs: List[Sequence]) -> None:
        for seq in seqs:
            self.scheduler.add(seq)

    def step(self) -> Tuple[List[Sequence], List[Sequence]]:
        if self.scheduler.is_finished():
            return [], []

        schedule_result = self.scheduler.schedule()
        finished_ids: List[int] = []

        try:
            if schedule_result.is_prefill:
                finished_ids = self.executor.execute_prefill(schedule_result.sequences)
            else:
                if self.decode_mode == DecodeMode.SPECULATIVE:
                    finished_ids = self.executor.execute_decode_speculative(schedule_result.sequences)
                else:
                    finished_ids = self.executor.execute_decode_standard(schedule_result.sequences)
        except Exception as e:
            for seq in schedule_result.sequences:
                seq.mark_error(f"Execution error: {e}")
            finished_ids = [seq.seq_id for seq in schedule_result.sequences]

        self.scheduler.postprocess(schedule_result.sequences, finished_ids)
        finished_sequences = [seq for seq in schedule_result.sequences if seq.seq_id in finished_ids]
        return schedule_result.sequences, finished_sequences

    def generate(
        self,
        prompts: List[List[int]],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> List[Dict]:
        seq_id_to_idx: Dict[int, int] = {}
        seqs = []
        for idx, prompt in enumerate(prompts):
            seq = Sequence(
                token_ids=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
            )
            self.scheduler.add(seq)
            seq_id_to_idx[seq.seq_id] = idx
            seqs.append(seq)

        results: Dict[int, Dict] = {}
        while not self.scheduler.is_finished():
            _, finished_sequences = self.step()
            for seq in finished_sequences:
                idx = seq_id_to_idx.get(seq.seq_id)
                if idx is not None:
                    results[idx] = {
                        "seq_id": seq.seq_id,
                        "prompt_token_ids": seq.prompt_token_ids,
                        "output_token_ids": seq.output_token_ids,
                        "error": seq.error_msg,
                    }

        return [results.get(i, {"error": "not found"}) for i in range(len(prompts))]
