from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List

from ..core.sequence import Sequence, SequenceStatus
from ..memory.paged_kv_cache import PagedKVCache


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
        del seqs
        for seq_id in finished_seq_ids:
            self.kv_cache.remove_sequence(seq_id)

        self.running = deque(seq for seq in self.running if not seq.is_finished)
