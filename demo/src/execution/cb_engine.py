from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.model_runner import ModelRunner
from ..core.sequence import DecodeMode, Sequence
from ..memory.expert_cache import ExpertCache
from ..memory.paged_kv_cache import PagedKVCache
from ..memory.parameter_loader import ParameterLoader
from ..scheduling.draft_schduler import DraftSchedulingStrategy
from ..scheduling.prefetcher import ExpertPrefetcher
from .acceptance_strategy import AcceptanceStrategy
from .cb_executor import CBExecutor
from .cb_scheduler import CBScheduler


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
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> List[Dict]:
        seq_id_to_idx: Dict[int, int] = {}
        for idx, prompt in enumerate(prompts):
            seq = Sequence(
                token_ids=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
            )
            self.scheduler.add(seq)
            seq_id_to_idx[seq.seq_id] = idx

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
