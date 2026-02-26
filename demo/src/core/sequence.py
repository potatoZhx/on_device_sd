from __future__ import annotations

from enum import Enum, auto
from itertools import count
from typing import List, Optional


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    DRAFTING = auto()
    FINISHED = auto()
    ERROR = auto()


class DecodeMode(Enum):
    STANDARD = "standard"
    SPECULATIVE = "speculative"


class Sequence:
    _counter = count()

    def __init__(
        self,
        token_ids: List[int],
        max_new_tokens: int = 64,
        do_sample: bool = True,
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
        self.do_sample = do_sample
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
        if self.status == SequenceStatus.WAITING:
            self.status = SequenceStatus.RUNNING
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
