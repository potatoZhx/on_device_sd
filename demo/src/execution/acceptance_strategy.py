from abc import ABC, abstractmethod
from typing import Dict
import torch
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AcceptanceStrategy(ABC):
    """
    Abstract base class for speculative sampling acceptance strategies.
    """
    
    @abstractmethod
    def accept(
        self,
        draft_token_ids: torch.Tensor,
        verify_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict:
        """
        Determine which draft tokens to accept.
        
        Args:
            draft_token_ids: Draft token IDs [num_draft_tokens]
            verify_logits: Verification logits [num_draft_tokens, vocab_size]
            temperature: Sampling temperature
        
        Returns:
            Dict with 'num_accepted', 'accepted_tokens', 'rejection_position'
        """
        pass


class StandardAcceptanceStrategy(AcceptanceStrategy):
    """
    Standard speculative sampling acceptance.
    Accept tokens while verify probability is high enough.
    """
    
    def __init__(self, acceptance_threshold: float = 0.7):
        self.acceptance_threshold = acceptance_threshold
    
    def accept(
        self,
        draft_token_ids: torch.Tensor,
        verify_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict:
        """
        Standard acceptance: accept while P_verify(token) >= threshold * P_draft(token)
        """
        num_draft_tokens = len(draft_token_ids)
        
        # Get verify probabilities
        verify_probs = torch.softmax(verify_logits / temperature, dim=-1)  # [num_tokens, vocab]
        
        # Get probabilities of draft tokens
        draft_token_probs = verify_probs[
            torch.arange(num_draft_tokens),
            draft_token_ids
        ]  # [num_tokens]
        
        # Accept tokens while probability is above threshold
        acceptance_mask = draft_token_probs >= self.acceptance_threshold
        
        # Find first rejection
        if torch.all(acceptance_mask):
            num_accepted = num_draft_tokens
            rejection_position = -1
        else:
            rejection_position = torch.where(~acceptance_mask)[0][0].item()
            num_accepted = rejection_position
        
        accepted_tokens = draft_token_ids[:num_accepted]
        
        logger.info(f"Accepted {num_accepted}/{num_draft_tokens} draft tokens")
        
        return {
            'num_accepted': num_accepted,
            'accepted_tokens': accepted_tokens,
            'rejection_position': rejection_position,
            'acceptance_probs': draft_token_probs
        }


class AdaptiveAcceptanceStrategy(AcceptanceStrategy):
    """
    Adaptive acceptance with dynamic threshold adjustment.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.7,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.5,
        max_threshold: float = 0.95
    ):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Track acceptance rates
        self.recent_acceptance_rates = []
        self.window_size = 10
    
    def accept(
        self,
        draft_token_ids: torch.Tensor,
        verify_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict:
        """Adaptive acceptance with learning"""
        num_draft_tokens = len(draft_token_ids)
        
        verify_probs = torch.softmax(verify_logits / temperature, dim=-1)
        draft_token_probs = verify_probs[
            torch.arange(num_draft_tokens),
            draft_token_ids
        ]
        
        # Accept based on current threshold
        acceptance_mask = draft_token_probs >= self.threshold
        
        if torch.all(acceptance_mask):
            num_accepted = num_draft_tokens
            rejection_position = -1
        else:
            rejection_position = torch.where(~acceptance_mask)[0][0].item()
            num_accepted = rejection_position
        
        accepted_tokens = draft_token_ids[:num_accepted]
        
        # Update threshold based on acceptance rate
        acceptance_rate = num_accepted / num_draft_tokens
        self.recent_acceptance_rates.append(acceptance_rate)
        
        if len(self.recent_acceptance_rates) > self.window_size:
            self.recent_acceptance_rates.pop(0)
        
        # Adjust threshold
        avg_acceptance_rate = sum(self.recent_acceptance_rates) / len(self.recent_acceptance_rates)
        
        if avg_acceptance_rate < 0.5:
            # Too many rejections, lower threshold
            self.threshold = max(
                self.min_threshold,
                self.threshold * (1 - self.adaptation_rate)
            )
        elif avg_acceptance_rate > 0.9:
            # High acceptance, can raise threshold
            self.threshold = min(
                self.max_threshold,
                self.threshold * (1 + self.adaptation_rate)
            )
        
        logger.info(f"Accepted {num_accepted}/{num_draft_tokens} tokens, "
                   f"threshold: {self.threshold:.3f}")
        
        return {
            'num_accepted': num_accepted,
            'accepted_tokens': accepted_tokens,
            'rejection_position': rejection_position,
            'acceptance_probs': draft_token_probs,
            'current_threshold': self.threshold
        }