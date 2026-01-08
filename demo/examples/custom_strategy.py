"""
Example showing how to implement custom strategies.
"""

from src.api.inference import MoEInferenceEngine
from src.scheduling.base_scheduler import DraftSchedulingStrategy
from src.core.types import ExpertID, LayerExpertActivations
from typing import List, Dict, Set


class MyCustomDraftScheduler(DraftSchedulingStrategy):
    """Custom draft scheduling strategy"""
    
    def select_cpu_experts(
        self,
        activations: LayerExpertActivations,
        top_c: int
    ) -> List[ExpertID]:
        # Custom logic here
        return []
    
    def select_gpu_substitutes(
        self,
        requested_experts: List[ExpertID],
        cached_experts: Set[ExpertID],
        all_experts: List[ExpertID]
    ) -> Dict[ExpertID, ExpertID]:
        # Custom logic here
        return {}
    
    def select_experts_to_transfer(
        self,
        recent_activations: List[LayerExpertActivations],
        cached_experts: Set[ExpertID],
        cache_capacity: int
    ) -> List[ExpertID]:
        # Custom logic here
        return []
    
    def should_trigger_verify(
        self,
        num_drafted_tokens: int,
        perplexity: float,
        cache_hit_rate: float,
        max_draft_tokens: int
    ) -> bool:
        # Custom logic here
        return num_drafted_tokens >= max_draft_tokens


def main():
    # Initialize with custom strategy
    engine = MoEInferenceEngine(
        model_path="path/to/model",
        config_dir="configs"
    )
    
    # Replace draft scheduler with custom one
    engine.orchestrator.draft_scheduler = MyCustomDraftScheduler()
    
    # Run inference
    output = engine.generate("Hello world", max_new_tokens=100)
    print(output)


if __name__ == "__main__":
    main()