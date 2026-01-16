import unittest
import torch
from src.core.types import ExpertID, LayerExpertActivations, ExpertActivation
from src.core.model import MoEConfig
from src.scheduling.prefetcher import SimplePrefetchStrategy
from src.scheduling.draft_scheduler import SimpleDraftScheduler
from src.scheduling.cache_strategy import LRUCacheStrategy, AdaptiveCacheStrategy


class TestPrefetchStrategy(unittest.TestCase):
    """Test prefetch strategies"""
    
    def setUp(self):
        self.strategy = SimplePrefetchStrategy(num_experts_to_prefetch=4)
    
    def test_predict_next_experts(self):
        """Test expert prediction"""
        # Create mock activations
        activations = [
            ExpertActivation(
                expert_id=ExpertID(0, i),
                token_indices=torch.tensor([0]),
                scores=torch.tensor([0.9 - i * 0.1]),
                top_k_rank=i
            )
            for i in range(8)
        ]
        
        layer_acts = LayerExpertActivations(
            layer_idx=0,
            activations=activations,
            routing_scores=torch.randn(1, 8)
        )
        
        # Predict
        predictions = self.strategy.predict_next_experts(0, layer_acts, None)
        
        # Should predict 4 experts for next layer
        self.assertEqual(len(predictions), 4)
        self.assertTrue(all(p.layer_idx == 1 for p in predictions))


class TestDraftScheduler(unittest.TestCase):
    """Test draft scheduling"""
    
    def setUp(self):
        self.scheduler = SimpleDraftScheduler()
    
    def test_select_cpu_experts(self):
        """Test CPU expert selection"""
        activations = [
            ExpertActivation(
                expert_id=ExpertID(0, i),
                token_indices=torch.tensor([0]),
                scores=torch.tensor([0.9 - i * 0.1]),
                top_k_rank=i
            )
            for i in range(8)
        ]
        
        layer_acts = LayerExpertActivations(
            layer_idx=0,
            activations=activations,
            routing_scores=torch.randn(1, 8)
        )
        
        # Select top-2
        cpu_experts = self.scheduler.select_cpu_experts(layer_acts, top_c=2)
        
        self.assertEqual(len(cpu_experts), 2)
        # Should select experts with highest scores
        self.assertEqual(cpu_experts[0].expert_idx, 0)
        self.assertEqual(cpu_experts[1].expert_idx, 1)
    
    def test_should_trigger_verify(self):
        """Test verify trigger conditions"""
        # Test max tokens
        should_verify = self.scheduler.should_trigger_verify(
            num_drafted_tokens=8,
            perplexity=1.0,
            cache_hit_rate=0.8,
            max_draft_tokens=8
        )
        self.assertTrue(should_verify)
        
        # Test high perplexity
        should_verify = self.scheduler.should_trigger_verify(
            num_drafted_tokens=4,
            perplexity=2.0,  # Above threshold
            cache_hit_rate=0.8,
            max_draft_tokens=8
        )
        self.assertTrue(should_verify)
        
        # Test low cache hit rate
        should_verify = self.scheduler.should_trigger_verify(
            num_drafted_tokens=4,
            perplexity=1.0,
            cache_hit_rate=0.3,  # Below threshold
            max_draft_tokens=8
        )
        self.assertTrue(should_verify)


class TestCacheStrategy(unittest.TestCase):
    """Test cache replacement strategies"""
    
    def test_lru_strategy(self):
        """Test LRU replacement"""
        strategy = LRUCacheStrategy()
        
        experts = [ExpertID(0, i) for i in range(5)]
        
        # Simulate access pattern
        for exp in experts:
            strategy.on_insert(exp)
            strategy.on_access(exp)
        
        # Access some experts again
        strategy.on_access(experts[0])
        strategy.on_access(experts[2])
        
        # Select victim (should be expert 1 - least recently used)
        victim = strategy.select_victim(experts, set())
        self.assertEqual(victim, experts[1])
    
    def test_adaptive_strategy(self):
        """Test adaptive cache strategy"""
        strategy = AdaptiveCacheStrategy(recency_weight=0.5)
        
        experts = [ExpertID(0, i) for i in range(3)]
        
        for exp in experts:
            strategy.on_insert(exp)
        
        # Frequent access to expert 0
        for _ in range(10):
            strategy.on_access(experts[0])
        
        # Single access to others
        strategy.on_access(experts[1])
        strategy.on_access(experts[2])
        
        # Should evict expert with lowest score
        victim = strategy.select_victim(experts, set())
        # Expert 0 should not be victim due to high frequency
        self.assertNotEqual(victim, experts[0])


if __name__ == '__main__':
    unittest.main()