import unittest
import torch
from src.core.types import ExpertID
from src.core.model import MoEConfig
from src.memory.expert_cache import ExpertCache
from src.memory.kv_cache import KVCache
from src.scheduling.cache_strategy import LRUCacheStrategy


class TestExpertCache(unittest.TestCase):
    """Test expert cache functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = LRUCacheStrategy()
        self.cache = ExpertCache(
            max_cache_size_gb=1.0,
            expert_size_mb=10.0,
            replacement_strategy=self.strategy
        )
        
        # Create dummy expert parameters
        self.expert_params = {
            'gate_proj': torch.randn(256, 128),
            'up_proj': torch.randn(256, 128),
            'down_proj': torch.randn(128, 256)
        }
    
    def test_cache_put_and_get(self):
        """Test basic cache insertion and retrieval"""
        expert_id = ExpertID(0, 0)
        
        # Put expert in cache
        success = self.cache.put(expert_id, self.expert_params)
        self.assertTrue(success)
        
        # Retrieve expert
        retrieved = self.cache.get(expert_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(set(retrieved.keys()), set(self.expert_params.keys()))
    
    def test_cache_eviction(self):
        """Test cache eviction when full"""
        # Fill cache to capacity
        num_experts = self.cache.max_experts
        expert_ids = [ExpertID(0, i) for i in range(num_experts)]
        
        for eid in expert_ids:
            self.cache.put(eid, self.expert_params)
        
        # Add one more, should trigger eviction
        new_expert = ExpertID(0, num_experts)
        success = self.cache.put(new_expert, self.expert_params)
        self.assertTrue(success)
        
        # First expert should be evicted (LRU)
        self.assertFalse(self.cache.is_cached(expert_ids[0]))
    
    def test_pinned_experts_not_evicted(self):
        """Test that pinned experts are not evicted"""
        pinned_expert = ExpertID(0, 0)
        
        # Put and pin expert
        self.cache.put(pinned_expert, self.expert_params, is_pinned=True)
        
        # Fill rest of cache
        for i in range(1, self.cache.max_experts + 5):
            self.cache.put(ExpertID(0, i), self.expert_params)
        
        # Pinned expert should still be in cache
        self.assertTrue(self.cache.is_cached(pinned_expert))


class TestKVCache(unittest.TestCase):
    """Test KV cache functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=512,
            vocab_size=1000,
            num_experts=8,
            num_experts_per_token=2
        )
        self.kv_cache = KVCache(self.config, max_batch_size=2)
    
    def test_kv_cache_append(self):
        """Test appending to KV cache"""
        batch_size = 1
        seq_len = 10
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads
        
        # Create dummy key-value tensors
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Append to cache
        full_keys, full_values = self.kv_cache.append(0, keys, values)
        
        # Check shapes
        self.assertEqual(full_keys.shape, keys.shape)
        self.assertEqual(full_values.shape, values.shape)
        
        # Check cache length updated
        self.assertEqual(self.kv_cache.current_length, 0)  # Not updated yet
        self.kv_cache.update_length(seq_len)
        self.assertEqual(self.kv_cache.current_length, seq_len)
    
    def test_kv_cache_backup_restore(self):
        """Test backup and restore functionality"""
        # Add some data
        keys = torch.randn(1, 8, 5, 32, device='cuda')
        values = torch.randn(1, 8, 5, 32, device='cuda')
        
        self.kv_cache.append(0, keys, values)
        self.kv_cache.update_length(5)
        
        # Backup
        self.kv_cache.backup_for_draft()
        
        # Modify cache
        more_keys = torch.randn(1, 8, 3, 32, device='cuda')
        more_values = torch.randn(1, 8, 3, 32, device='cuda')
        self.kv_cache.append(0, more_keys, more_values, start_pos=5)
        self.kv_cache.update_length(8)
        
        # Restore
        self.kv_cache.restore_from_backup()
        
        # Check length restored
        self.assertEqual(self.kv_cache.current_length, 5)


class TestParameterLoader(unittest.TestCase):
    """Test parameter loading"""
    
    def test_expert_location_tracking(self):
        """Test that expert locations are tracked correctly"""
        # This would require mock model files
        # Placeholder for actual implementation
        pass


if __name__ == '__main__':
    unittest.main()