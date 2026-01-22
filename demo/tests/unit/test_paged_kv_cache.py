"""
测试 PagedKVCache 实现
"""

import os
import sys
import pytest
import torch

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.core.model import MoEConfig
from src.memory.paged_kv_cache import (
    Block, BlockManager, SequenceState, PagedKVCache
)


class TestBlock:
    """测试 Block 基本功能"""
    
    def test_block_creation(self):
        """测试 Block 创建"""
        block = Block(block_id=0)
        assert block.block_id == 0
        assert block.ref_count == 0
        print("✓ Block creation works")
    
    def test_block_reset(self):
        """测试 Block 重置"""
        block = Block(block_id=0)
        block.reset()
        assert block.ref_count == 1
        print("✓ Block reset works")
    
    def test_block_ref_count(self):
        """测试 Block 引用计数"""
        block = Block(block_id=0)
        block.reset()  # ref_count = 1
        
        block.inc_ref()
        assert block.ref_count == 2
        
        should_free = block.dec_ref()
        assert not should_free
        assert block.ref_count == 1
        
        should_free = block.dec_ref()
        assert should_free
        assert block.ref_count == 0
        
        print("✓ Block reference counting works")


class TestBlockManager:
    """测试 BlockManager"""
    
    def test_block_manager_creation(self):
        """测试 BlockManager 创建"""
        manager = BlockManager(num_blocks=10, block_size=256)
        assert manager.num_blocks == 10
        assert manager.block_size == 256
        assert manager.get_num_free_blocks() == 10
        print("✓ BlockManager creation works")
    
    def test_block_allocation(self):
        """测试 block 分配"""
        manager = BlockManager(num_blocks=5, block_size=256)
        
        # 分配 3 个 blocks
        block_ids = []
        for _ in range(3):
            block_id = manager.allocate_block()
            block_ids.append(block_id)
        
        assert len(set(block_ids)) == 3  # 应该是不同的 blocks
        assert manager.get_num_free_blocks() == 2
        print(f"✓ Allocated blocks: {block_ids}")
    
    def test_block_deallocation(self):
        """测试 block 释放"""
        manager = BlockManager(num_blocks=5, block_size=256)
        
        # 分配然后释放
        block_id = manager.allocate_block()
        assert manager.get_num_free_blocks() == 4
        
        # 需要先 dec_ref
        should_free = manager.dec_ref(block_id)
        assert should_free
        manager.deallocate_block(block_id)
        assert manager.get_num_free_blocks() == 5
        print("✓ Block deallocation works")
    
    def test_can_allocate(self):
        """测试 can_allocate"""
        manager = BlockManager(num_blocks=5, block_size=256)
        
        assert manager.can_allocate(5)
        assert not manager.can_allocate(6)
        
        # 分配 3 个
        for _ in range(3):
            manager.allocate_block()
        
        assert manager.can_allocate(2)
        assert not manager.can_allocate(3)
        print("✓ can_allocate works")


class TestSequenceState:
    """测试 SequenceState"""
    
    def test_sequence_state_creation(self):
        """测试 SequenceState 创建"""
        seq = SequenceState(seq_id=0, block_size=256)
        assert seq.seq_id == 0
        assert seq.block_size == 256
        assert seq.num_tokens == 0
        assert len(seq.block_table) == 0
        print("✓ SequenceState creation works")
    
    def test_num_blocks_calculation(self):
        """测试 num_blocks 计算"""
        seq = SequenceState(seq_id=0, block_size=256)
        
        seq.num_tokens = 0
        assert seq.num_blocks == 0
        
        seq.num_tokens = 1
        assert seq.num_blocks == 1
        
        seq.num_tokens = 256
        assert seq.num_blocks == 1
        
        seq.num_tokens = 257
        assert seq.num_blocks == 2
        
        seq.num_tokens = 512
        assert seq.num_blocks == 2
        
        print("✓ num_blocks calculation works")
    
    def test_last_block_num_tokens(self):
        """测试 last_block_num_tokens 计算"""
        seq = SequenceState(seq_id=0, block_size=256)
        
        seq.num_tokens = 0
        assert seq.last_block_num_tokens == 0
        
        seq.num_tokens = 100
        assert seq.last_block_num_tokens == 100
        
        seq.num_tokens = 256
        assert seq.last_block_num_tokens == 256
        
        seq.num_tokens = 300
        assert seq.last_block_num_tokens == 44
        
        print("✓ last_block_num_tokens calculation works")
    
    def test_get_slot_mapping(self):
        """测试 get_slot_mapping"""
        seq = SequenceState(seq_id=0, block_size=256)
        seq.block_table = [0, 1, 2]  # 3 blocks
        seq.num_tokens = 300
        
        # Get slots for first 3 tokens
        slots = seq.get_slot_mapping(start_pos=0, num_new_tokens=3)
        assert slots == [0, 1, 2]  # block 0, offsets 0,1,2
        
        # Get slots for tokens spanning block boundary
        slots = seq.get_slot_mapping(start_pos=254, num_new_tokens=4)
        # block 0 offset 254, 255, block 1 offset 0, 1
        assert slots == [254, 255, 256, 257]
        
        print("✓ get_slot_mapping works")
    
    def test_draft_verify_state(self):
        """测试 draft-verify 状态管理"""
        seq = SequenceState(seq_id=0, block_size=256)
        seq.num_tokens = 100
        seq.block_table = [0]
        
        # Start draft
        seq.start_draft()
        assert seq.is_in_draft
        assert seq.draft_start_num_tokens == 100
        assert seq.draft_start_num_blocks == 1
        
        # Simulate draft adding tokens
        seq.num_tokens = 150
        seq.block_table = [0]  # Still 1 block
        
        # Accept 30 tokens
        seq.accept_draft_tokens(num_accepted=30)
        assert not seq.is_in_draft
        assert seq.num_tokens == 130  # 100 + 30
        
        print("✓ draft-verify state management works")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPagedKVCache:
    """测试 PagedKVCache"""
    
    def test_paged_kv_cache_creation(self):
        """测试 PagedKVCache 创建"""
        # 创建一个小配置用于测试
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(
            config=config,
            block_size=16,
            gpu_memory_utilization=0.5
        )
        
        assert cache.block_size == 16
        assert cache.num_layers == 2
        assert cache.num_kv_heads == 2
        assert cache.head_dim == 32
        assert cache.num_blocks > 0
        
        print(f"✓ PagedKVCache created with {cache.num_blocks} blocks")
        print(f"  Memory usage: {cache.get_memory_usage_mb():.2f} MB")
    
    def test_add_and_remove_sequence(self):
        """测试添加和删除 sequence"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(config=config, block_size=16)
        
        # Add sequence
        success = cache.add_sequence(seq_id=0, prompt_len=50)
        assert success
        assert 0 in cache.sequences
        
        seq_state = cache.sequences[0]
        assert seq_state.num_tokens == 50
        assert len(seq_state.block_table) == 4  # ceil(50/16) = 4
        
        # Remove sequence
        cache.remove_sequence(seq_id=0)
        assert 0 not in cache.sequences
        
        print("✓ Add/remove sequence works")
    
    def test_append_token(self):
        """测试 append token"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(config=config, block_size=16)
        cache.add_sequence(seq_id=0, prompt_len=15)
        
        seq_state = cache.sequences[0]
        assert seq_state.num_tokens == 15
        assert len(seq_state.block_table) == 1
        
        # Append one token (should stay in same block)
        success = cache.append_token(seq_id=0)
        assert success
        assert seq_state.num_tokens == 16
        assert len(seq_state.block_table) == 1
        
        # Append another (should allocate new block)
        success = cache.append_token(seq_id=0)
        assert success
        assert seq_state.num_tokens == 17
        assert len(seq_state.block_table) == 2
        
        print("✓ append_token works")
    
    def test_store_and_retrieve_kv(self):
        """测试存储和检索 KV"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(config=config, block_size=16, dtype=torch.float16)
        cache.add_sequence(seq_id=0, prompt_len=10)
        
        # Create dummy KV states
        num_tokens = 10
        key = torch.randn(num_tokens, 2, 32, dtype=torch.float16, device='cuda')
        value = torch.randn(num_tokens, 2, 32, dtype=torch.float16, device='cuda')
        
        # Store KV for layer 0
        cache.store_kv(
            layer_idx=0,
            seq_id=0,
            key=key,
            value=value,
            start_pos=0
        )
        
        # Retrieve KV cache
        k_cache, v_cache = cache.get_kv_cache_for_layer(layer_idx=0)
        assert k_cache.shape[0] == cache.num_blocks
        assert k_cache.shape[1] == cache.block_size
        
        print("✓ store/retrieve KV works")
    
    def test_prefill_context(self):
        """测试 prefill 上下文"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(config=config, block_size=16)
        cache.add_sequence(seq_id=0, prompt_len=20)
        cache.add_sequence(seq_id=1, prompt_len=30)
        
        # Get prefill context
        context = cache.get_attention_context(seq_ids=[0, 1], is_prefill=True)
        
        assert 'cu_seqlens_q' in context
        assert 'cu_seqlens_k' in context
        assert 'max_seqlen_q' in context
        assert 'max_seqlen_k' in context
        assert 'slot_mapping' in context
        
        assert context['max_seqlen_q'] == 30
        assert context['max_seqlen_k'] == 30
        assert context['cu_seqlens_q'].tolist() == [0, 20, 50]
        
        print("✓ prefill context works")
    
    def test_decode_context(self):
        """测试 decode 上下文"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(config=config, block_size=16)
        cache.add_sequence(seq_id=0, prompt_len=20)
        cache.add_sequence(seq_id=1, prompt_len=30)
        
        # Get decode context
        context = cache.get_attention_context(seq_ids=[0, 1], is_prefill=False)
        
        assert 'context_lens' in context
        assert 'block_tables' in context
        assert 'slot_mapping' in context
        
        assert context['context_lens'].tolist() == [20, 30]
        assert context['block_tables'].shape[0] == 2  # 2 sequences
        
        print("✓ decode context works")
    
    def test_draft_verify_workflow(self):
        """测试 draft-verify 工作流"""
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            vocab_size=1000
        )
        
        cache = PagedKVCache(config=config, block_size=16)
        
        # Add draft sequence
        cache.add_sequence(seq_id=0, prompt_len=20)
        
        # Start draft
        cache.start_draft(seq_id=0)
        seq_state = cache.sequences[0]
        assert seq_state.is_in_draft
        assert seq_state.draft_start_num_tokens == 20
        
        # Simulate draft adding tokens
        for _ in range(10):
            cache.append_token(seq_id=0)
        
        assert seq_state.num_tokens == 30
        
        # Create verify sequence
        cache.add_sequence(seq_id=1, prompt_len=25)  # 20 + 5 accepted
        
        # Replace draft with verify (accept 5 tokens)
        cache.replace_draft_with_verify(
            seq_id=0,
            verify_seq_id=1,
            num_accepted_tokens=5
        )
        
        seq_state = cache.sequences[0]
        assert not seq_state.is_in_draft
        assert seq_state.num_tokens == 25  # 20 + 5
        assert 1 not in cache.sequences  # verify seq removed
        
        print("✓ draft-verify workflow works")


def run_quick_tests():
    """运行快速测试"""
    print("=" * 60)
    print("Testing PagedKVCache Implementation")
    print("=" * 60)
    
    # Test 1: Block
    print("\n--- Test 1: Block ---")
    block = Block(block_id=0)
    block.reset()
    block.inc_ref()
    assert block.ref_count == 2
    print("  ✓ Pass")
    
    # Test 2: BlockManager
    print("\n--- Test 2: BlockManager ---")
    manager = BlockManager(num_blocks=10, block_size=256)
    block_id = manager.allocate_block()
    assert manager.get_num_free_blocks() == 9
    should_free = manager.dec_ref(block_id)
    assert should_free
    manager.deallocate_block(block_id)
    assert manager.get_num_free_blocks() == 10
    print("  ✓ Pass")
    
    # Test 3: SequenceState
    print("\n--- Test 3: SequenceState ---")
    seq = SequenceState(seq_id=0, block_size=256)
    seq.num_tokens = 300
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 44
    print("  ✓ Pass")
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping GPU tests")
        return
    
    # Test 4: PagedKVCache
    print("\n--- Test 4: PagedKVCache ---")
    config = MoEConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_token=2,
        vocab_size=1000
    )
    cache = PagedKVCache(config=config, block_size=16)
    print(f"  Created with {cache.num_blocks} blocks")
    print(f"  Memory: {cache.get_memory_usage_mb():.2f} MB")
    print("  ✓ Pass")
    
    # Test 5: Sequence management
    print("\n--- Test 5: Sequence Management ---")
    success = cache.add_sequence(seq_id=0, prompt_len=50)
    assert success
    cache.append_token(seq_id=0)
    assert cache.sequences[0].num_tokens == 51
    cache.remove_sequence(seq_id=0)
    assert 0 not in cache.sequences
    print("  ✓ Pass")
    
    print("\n" + "=" * 60)
    print("All quick tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full pytest")
    args = parser.parse_args()
    
    if args.full:
        pytest.main([__file__, "-v", "-s"])
    else:
        run_quick_tests()

