#!/usr/bin/env python3
"""
Simple test script for Heterogeneous Inference Engine
"""

import sys
import torch
from pathlib import Path

# Add heterSD to path
sys.path.append(str(Path(__file__).parent / "heterSD"))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from heterSD.utils.config import EngineConfig, create_default_config
        print("✓ Config module imported successfully")
        
        from heterSD.utils.logger import setup_logger
        print("✓ Logger module imported successfully")
        
        from heterSD.utils.metrics import MetricsCollector, Profiler
        print("✓ Metrics module imported successfully")
        
        from heterSD.core.device_manager import DeviceManager
        print("✓ Device manager imported successfully")
        
        from heterSD.core.memory_manager import MemoryManager
        print("✓ Memory manager imported successfully")
        
        from heterSD.optimization.expert_scheduler import HeterogeneousScheduler
        print("✓ Expert scheduler imported successfully")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    Memory: {memory:.1f} GB")
    else:
        print("✗ CUDA not available")
    
    return torch.cuda.is_available()

def test_config():
    """Test configuration creation"""
    print("\nTesting configuration...")
    
    try:
        from heterSD.utils.config import create_default_config
        
        # Create default config
        config = create_default_config("deepseek-ai/DeepSeek-V2-Lite")
        print("✓ Default configuration created successfully")
        
        # Test config properties
        print(f"  Model path: {config.model_path}")
        print(f"  GPU memory limit: {config.device_config.gpu_memory_limit} GB")
        print(f"  CPU memory limit: {config.device_config.cpu_memory_limit} GB")
        print(f"  Expert cache size: {config.scheduler_config.expert_cache_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_components():
    """Test component initialization"""
    print("\nTesting component initialization...")
    
    try:
        from heterSD.utils.config import create_default_config
        from heterSD.core.device_manager import DeviceManager
        from heterSD.core.memory_manager import MemoryManager
        from heterSD.optimization.expert_scheduler import HeterogeneousScheduler
        
        # Create config
        config = create_default_config("deepseek-ai/DeepSeek-V2-Lite")
        
        # Test device manager
        device_manager = DeviceManager(config.device_config)
        print("✓ Device manager initialized")
        
        # Test memory manager
        memory_manager = MemoryManager(config.memory_config)
        print("✓ Memory manager initialized")
        
        # Test scheduler
        scheduler = HeterogeneousScheduler(config.scheduler_config, device_manager, memory_manager)
        print("✓ Scheduler initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("Heterogeneous Inference Engine Test Suite")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Configuration Test", test_config),
        ("Component Test", test_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Heterogeneous Inference Engine is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 