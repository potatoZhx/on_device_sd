# Heterogeneous Inference Engine (HeterSD)

A flexible framework for CPU+GPU mixed inference with dynamic expert offloading, designed specifically for DeepSeek-V2-Lite and other MoE models.

## Features

- **Heterogeneous Computing**: Seamless CPU+GPU mixed inference
- **Dynamic Expert Offloading**: Intelligent expert placement based on usage patterns
- **Memory Management**: Efficient KV cache and expert cache management
- **Performance Monitoring**: Real-time performance metrics and profiling
- **Configurable**: YAML-based configuration system
- **Modular Design**: Extensible architecture for custom optimizations

## Architecture

```
heterSD/
├── core/                    # Core inference components
│   ├── engine.py           # Main inference engine
│   ├── device_manager.py   # Device management
│   └── memory_manager.py   # Memory management
├── optimization/           # Optimization strategies
│   └── expert_scheduler.py # Expert scheduling
├── utils/                  # Utility modules
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging system
│   └── metrics.py         # Performance metrics
└── config.yaml            # Default configuration
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd on_device_sd

# Install dependencies
pip install torch transformers pyyaml psutil
```

### 2. Test the Installation

```bash
python test_heterSD.py
```

### 3. Run Inference

```bash
# Basic usage
python run_heterSD_example.py

# Custom parameters
python run_heterSD_example.py \
    --prompt "Once upon a time, there was a magical forest where" \
    --max_tokens 100 \
    --temperature 0.7
```

## Configuration

The engine is configured via YAML files. Key configuration options:

```yaml
engine:
  model_path: "deepseek-ai/DeepSeek-V2-Lite"
  device_config:
    gpu_memory_limit: 24  # GB
    cpu_memory_limit: 64  # GB
    gpu_devices: [0]
  
  scheduler_config:
    type: "popularity_based"
    expert_cache_size: 100
    top_k_experts: 2
  
  memory_config:
    max_kv_cache_size: 8  # GB
    max_expert_cache_size: 4  # GB
```

## Usage Examples

### Basic Text Generation

```python
from heterSD.utils.config import EngineConfig
from heterSD.core.engine import HeterogeneousInferenceEngine

# Load configuration
config = EngineConfig.from_yaml("heterSD/config.yaml")

# Initialize engine
engine = HeterogeneousInferenceEngine(config)

# Generate text
result = engine.generate(
    prompt="Once upon a time, there was a magical forest where",
    max_new_tokens=100,
    temperature=0.7
)

print(result)
```

### Performance Monitoring

```python
# Get performance metrics
metrics = engine.get_performance_metrics()
print(f"Tokens per second: {metrics.tokens_per_second:.2f}")
print(f"Expert hit rate: {metrics.expert_hit_rate:.3f}")

# Get memory usage
memory_usage = engine.get_memory_usage()
print(f"KV cache: {memory_usage['kv_cache_gb']:.2f} GB")
print(f"Expert cache: {memory_usage['expert_cache_gb']:.2f} GB")

# Log detailed performance summary
engine.log_performance_summary()
```

## Key Components

### 1. Device Manager

Manages CPU and GPU resources, handles tensor transfers, and provides device selection logic.

### 2. Memory Manager

Manages KV cache and expert cache with LRU eviction policies.

### 3. Expert Scheduler

Implements popularity-based expert scheduling with dynamic offloading between CPU and GPU.

### 4. Performance Metrics

Tracks prefill/decode times, memory usage, expert hit rates, and device utilization.

## Dynamic Offloading Strategy

The engine implements intelligent expert offloading:

1. **Prefill Phase**: Analyzes expert usage patterns and updates popularity statistics
2. **Runtime Scheduling**: Dynamically moves experts between CPU and GPU based on:
   - Usage frequency
   - Memory availability
   - Compute intensity
3. **Migration Optimization**: Batch migrations with cooldown periods to minimize overhead

## Performance Optimization

- **Asynchronous Transfers**: Non-blocking GPU-CPU data transfers
- **Batch Processing**: Efficient expert execution on each device
- **Memory Pooling**: Pre-allocated memory pools to reduce allocation overhead
- **LRU Caching**: Intelligent cache eviction for both KV and expert caches

## Monitoring and Debugging

The engine provides comprehensive monitoring:

```python
# Enable detailed logging
from heterSD.utils.logger import setup_logger
logger = setup_logger("heterSD", log_file="runtime.log")

# Monitor performance
engine.log_performance_summary()

# Clear caches if needed
engine.clear_caches()

# Reset metrics
engine.reset_metrics()
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for GPU support)
- 24GB+ GPU memory (recommended)
- 64GB+ system memory (recommended)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 