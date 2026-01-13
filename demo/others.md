8. Key Design Decisions & Rationale
8.1 Modular Architecture

Decision: Separate memory, scheduling, and execution into distinct modules
Rationale: Enables independent testing, easy replacement of strategies, and clear responsibility boundaries

8.2 Strategy Pattern for Algorithms

Decision: All scheduling/caching algorithms use abstract base classes
Rationale: Allows users to plug in custom strategies without modifying core code

8.3 Two-Level Cache Management

Decision: Separate tracking of expert location vs. GPU cache management
Rationale: Parameter loader tracks where experts are stored, while expert cache manages GPU memory dynamically

8.4 Async Transfer Design

Decision: Use CUDA streams for non-blocking transfers
Rationale: Overlap computation and data movement for better performance

8.5 KV Cache Backup/Restore

Decision: Explicit backup before draft, restore on rejection
Rationale: Simpler than trying to compute correct KV states after rejection

8.6 Metrics Collection

Decision: Centralized metrics collector with phase tracking
Rationale: Essential for debugging and optimization; minimal performance overhead

8.7 Configuration-Driven Design

Decision: YAML configurations for all major settings
Rationale: Easy experimentation without code changes


9. Future Extensions
9.1 Planned Improvements

Advanced Prefetching:

Neural network-based predictor
Multi-layer look-ahead


Optimized Operators:

Custom CUDA kernels for expert FFN
Fused operations (router + top-k)


Batching Support:

Multi-request batching
Dynamic batching with varying expert activations


Multi-GPU Support:

Expert distribution across GPUs
Pipeline parallelism


Quantization:

INT8/INT4 expert weights
Mixed precision inference


Better Draft Strategies:

Learning-based substitution
Confidence-aware drafting



9.2 Extension Points
The system provides clear extension points:

PrefetchStrategy: Custom prefetching logic
DraftSchedulingStrategy: Custom draft scheduling
CacheReplacementStrategy: Custom cache policies
AcceptanceStrategy: Custom acceptance criteria
Operators: Custom GPU/CPU implementations


10. Performance Considerations
10.1 Bottlenecks to Monitor

Expert Transfers: CPU→GPU bandwidth
Cache Misses: Excessive CPU computation
Draft Quality: Low acceptance rates
KV Cache Size: Memory consumption

10.2 Optimization Opportunities

Batched Transfers: Group multiple expert transfers
Prefetch Accuracy: Better prediction = fewer stalls
Cache Size Tuning: Balance memory vs. hit rate
Draft Length: Tune max_draft_tokens based on acceptance


11. Summary
This design provides a comprehensive, extensible system for CPU-GPU heterogeneous MoE inference with speculative decoding. Key features:
✅ Modular architecture with clear separation of concerns
✅ Pluggable strategies for all major algorithms
✅ Complete implementation plan with interfaces and pseudo-code
✅ Comprehensive testing strategy
✅ Production-ready features (metrics, logging, configuration)
✅ Clear extension points for future improvements
The system balances performance (async transfers, caching, prefetching) with maintainability (modular design, extensive documentation) and flexibility (strategy pattern, configuration-driven).



# Performance Tuning Guide

# For Maximum Throughput (Speculative Mode):
expert_cache_size_gb: 24.0  # Larger cache
draft_scheduler: "adaptive"
acceptance_threshold: 0.6    # Lower threshold = more drafts accepted
max_draft_tokens: 12         # More tokens per draft
max_batch_size: 64          # Larger batches

# For Minimum Latency (Standard Mode):
expert_cache_size_gb: 16.0
max_batch_size: 8           # Smaller batches
prefetch_strategy: "history_based"  # Better predictions
max_concurrent_transfers: 8  # More parallel transfers

# For Memory-Constrained Environments:
expert_cache_size_gb: 4.0   # Smaller cache
max_batch_size: 4           # Smaller batches
cache_strategy: "adaptive"   # Better cache utilization
pin_shared_experts: false   # More flexible cache

# For Quality-Critical Applications:
default_mode: "standard"    # Use standard mode
acceptance_threshold: 0.9   # Very conservative
verify_threshold_perplexity: 1.2  # Tight perplexity control

# For Development/Testing:
log_level: "DEBUG"
enable_profiling: true
max_batch_size: 2