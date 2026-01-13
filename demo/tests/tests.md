┌─────────────────────────────────────────────────────────────────┐
│                      TEST COVERAGE MAP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Unit Tests (tests/unit/)                                       │
│  ├─ test_memory.py                                              │
│  │  ├─ ExpertCache operations                                   │
│  │  ├─ KVCache backup/restore                                   │
│  │  ├─ ParameterLoader placement                                │
│  │  └─ Cache eviction policies                                  │
│  │                                                               │
│  ├─ test_scheduling.py                                          │
│  │  ├─ Prefetch strategies                                      │
│  │  ├─ Draft schedulers                                         │
│  │  ├─ Cache replacement                                        │
│  │  └─ Expert selection                                         │
│  │                                                               │
│  ├─ test_operators.py                                           │
│  │  ├─ GPU operator correctness                                 │
│  │  ├─ CPU operator correctness                                 │
│  │  └─ Transfer operations                                      │
│  │                                                               │
│  └─ test_batch_manager.py                                       │
│     ├─ Batch formation                                          │
│     ├─ Priority ordering                                        │
│     ├─ Dynamic batching                                         │
│     └─ Request tracking                                         │
│                                                                  │
│  Integration Tests (tests/integration/)                         │
│  ├─ test_prefill.py                                             │
│  │  ├─ Single request prefill                                   │
│  │  └─ Batch prefill                                            │
│  │                                                               │
│  ├─ test_draft_verify.py                                        │
│  │  ├─ Draft phase correctness                                  │
│  │  ├─ Verify phase correctness                                 │
│  │  └─ Acceptance logic                                         │
│  │                                                               │
│  ├─ test_standard_decode.py                                     │
│  │  ├─ Standard single request                                  │
│  │  └─ Standard batch processing                                │
│  │                                                               │
│  ├─ test_batch_processing.py                                    │
│  │  ├─ Async submission                                         │
│  │  ├─ Concurrent requests                                      │
│  │  ├─ Priority handling                                        │
│  │  └─ Batch formation timing                                   │
│  │                                                               │
│  └─ test_end_to_end.py                                          │
│     ├─ Full generation pipeline                                 │
│     ├─ Mode comparison                                          │
│     └─ Multi-batch scenarios                                    │
│                                                                  │
│  Benchmark Tests (benchmarks/)                                  │
│  ├─ benchmark_prefill.py                                        │
│  ├─ benchmark_decode.py                                         │
│  ├─ benchmark_batch.py                                          │
│  ├─ compare_modes.py                                            │
│  └─ profile_memory.py                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘