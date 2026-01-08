
# architecture

┌─────────────────────────────────────────────────────────────────┐
│                        Inference Engine                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Execution Orchestrator                        │ │
│  │  - Phase Controller (Prefill/Draft/Verify)                │ │
│  │  - Request Scheduler                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│  ┌──────────────────────────┼──────────────────────────────┐   │
│  │                          │                               │   │
│  ▼                          ▼                               ▼   │
│ ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│ │   Prefill    │  │   Draft Engine   │  │  Verify Engine  │   │
│ │   Engine     │  │                  │  │                 │   │
│ └──────────────┘  └──────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌──────────────────┐  ┌──────────────┐
│   Memory      │  │    Scheduling    │  │   Operator   │
│  Management   │  │    Subsystem     │  │   Executor   │
│               │  │                  │  │              │
│ - Parameter   │  │ - Expert         │  │ - GPU Ops    │
│   Loader      │  │   Prefetcher     │  │ - CPU Ops    │
│ - Expert      │  │ - Draft          │  │ - Transfer   │
│   Cache       │  │   Scheduler      │  │              │
│ - KV Cache    │  │ - Cache          │  │              │
│               │  │   Strategy       │  │              │
└───────────────┘  └──────────────────┘  └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
           ┌──────────────┐    ┌──────────────┐
           │  GPU Memory  │    │  CPU Memory  │
           │              │    │              │
           │ - Static     │    │ - Expert     │
           │   Params     │    │   Pool       │
           │ - Expert     │    │ - Staging    │
           │   Cache      │    │   Buffer     │
           │ - KV Cache   │    │              │
           └──────────────┘    └──────────────┘


# System Interaction FlowCharts
## Overall System Flow

┌─────────────────────────────────────────────────────────────┐
│                     User Request                            │
│              generate(prompt, max_tokens, ...)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Initialization (One-time)                   │
│  1. Load static params → GPU                                │
│  2. Load shared experts → GPU                               │
│  3. Load other experts → CPU/GPU per config                 │
│  4. Initialize expert cache                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREFILL PHASE                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  For each layer:                                       │ │
│  │    1. Compute routing scores                           │ │
│  │    2. Select top-k experts                             │ │
│  │    3. Prefetch experts for next layer                  │ │
│  │    4. Execute experts (CPU/GPU heterogeneous)          │ │
│  │    5. Store activation history                         │ │
│  └────────────────────────────────────────────────────────┘ │
│  Output: First generated token                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     DECODE PHASE                            │
│                  (Draft-Verify Loop)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌──────────────┐      ┌─────────────────┐
    │ DRAFT PHASE  │      │  VERIFY PHASE   │
    └──────────────┘      └─────────────────┘
            │                     │
            └──────────┬──────────┘│
                       ▼
            ┌──────────────────────┐
            │  Accept/Reject Tokens │
            └──────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Continue or Done?    │
            └──────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    Continue Loop            Return Results



## Draft Phase Detailed Flow
┌─────────────────────────────────────────────────────────────┐
│                    DRAFT PHASE START                         │
│  Input: Last generated token, KV cache                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Backup Current KV Cache                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Draft Loop (up to max_draft_tokens)                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  For each layer:                                       │ │
│  │                                                        │ │
│  │  1. Compute routing → select top-k experts            │ │
│  │                                                        │ │
│  │  2. Draft Scheduler selects:                          │ │
│  │     - Top-c experts for CPU execution                 │ │
│  │     - GPU substitutes for remaining experts           │ │
│  │                                                        │ │
│  │  3. Execute:                                           │ │
│  │     ┌─────────────────────────────────────┐          │ │
│  │     │ CPU experts  │ GPU cached │ GPU subs │          │ │
│  │     │   (top-c)    │  experts   │          │          │ │
│  │     └─────────────────────────────────────┘          │ │
│  │                                                        │ │
│  │  4. Track cache hits/misses                           │ │
│  │  5. Collect activation patterns                       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Sample next token → add to drafted_tokens                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Calculate Draft Metrics                          │
│  - Perplexity                                               │
│  - Cache hit rate                                           │
│  - CPU compute ratio                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│       Schedule Expert Transfers to GPU Cache                │
│  Based on activation frequency                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Check Verify Trigger                           │
│  - Max draft tokens reached?                                │
│  - Perplexity too high?                                     │
│  - Cache hit rate too low?                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    Trigger Verify          Continue Drafting

## Verify Phase Detailed Flow
┌─────────────────────────────────────────────────────────────┐
│                   VERIFY PHASE START                         │
│  Input: Draft tokens, Original prompt                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        Concatenate: Prompt + Draft Tokens                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Create Fresh KV Cache for Verification             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Full Model Forward Pass (like Prefill)              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  For each layer:                                       │ │
│  │    1. Self-attention with new KV cache                 │ │
│  │    2. Route to top-k experts                           │ │
│  │    3. Execute on available device (CPU/GPU)            │ │
│  │    4. Prefetch for next layer                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Generate logits for ALL token positions                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Acceptance Strategy                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  For each draft token:                                 │ │
│  │    Compare P_verify(token) with threshold              │ │
│  │    Accept if probability high enough                   │ │
│  │    Stop at first rejection                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Output: Accepted tokens, rejection position                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Replace Draft KV Cache with Verify Cache          │
│  Keep only accepted tokens in KV cache                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Return to Draft Phase                    │
│  Continue generating from last accepted token               │
└─────────────────────────────────────────────────────────────┘

## Expert Execution Decision Flow
┌─────────────────────────────────────────────────────────────┐
│            Expert Needs to be Executed                      │
│                 (Expert ID known)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ In GPU Cache?  │
              └────────┬───────┘
                       │
            ┌──────────┴──────────┐
            │                     │
           YES                   NO
            │                     │
            ▼                     ▼
    ┌──────────────┐      ┌─────────────────┐
    │ Execute on   │      │  In Draft Mode?  │
    │   GPU        │      └────────┬─────────┘
    │ (Cache Hit)  │               │
    └──────────────┘    ┌──────────┴──────────┐
                        │                     │
                       YES                   NO
                        │                     │
                        ▼                     ▼
              ┌──────────────────┐   ┌──────────────────┐
              │ Is it in top-c   │   │ Available on CPU?│
              │ selected by      │   └────────┬─────────┘
              │ Draft Scheduler? │            │
              └────────┬─────────┘   ┌────────┴────────┐
                       │             │                 │
            ┌──────────┴──────┐    YES               NO
            │                 │     │                 │
           YES               NO     ▼                 ▼
            │                 │  ┌────────┐    ┌──────────┐
            ▼                 │  │Execute │    │ Transfer │
    ┌──────────────┐          │  │on CPU  │    │ from CPU │
    │ Execute on   │          │  └────────┘    │ to GPU   │
    │   CPU        │          │                └────┬─────┘
    │ (top-c)      │          │                     │
    └──────────────┘          ▼                     ▼
                    ┌──────────────────┐    ┌──────────────┐
                    │ Find GPU         │    │ Execute on   │
                    │ Substitute from  │    │   GPU        │
                    │ Cached Experts   │    └──────────────┘
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Execute Substitute│
                    │ on GPU           │
                    └──────────────────┘




