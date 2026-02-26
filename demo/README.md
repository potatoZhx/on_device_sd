test model path: /zx_data1/models/Qwen--Qwen3-30B-A3B-Base
# prompt
Your task is to build a high-performance **CPU-GPU heterogeneous inference system for on-device MoE (Mixture of Experts) LLMs** (targeting an initial implementation for offline inference scenarios with typically small batch sizes).

**The system workflow is as follows:**

**Parameter Loading:** Initially, load static parameters (e.g., non-FFN layer parameters and shared expert parameters) into GPU memory. Then, based on a configuration file (or a random strategy if no configuration is provided), load expert parameters into either CPU memory or GPU memory.
**Online Execution:** The system receives input and runs full-model CPU-GPU inference during the **Prefill Phase**.
When executing the current layer, use a simple prefetch prediction strategy (reserving an interface for improved strategies) to fetch the expert parameters required for the next layer from the CPU in advance.
For parameters not successfully prefetched, use a scheduling strategy (reserving a scheduling interface) to decide whether to load them into the GPU or perform inference locally on the CPU.
**Decode Phase:** Alternate between **Draft** and **Verify** steps.
**Draft Step:** The GPU acts as the primary computing device. Inference is accelerated by minimizing CPU participation and parameter transfer. Specifically, if experts activated on the CPU, use a **Draft Scheduler** to select less than top-*c* experts (initially setting *c=2*) to run locally on the CPU. For the remaining experts, select substitute experts present in GPU memory to run on the GPU.
Simultaneously, the Draft Scheduler collects expert replacement data and metrics (like perplexity) to determine when to switch to the Verify step.
The Draft Scheduler must also select a set of experts based on the activation results during drafting to be transferred to the GPU in parallel.
The replacement of the expert cache in the GPU is managed by a dedicated **Expert Cache Strategy** (interfaces must be reserved for both the Draft Scheduler and Expert Cache replacement to facilitate future improvements).
**Verify Step:** Concatenate all inputs from the previous draft round and the output of the final draft round into a sequence as input. Execute full-model CPU-GPU inference using the same strategy as the Prefill phase.
Crucially, you must replace the KV cache generated during the draft with the KV cache generated during verification.
After obtaining the output, execute the acceptance strategy for speculative sampling and begin the next draft round.
**Note:** Existing operators can be used for operator implementation initially, but interfaces for improvements must be reserved.

