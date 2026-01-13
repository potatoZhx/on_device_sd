"""
Example demonstrating batch processing capabilities.
"""

import time
from src.api.inference import MoEInferenceEngine
from src.core.types import InferenceMode


def main():
    # Initialize engine with batch processing
    engine = MoEInferenceEngine(
        model_path="path/to/model",
        config_dir="configs",
        max_batch_size=8,
        enable_batch_processing=True,
        default_mode=InferenceMode.SPECULATIVE
    )
    
    print("MoE Inference Engine initialized with batch processing")
    print(f"Max batch size: 8")
    print()
    
    # Example 1: Submit multiple requests asynchronously
    print("="*60)
    print("Example 1: Asynchronous Batch Processing")
    print("="*60)
    
    prompts = [
        "The future of artificial intelligence will",
        "Climate change is affecting",
        "The most important technological advancement",
        "In the year 2050, humanity will",
        "The secret to happiness is",
    ]
    
    # Submit all requests
    print(f"Submitting {len(prompts)} requests...")
    request_ids = []
    
    for i, prompt in enumerate(prompts):
        request_id = engine.submit(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.8,
            priority=i  # Higher index = higher priority
        )
        request_ids.append(request_id)
        print(f"  Submitted request {i+1}: {request_id[:16]}...")
    
    print(f"\\nWaiting for results...")
    
    # Collect results
    results = []
    for i, request_id in enumerate(request_ids):
        print(f"  Waiting for request {i+1}...")
        response = engine.get_result(request_id, timeout=60.0)
        
        if response and response.success:
            print(f"    ✓ Completed: {len(response.generated_ids)} tokens, "
                  f"{response.tokens_per_second:.1f} tok/s")
            results.append(response)
        else:
            print(f"    ✗ Failed")
    
    print(f"\\nCompleted {len(results)}/{len(prompts)} requests")
    
    # Example 2: Synchronous generation with mode comparison
    print("\\n" + "="*60)
    print("Example 2: Mode Comparison")
    print("="*60)
    
    test_prompt = "Artificial intelligence is transforming"
    
    # Standard mode
    print("\\nGenerating with STANDARD mode...")
    start = time.time()
    standard_output = engine.generate(
        test_prompt,
        max_new_tokens=50,
        mode=InferenceMode.STANDARD,
        temperature=0.8
    )
    standard_time = time.time() - start
    
    print(f"  Generated {len(standard_output)} tokens in {standard_time:.2f}s")
    print(f"  Throughput: {len(standard_output)/standard_time:.1f} tok/s")
    
    # Speculative mode
    print("\\nGenerating with SPECULATIVE mode...")
    start = time.time()
    speculative_output = engine.generate(
        test_prompt,
        max_new_tokens=50,
        mode=InferenceMode.SPECULATIVE,
        temperature=0.8
    )
    speculative_time = time.time() - start
    
    print(f"  Generated {len(speculative_output)} tokens in {speculative_time:.2f}s")
    print(f"  Throughput: {len(speculative_output)/speculative_time:.1f} tok/s")
    print(f"  Speedup: {standard_time/speculative_time:.2f}x")
    
    # Print statistics
    print("\\n" + "="*60)
    print("System Statistics")
    print("="*60)
    engine.print_statistics()
    
    # Cleanup
    engine.stop_batch_processor()
    print("\\nEngine stopped.")


if __name__ == "__main__":
    main()