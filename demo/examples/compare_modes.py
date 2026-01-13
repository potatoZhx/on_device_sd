"""
Example demonstrating comparison between standard and speculative modes.
"""

from src.api.inference import MoEInferenceEngine
from src.core.types import InferenceMode
import time


def run_comparison_test():
    """Run a simple comparison test"""
    
    # Initialize engine
    engine = MoEInferenceEngine(
        model_path="path/to/model",
        config_dir="configs",
        enable_batch_processing=False  # For fair comparison
    )
    
    # Test prompts
    prompts = [
        "The evolution of technology has",
        "In the next decade, we will see",
        "The most significant challenge facing",
    ]
    
    print("="*70)
    print("INFERENCE MODE COMPARISON")
    print("="*70)
    
    for mode in [InferenceMode.STANDARD, InferenceMode.SPECULATIVE]:
        print(f"\\n{'='*70}")
        print(f"Testing {mode.value.upper()} Mode")
        print(f"{'='*70}\\n")
        
        total_tokens = 0
        total_time = 0.0
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Prompt {i}/{len(prompts)}: '{prompt}'")
            
            start = time.time()
            output = engine.generate(
                prompt,
                max_new_tokens=30,
                temperature=0.7,
                mode=mode
            )
            elapsed = time.time() - start
            
            total_tokens += len(output)
            total_time += elapsed
            
            print(f"  Generated: {len(output)} tokens")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {len(output)/elapsed:.1f} tok/s\\n")
        
        # Summary for this mode
        avg_throughput = total_tokens / total_time
        print(f"Mode Summary:")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Throughput: {avg_throughput:.1f} tok/s")
    
    # Final statistics
    print(f"\\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}\\n")
    engine.print_statistics()


if __name__ == "__main__":
    run_comparison_test()