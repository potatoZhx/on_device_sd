"""
Benchmark batch processing throughput at various batch sizes.
"""

import time
import torch
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

from src.api.inference import MoEInferenceEngine
from src.core.types import InferenceMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchThroughputBenchmark:
    """
    Benchmark throughput scaling with batch size.
    """
    
    def __init__(
        self,
        model_path: str,
        config_dir: str = "configs"
    ):
        self.model_path = model_path
        self.config_dir = config_dir
    
    def run_benchmark(
        self,
        batch_sizes: List[int],
        num_tokens_per_request: int = 50,
        num_trials: int = 3,
        mode: InferenceMode = InferenceMode.STANDARD
    ) -> Dict[int, Dict]:
        """
        Run throughput benchmark across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_tokens_per_request: Tokens to generate per request
            num_trials: Number of trials per batch size
            mode: Inference mode
        
        Returns:
            Dict mapping batch_size to metrics
        """
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Testing batch_size={batch_size}")
            logger.info(f"{'='*60}")
            
            # Initialize engine for this batch size
            engine = MoEInferenceEngine(
                model_path=self.model_path,
                config_dir=self.config_dir,
                max_batch_size=batch_size,
                default_mode=mode,
                enable_batch_processing=True
            )
            
            trial_results = []
            
            for trial in range(num_trials):
                logger.info(f"Trial {trial + 1}/{num_trials}")
                
                # Generate test prompts
                prompts = [f"Test prompt {i}" for i in range(batch_size)]
                
                # Submit all requests
                start_time = time.time()
                request_ids = [
                    engine.submit(prompt, max_new_tokens=num_tokens_per_request)
                    for prompt in prompts
                ]
                
                # Wait for all completions
                responses = [
                    engine.get_result(rid, timeout=120.0)
                    for rid in request_ids
                ]
                
                total_time = time.time() - start_time
                
                # Calculate metrics
                successful = [r for r in responses if r and r.success]
                total_tokens = sum(r.num_tokens_generated for r in successful)
                throughput = total_tokens / total_time
                
                trial_results.append({
                    'total_time': total_time,
                    'total_tokens': total_tokens,
                    'throughput': throughput,
                    'success_rate': len(successful) / batch_size
                })
                
                logger.info(f"  Throughput: {throughput:.2f} tokens/s")
            
            # Aggregate trial results
            avg_throughput = np.mean([r['throughput'] for r in trial_results])
            std_throughput = np.std([r['throughput'] for r in trial_results])
            avg_latency = np.mean([r['total_time'] for r in trial_results])
            
            results[batch_size] = {
                'avg_throughput': avg_throughput,
                'std_throughput': std_throughput,
                'avg_latency': avg_latency,
                'trials': trial_results
            }
            
            logger.info(f"\\nBatch size {batch_size} summary:")
            logger.info(f"  Average throughput: {avg_throughput:.2f} ± {std_throughput:.2f} tokens/s")
            logger.info(f"  Average latency: {avg_latency:.2f}s")
            
            # Cleanup
            engine.stop_batch_processor()
            del engine
        
        return results
    
    def plot_results(
        self,
        results: Dict[int, Dict],
        save_path: str = "batch_throughput.png"
    ) -> None:
        """
        Plot throughput scaling results.
        
        Args:
            results: Benchmark results
            save_path: Path to save plot
        """
        batch_sizes = sorted(results.keys())
        throughputs = [results[bs]['avg_throughput'] for bs in batch_sizes]
        throughput_stds = [results[bs]['std_throughput'] for bs in batch_sizes]
        latencies = [results[bs]['avg_latency'] for bs in batch_sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Throughput plot
        ax1.errorbar(
            batch_sizes,
            throughputs,
            yerr=throughput_stds,
            marker='o',
            capsize=5,
            linewidth=2,
            markersize=8
        )
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (tokens/s)')
        ax1.set_title('Throughput vs. Batch Size')
        ax1.grid(True, alpha=0.3)
        
        # Latency plot
        ax2.plot(batch_sizes, latencies, marker='s', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Latency (seconds)')
        ax2.set_title('Latency vs. Batch Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")


def main():
    """Run batch throughput benchmark"""
    
    benchmark = BatchThroughputBenchmark(
        model_path="path/to/model",
        config_dir="configs"
    )
    
    # Test batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    # Run for both modes
    for mode in [InferenceMode.STANDARD, InferenceMode.SPECULATIVE]:
        logger.info(f"\\n{'='*70}")
        logger.info(f"Benchmarking {mode.value.upper()} mode")
        logger.info(f"{'='*70}")
        
        results = benchmark.run_benchmark(
            batch_sizes=batch_sizes,
            num_tokens_per_request=50,
            num_trials=3,
            mode=mode
        )
        
        # Plot results
        benchmark.plot_results(
            results,
            save_path=f"batch_throughput_{mode.value}.png"
        )
        
        # Print summary
        print(f"\\n{mode.value.upper()} Mode Summary:")
        print(f"{'Batch Size':<12} {'Throughput':<15} {'Latency':<12}")
        print("-" * 40)
        for bs in sorted(results.keys()):
            tp = results[bs]['avg_throughput']
            lat = results[bs]['avg_latency']
            print(f"{bs:<12} {tp:>10.2f} tok/s  {lat:>8.2f}s")


if __name__ == "__main__":
    main()

