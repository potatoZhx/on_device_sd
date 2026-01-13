"""
Tool for comparing standard vs. speculative decoding performance.
"""

import time
import torch
from typing import List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from src.api.inference import MoEInferenceEngine
from src.core.types import InferenceMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Results from comparing inference modes"""
    mode: InferenceMode
    num_requests: int
    total_tokens: int
    total_time_ms: float
    tokens_per_second: float
    average_latency_ms: float
    
    # Mode-specific metrics
    cache_hit_rate: float = 0.0
    acceptance_rate: float = 0.0  # For speculative
    speedup: float = 1.0


class InferenceModeComparator:
    """
    Compare performance between standard and speculative decoding.
    """
    
    def __init__(
        self,
        model_path: str,
        config_dir: str = "configs"
    ):
        self.model_path = model_path
        self.config_dir = config_dir
        
        # Initialize engines for each mode
        self.engines = {
            InferenceMode.STANDARD: self._init_engine(InferenceMode.STANDARD),
            InferenceMode.SPECULATIVE: self._init_engine(InferenceMode.SPECULATIVE)
        }
    
    def _init_engine(self, mode: InferenceMode) -> MoEInferenceEngine:
        """Initialize engine for specific mode"""
        logger.info(f"Initializing engine for {mode.value} mode")
        return MoEInferenceEngine(
            model_path=self.model_path,
            config_dir=self.config_dir,
            default_mode=mode,
            enable_batch_processing=False  # For fair comparison
        )
    
    def run_comparison(
        self,
        test_prompts: List[str],
        max_new_tokens: int = 100,
        warmup_runs: int = 3
    ) -> Dict[InferenceMode, ComparisonResult]:
        """
        Run comparison between modes.
        
        Args:
            test_prompts: List of test prompts
            max_new_tokens: Tokens to generate per prompt
            warmup_runs: Number of warmup runs
        
        Returns:
            Dict mapping mode to results
        """
        results = {}
        
        for mode in [InferenceMode.STANDARD, InferenceMode.SPECULATIVE]:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Testing {mode.value.upper()} mode")
            logger.info(f"{'='*60}")
            
            engine = self.engines[mode]
            
            # Warmup
            logger.info(f"Running {warmup_runs} warmup iterations...")
            for i in range(warmup_runs):
                _ = engine.generate(
                    test_prompts[0],
                    max_new_tokens=10,
                    mode=mode
                )
            
            # Actual test
            logger.info(f"Running test on {len(test_prompts)} prompts...")
            
            start_time = time.time()
            total_tokens = 0
            latencies = []
            
            for prompt in test_prompts:
                prompt_start = time.time()
                
                output_ids = engine.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    mode=mode
                )
                
                prompt_time = (time.time() - prompt_start) * 1000
                latencies.append(prompt_time)
                total_tokens += len(output_ids)
            
            total_time = (time.time() - start_time) * 1000
            
            # Get statistics
            stats = engine.get_statistics()
            cache_stats = stats.get('cache_stats', {})
            metrics = stats.get('metrics', {})
            
            # Create result
            result = ComparisonResult(
                mode=mode,
                num_requests=len(test_prompts),
                total_tokens=total_tokens,
                total_time_ms=total_time,
                tokens_per_second=total_tokens / (total_time / 1000),
                average_latency_ms=np.mean(latencies),
                cache_hit_rate=cache_stats.get('hit_rate', 0.0)
            )
            
            # Add mode-specific metrics
            if mode == InferenceMode.SPECULATIVE:
                # Calculate acceptance rate from metrics
                result.acceptance_rate = 0.0  # TODO: Extract from metrics
            
            results[mode] = result
            
            # Print results
            self._print_result(result)
        
        # Calculate speedup
        if InferenceMode.STANDARD in results and InferenceMode.SPECULATIVE in results:
            baseline_tps = results[InferenceMode.STANDARD].tokens_per_second
            speculative_tps = results[InferenceMode.SPECULATIVE].tokens_per_second
            results[InferenceMode.SPECULATIVE].speedup = speculative_tps / baseline_tps
        
        return results
    
    def _print_result(self, result: ComparisonResult) -> None:
        """Print formatted result"""
        print(f"\\nResults for {result.mode.value.upper()}:")
        print(f"  Total Tokens: {result.total_tokens}")
        print(f"  Total Time: {result.total_time_ms:.2f}ms")
        print(f"  Tokens/Second: {result.tokens_per_second:.2f}")
        print(f"  Average Latency: {result.average_latency_ms:.2f}ms")
        print(f"  Cache Hit Rate: {result.cache_hit_rate:.2%}")
        
        if result.mode == InferenceMode.SPECULATIVE:
            print(f"  Acceptance Rate: {result.acceptance_rate:.2%}")
            print(f"  Speedup: {result.speedup:.2f}x")
    
    def plot_comparison(
        self,
        results: Dict[InferenceMode, ComparisonResult],
        save_path: str = "comparison_results.png"
    ) -> None:
        """
        Plot comparison results.
        
        Args:
            results: Comparison results
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Standard vs. Speculative Decoding Comparison', fontsize=16)
        
        modes = list(results.keys())
        mode_names = [m.value.capitalize() for m in modes]
        
        # 1. Tokens per Second
        ax = axes[0, 0]
        tps_values = [results[m].tokens_per_second for m in modes]
        ax.bar(mode_names, tps_values, color=['blue', 'green'])
        ax.set_ylabel('Tokens/Second')
        ax.set_title('Throughput')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Average Latency
        ax = axes[0, 1]
        latency_values = [results[m].average_latency_ms for m in modes]
        ax.bar(mode_names, latency_values, color=['blue', 'green'])
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Average Latency per Request')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Cache Hit Rate
        ax = axes[1, 0]
        cache_values = [results[m].cache_hit_rate * 100 for m in modes]
        ax.bar(mode_names, cache_values, color=['blue', 'green'])
        ax.set_ylabel('Cache Hit Rate (%)')
        ax.set_title('Expert Cache Hit Rate')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Speedup (if speculative available)
        ax = axes[1, 1]
        if InferenceMode.SPECULATIVE in results:
            speedup = results[InferenceMode.SPECULATIVE].speedup
            ax.bar(['Speedup'], [speedup], color='green')
            ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Speculative Decoding Speedup')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")


def main():
    """Run comparison benchmark"""
    
    # Test prompts
    test_prompts = [
        "Once upon a time in a distant galaxy",
        "The future of artificial intelligence is",
        "In the year 2050, technology will have",
        "The most important scientific discovery of the 21st century",
        "Climate change requires immediate action because"
    ]
    
    # Initialize comparator
    comparator = InferenceModeComparator(
        model_path="path/to/model",
        config_dir="configs"
    )
    
    # Run comparison
    results = comparator.run_comparison(
        test_prompts=test_prompts,
        max_new_tokens=50,
        warmup_runs=2
    )
    
    # Plot results
    comparator.plot_comparison(results)
    
    # Print summary
    print("\\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    standard = results[InferenceMode.STANDARD]
    speculative = results[InferenceMode.SPECULATIVE]
    
    print(f"\\nStandard Decoding:")
    print(f"  Throughput: {standard.tokens_per_second:.2f} tokens/s")
    print(f"  Latency: {standard.average_latency_ms:.2f}ms")
    
    print(f"\\nSpeculative Decoding:")
    print(f"  Throughput: {speculative.tokens_per_second:.2f} tokens/s")
    print(f"  Latency: {speculative.average_latency_ms:.2f}ms")
    print(f"  Speedup: {speculative.speedup:.2f}x")
    print(f"  Cache Hit Rate: {speculative.cache_hit_rate:.2%}")


if __name__ == "__main__":
    main()