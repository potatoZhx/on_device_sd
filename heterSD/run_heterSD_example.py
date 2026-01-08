#!/usr/bin/env python3
"""
Example script for running Heterogeneous Inference Engine with DeepSeek-V2-Lite
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add heterSD to path
sys.path.append(str(Path(__file__).parent / "heterSD"))

from heterSD.utils.config import EngineConfig
from heterSD.utils.logger import setup_logger
from heterSD.core.engine import HeterogeneousInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run Heterogeneous Inference Engine")
    parser.add_argument("--config", type=str, default="heterSD/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--prompt", type=str, 
                       default="Once upon a time, there was a magical forest where",
                       help="Input prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--log_file", type=str, default="heterSD_runtime.log",
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("heterSD_example", log_file=args.log_file)
    logger.info("Starting Heterogeneous Inference Engine Example")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available, will use CPU only")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = EngineConfig.from_yaml(args.config)
        
        # Initialize engine
        logger.info("Initializing Heterogeneous Inference Engine...")
        engine = HeterogeneousInferenceEngine(config)
        
        # Generate text
        logger.info(f"Generating text with prompt: {args.prompt}")
        logger.info(f"Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}")
        
        result = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Print results
        print("\n" + "="*60)
        print("GENERATION RESULT")
        print("="*60)
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {result}")
        print("="*60)
        
        # Log performance summary
        logger.info("Performance Summary:")
        engine.log_performance_summary()
        
        # Print performance metrics
        metrics = engine.get_performance_metrics()
        print("\nPERFORMANCE METRICS:")
        print(f"Prefill Time: {metrics.prefill_time:.3f}s")
        print(f"Decode Time: {metrics.decode_time:.3f}s")
        print(f"Total Time: {metrics.total_time:.3f}s")
        print(f"Tokens per Second: {metrics.tokens_per_second:.2f}")
        print(f"Expert Hit Rate: {metrics.expert_hit_rate:.3f}")
        
        # Print memory usage
        memory_usage = engine.get_memory_usage()
        print("\nMEMORY USAGE:")
        for key, value in memory_usage.items():
            print(f"  {key}: {value:.2f} GB")
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 