"""
Main heterogeneous inference engine
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import time

from ..utils.logger import get_logger
from ..utils.config import EngineConfig
from ..utils.metrics import MetricsCollector, Profiler
from .device_manager import DeviceManager
from .memory_manager import MemoryManager
from ..optimization.expert_scheduler import HeterogeneousScheduler


class HeterogeneousInferenceEngine:
    """Main heterogeneous inference engine for DeepSeek-V2-Lite"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = get_logger()
        
        # Initialize components
        self.device_manager = DeviceManager(config.device_config)
        self.memory_manager = MemoryManager(config.memory_config)
        self.scheduler = HeterogeneousScheduler(config.scheduler_config, self.device_manager, self.memory_manager)
        
        # Performance tracking
        self.metrics_collector = MetricsCollector()
        self.profiler = Profiler()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Initialize model
        self._load_and_optimize_model()
        
        self.logger.info("HeterogeneousInferenceEngine initialized successfully")
    
    def _load_and_optimize_model(self):
        """Load and optimize the model"""
        self.logger.info(f"Loading model from: {self.config.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        # Load model on meta device to avoid memory allocation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map="meta",  # Avoid allocating memory initially
            trust_remote_code=True
        )
        
        # Move non-expert components to GPU
        self._setup_model_devices()
        
        self.logger.info("Model loaded and optimized")
    
    def _setup_model_devices(self):
        """Setup model device placement"""
        # Move embedding and output layers to GPU
        if hasattr(self.model, 'model'):
            model = self.model.model
        else:
            model = self.model
        
        # Move embedding layer to GPU
        if hasattr(model, 'embed_tokens'):
            model.embed_tokens.to(self.device_manager.gpu_devices[0])
        
        # Move output layer to GPU
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head.to(self.device_manager.gpu_devices[0])
        
        # Move layer norm to GPU
        if hasattr(model, 'norm'):
            model.norm.to(self.device_manager.gpu_devices[0])
        
        # Move attention and other non-expert components to GPU
        if hasattr(model, 'layers'):
            for layer in model.layers:
                # Move attention components to GPU
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.to(self.device_manager.gpu_devices[0])
                
                # Move layer norms to GPU
                if hasattr(layer, 'input_layernorm'):
                    layer.input_layernorm.to(self.device_manager.gpu_devices[0])
                if hasattr(layer, 'post_attention_layernorm'):
                    layer.post_attention_layernorm.to(self.device_manager.gpu_devices[0])
                
                # Move MoE gate to GPU
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                    layer.block_sparse_moe.gate.to(self.device_manager.gpu_devices[0])
                
                # Experts will be managed dynamically by the scheduler
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
                    # Keep experts on CPU initially
                    for expert in layer.block_sparse_moe.experts:
                        expert.to(torch.device("cpu"))
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text using heterogeneous inference"""
        self.logger.info(f"Generating {max_new_tokens} tokens for prompt: {prompt[:50]}...")
        
        # Start timing
        self.metrics_collector.start_timing()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        original_length = input_ids.shape[1]
        
        # Prefill phase
        self.metrics_collector.start_prefill()
        kv_cache = self._prefill(input_ids)
        self.metrics_collector.end_prefill()
        
        # Decode phase
        self.metrics_collector.start_decode()
        generated_ids = self._decode(input_ids, kv_cache, max_new_tokens, temperature)
        self.metrics_collector.end_decode()
        
        # End timing
        num_generated = generated_ids.shape[1] - original_length
        self.metrics_collector.end_timing(num_generated)
        
        # Record metrics
        self.metrics_collector.record_memory_usage()
        self.metrics_collector.record_utilization()
        
        # Decode result
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        self.logger.info(f"Generation completed. Generated {num_generated} tokens")
        
        return result
    
    def _prefill(self, input_ids: torch.Tensor) -> Dict:
        """Execute prefill phase"""
        self.profiler.start("prefill")
        
        batch_size, seq_len = input_ids.shape
        hidden_size = self.model.config.hidden_size
        
        # Allocate KV cache
        device = self.device_manager.gpu_devices[0]
        kv_cache = self.memory_manager.allocate_kv_cache(batch_size, seq_len, hidden_size, device)
        
        # Move input to GPU
        input_ids = input_ids.to(device)
        
        # Execute prefill
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True
            )
        
        self.profiler.end("prefill")
        return outputs.past_key_values
    
    def _decode(self, input_ids: torch.Tensor, kv_cache: Dict, 
               max_new_tokens: int, temperature: float) -> torch.Tensor:
        """Execute decode phase"""
        self.profiler.start("decode")
        
        device = self.device_manager.gpu_devices[0]
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get next token input
            next_input = generated_ids[:, -1:]
            next_input = next_input.to(device)
            
            # Execute forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_input,
                    past_key_values=kv_cache,
                    use_cache=True,
                    return_dict=True
                )
            
            # Sample next token
            logits = outputs.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Update KV cache
            kv_cache = outputs.past_key_values
        
        self.profiler.end("decode")
        return generated_ids
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics_collector.get_metrics()
    
    def get_profiling_metrics(self) -> Dict:
        """Get profiling metrics"""
        return self.profiler.get_metrics()
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage information"""
        return self.memory_manager.get_memory_usage()
    
    def log_performance_summary(self):
        """Log performance summary"""
        self.metrics_collector.print_summary()
        
        # Log profiling metrics
        profiling_metrics = self.profiler.get_metrics()
        if profiling_metrics:
            self.logger.info("Profiling Metrics:")
            for name, metrics in profiling_metrics.items():
                self.logger.info(f"  {name}: {metrics['mean']:.3f}s (avg), {metrics['total']:.3f}s (total)")
        
        # Log memory usage
        memory_usage = self.get_memory_usage()
        self.logger.info("Memory Usage:")
        for key, value in memory_usage.items():
            self.logger.info(f"  {key}: {value:.2f} GB")
        
        # Log device status
        self.device_manager.log_memory_status()
        
        # Log scheduling stats
        self.scheduler.log_scheduling_stats()
    
    def clear_caches(self):
        """Clear all caches"""
        self.memory_manager.clear_all()
        self.logger.info("All caches cleared")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics_collector = MetricsCollector()
        self.profiler.reset()
        self.logger.info("Performance metrics reset") 