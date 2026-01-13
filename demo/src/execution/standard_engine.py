from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from ..core.types import BatchedRequest, GenerationConfig
from ..core.model import MoEConfig
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..memory.kv_cache import KVCache
from ..scheduling.prefetcher import ExpertPrefetcher
from ..operators.gpu_operators import GPUOperators
from ..operators.cpu_operators import CPUOperators
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


class StandardDecodeEngine:
    """
    Standard autoregressive decoding engine.
    No speculative decoding - just straightforward token-by-token generation.
    Serves as a baseline for comparison.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: Optional[ExpertPrefetcher] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.metrics = metrics or MetricsCollector()
        
        # Operators
        self.gpu_ops = GPUOperators(config)
        self.cpu_ops = CPUOperators(config)
        
        logger.info("StandardDecodeEngine initialized")
    
    def generate_batch(
        self,
        batch: BatchedRequest
    ) -> Dict:
        """
        Generate tokens for a batch using standard autoregressive decoding.
        
        Args:
            batch: Batched inference requests
        
        Returns:
            Dict with generated sequences and statistics
        """
        self.metrics.start_phase('standard_generation')
        batch_size = len(batch.requests)
        
        # Initialize KV caches for each request in batch
        kv_caches = [KVCache(self.config, max_batch_size=1) for _ in range(batch_size)]
        
        # Phase 1: Prefill
        logger.info(f"Prefill phase for batch {batch.batch_id} ({batch_size} requests)")
        self.metrics.start_phase('prefill')
        
        prefill_output = self._prefill_batch(batch, kv_caches)
        
        self.metrics.end_phase('prefill')
        
        # Phase 2: Decode
        logger.info(f"Decode phase for batch {batch.batch_id}")
        self.metrics.start_phase('decode')
        
        generated_sequences = self._decode_batch(
            batch=batch,
            kv_caches=kv_caches,
            initial_tokens=prefill_output['next_token_ids']
        )
        
        self.metrics.end_phase('decode')
        self.metrics.end_phase('standard_generation')
        
        return {
            'generated_sequences': generated_sequences,
            'statistics': self._get_statistics()
        }
    
    def _prefill_batch(
        self,
        batch: BatchedRequest,
        kv_caches: List[KVCache]
    ) -> Dict:
        """
        Prefill phase for batch.
        Process all input tokens in parallel.
        
        Args:
            batch: Batched requests
            kv_caches: KV cache for each request
        
        Returns:
            Dict with next token IDs for each request
        """
        input_ids = batch.input_ids.cuda()  # [batch_size, seq_len]
        attention_mask = batch.attention_mask.cuda()
        
        # Embedding
        hidden_states = self._embed_tokens(input_ids)
        
        # Process each layer
        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states = self._process_layer_batch(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                kv_caches=kv_caches,
                is_prefill=True
            )
        
        # Final layers
        hidden_states = self._final_layernorm(hidden_states)
        logits = self._lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # Get next token for each request (last non-padding position)
        next_token_ids = []
        for i, request in enumerate(batch.requests):
            last_pos = len(request.input_ids) - 1
            next_token_logits = logits[i, last_pos, :]
            
            # Sample token
            next_token = self._sample_token(
                next_token_logits.unsqueeze(0),
                request.generation_config
            )
            next_token_ids.append(next_token)
            
            # Update KV cache length
            kv_caches[i].update_length(len(request.input_ids))
        
        next_token_ids = torch.tensor(next_token_ids, dtype=torch.long, device='cuda')
        
        return {'next_token_ids': next_token_ids}
    
    def _decode_batch(
        self,
        batch: BatchedRequest,
        kv_caches: List[KVCache],
        initial_tokens: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Autoregressive decode phase for batch.
        Generate tokens one at a time for each active request.
        
        Args:
            batch: Batched requests
            kv_caches: KV caches for each request
            initial_tokens: First generated token for each request
        
        Returns:
            List of generated sequences for each request
        """
        batch_size = len(batch.requests)
        generated_sequences = [[] for _ in range(batch_size)]
        current_tokens = initial_tokens
        
        # Add initial tokens
        for i in range(batch_size):
            generated_sequences[i].append(current_tokens[i].item())
        
        # Maximum generation length across all requests
        max_new_tokens = max(
            req.generation_config.max_new_tokens 
            for req in batch.requests
        )
        
        # Generate loop
        for step in range(max_new_tokens - 1):
            # Check which requests are still active
            active_mask = torch.tensor([
                len(generated_sequences[i]) < batch.requests[i].generation_config.max_new_tokens
                and generated_sequences[i][-1] != batch.requests[i].generation_config.eos_token_id
                for i in range(batch_size)
            ], dtype=torch.bool)
            
            if not active_mask.any():
                break
            
            # Forward pass for active requests
            next_tokens = self._decode_step_batch(
                current_tokens=current_tokens,
                kv_caches=kv_caches,
                active_mask=active_mask,
                batch=batch
            )
            
            # Update generated sequences
            for i in range(batch_size):
                if active_mask[i]:
                    generated_sequences[i].append(next_tokens[i].item())
            
            current_tokens = next_tokens
        
        # Convert to tensors
        return [torch.tensor(seq, dtype=torch.long) for seq in generated_sequences]
    
    def _decode_step_batch(
        self,
        current_tokens: torch.Tensor,
        kv_caches: List[KVCache],
        active_mask: torch.Tensor,
        batch: BatchedRequest
    ) -> torch.Tensor:
        """
        Single decode step for batch.
        
        Args:
            current_tokens: Current token for each request [batch_size]
            kv_caches: KV caches
            active_mask: Which requests are still active [batch_size]
            batch: Batch information
        
        Returns:
            Next tokens [batch_size]
        """
        batch_size = len(batch.requests)
        
        # Prepare input [batch_size, 1]
        input_ids = current_tokens.unsqueeze(1).cuda()
        
        # Embedding
        hidden_states = self._embed_tokens(input_ids)
        
        # Process layers
        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states = self._process_layer_batch(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=None,  # For decode, no masking needed
                kv_caches=kv_caches,
                is_prefill=False
            )
        
        # Final layers
        hidden_states = self._final_layernorm(hidden_states)
        logits = self._lm_head(hidden_states)  # [batch_size, 1, vocab_size]
        
        # Sample next token for each request
        next_tokens = []
        for i in range(batch_size):
            if active_mask[i]:
                next_token = self._sample_token(
                    logits[i],
                    batch.requests[i].generation_config
                )
                kv_caches[i].update_length(kv_caches[i].current_length + 1)
            else:
                # Inactive requests: just repeat current token (will be masked out)
                next_token = current_tokens[i]
            
            next_tokens.append(next_token)
        
        return torch.tensor(next_tokens, dtype=torch.long, device='cuda')
    
    def _process_layer_batch(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_caches: List[KVCache],
        is_prefill: bool
    ) -> torch.Tensor:
        """
        Process a single layer for batched inputs.
        
        Args:
            layer_idx: Layer index
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            kv_caches: KV caches for each request
            is_prefill: Whether this is prefill phase
        
        Returns:
            Output hidden states
        """
        batch_size = hidden_states.shape[0]
        residual = hidden_states
        
        # Pre-attention layer norm
        hidden_states = self._layernorm_batch(hidden_states, layer_idx, 'input_layernorm')
        
        # Self-attention (process each request separately for KV cache)
        attn_outputs = []
        for i in range(batch_size):
            attn_out = self._self_attention(
                hidden_states[i:i+1],
                layer_idx,
                kv_caches[i]
            )
            attn_outputs.append(attn_out)
        
        hidden_states = torch.cat(attn_outputs, dim=0)
        
        # Residual
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        # Post-attention layer norm
        hidden_states = self._layernorm_batch(hidden_states, layer_idx, 'post_attention_layernorm')
        
        # MoE FFN (process batch together)
        hidden_states = self._moe_forward_batch(
            hidden_states=hidden_states,
            layer_idx=layer_idx
        )
        
        # Residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _moe_forward_batch(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        MoE forward pass for batch.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            layer_idx: Current layer
        
        Returns:
            MoE output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        # Router
        router_weight = self.parameter_loader.static_params_gpu[f"layer_{layer_idx}.router"]
        routing_scores = self.gpu_ops.router(hidden_states, router_weight)
        
        # Select top-k experts
        top_k = self.config.num_experts_per_token
        topk_scores, topk_indices = torch.topk(routing_scores, top_k, dim=-1)
        topk_scores = torch.softmax(topk_scores, dim=-1)
        
        # Execute experts (heterogeneous CPU/GPU)
        expert_outputs = self._execute_experts_batch(
            hidden_states=flat_hidden,
            layer_idx=layer_idx,
            topk_indices=topk_indices,
            topk_scores=topk_scores
        )
        
        return expert_outputs.view(batch_size, seq_len, hidden_size)
    
    def _execute_experts_batch(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor
    ) -> torch.Tensor:
        """Execute experts for batch (similar to prefill engine)"""
        from ..core.types import ExpertID, DeviceType
        
        batch_seq, hidden_size = hidden_states.shape
        final_output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.config.num_experts):
            expert_id = ExpertID(layer_idx, expert_idx)
            
            # Find tokens using this expert
            expert_mask = (topk_indices == expert_idx)
            token_expert_pairs = torch.where(expert_mask)
            token_indices = token_expert_pairs[0]
            k_indices = token_expert_pairs[1]
            
            if len(token_indices) == 0:
                continue
            
            weights = topk_scores[token_indices, k_indices]
            expert_input = hidden_states[token_indices]
            # Execute expert (GPU or CPU)
            if self.expert_cache.is_cached(expert_id):
                # Execute on GPU
                expert_params = self.expert_cache.get(expert_id)
                expert_output = self.gpu_ops.expert_forward(
                    expert_input,
                    expert_params
                )
            else:
                # Execute on CPU
                cpu_params = self.parameter_loader.get_expert_params(
                    expert_id,
                    device=DeviceType.CPU
                )
                
                if cpu_params:
                    expert_input_cpu = expert_input.cpu()
                    expert_output_cpu = self.cpu_ops.expert_forward(
                        expert_input_cpu,
                        cpu_params
                    )
                    expert_output = expert_output_cpu.cuda()
                else:
                    logger.warning(f"Expert {expert_id} not found, skipping")
                    continue
            
            # Apply weights and accumulate
            weighted_output = expert_output * weights.unsqueeze(1)
            final_output[token_indices] += weighted_output
        
        return final_output
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Sample next token from logits.
        
        Args:
            logits: Logits [1, vocab_size]
            config: Generation configuration
        
        Returns:
            Sampled token ID
        """
        logits = logits / config.temperature
        
        # Apply repetition penalty if needed
        # TODO: Implement repetition penalty
        
        if config.do_sample:
            # Top-k filtering
            if config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, config.top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Top-p (nucleus) filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token.squeeze()
    
    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply embedding layer"""
        embed_weight = self.parameter_loader.static_params_gpu['embed_tokens']
        return self.gpu_ops.embedding(input_ids, embed_weight)
    
    def _layernorm_batch(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        norm_name: str
    ) -> torch.Tensor:
        """Apply layer normalization to batch"""
        norm_weight = self.parameter_loader.static_params_gpu[
            f"layer_{layer_idx}.{norm_name}"
        ]
        return self.gpu_ops.layernorm(hidden_states, norm_weight)
    
    def _self_attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kv_cache: KVCache
    ) -> torch.Tensor:
        """Self-attention with KV caching (reuse from prefill engine)"""
        layer_prefix = f"layer_{layer_idx}.self_attn"
        q_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.q_proj"]
        k_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.k_proj"]
        v_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.v_proj"]
        o_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.o_proj"]
        
        return self.gpu_ops.self_attention(
            hidden_states, q_proj, k_proj, v_proj, o_proj, kv_cache, layer_idx
        )
    
    def _final_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization"""
        norm_weight = self.parameter_loader.static_params_gpu['final_layernorm']
        return self.gpu_ops.layernorm(hidden_states, norm_weight)
    
    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply language model head"""
        lm_head_weight = self.parameter_loader.static_params_gpu['lm_head']
        return self.gpu_ops.linear(hidden_states, lm_head_weight)
    
    def _get_statistics(self) -> Dict:
        """Get generation statistics"""
        return {
            'prefill_time_ms': self.metrics.phase_starts.get('prefill', 0),
            'decode_time_ms': self.metrics.phase_starts.get('decode', 0)
        }