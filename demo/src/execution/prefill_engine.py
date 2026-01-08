from typing import Dict, List, Optional
import torch
from ..core.types import ExpertID, LayerExpertActivations, ExpertActivation
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


class PrefillEngine:
    """
    Prefill phase execution engine.
    Performs full model inference with CPU-GPU heterogeneous execution.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        prefetcher: ExpertPrefetcher,
        metrics: MetricsCollector
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.metrics = metrics
        
        # Operators
        self.gpu_ops = GPUOperators(config)
        self.cpu_ops = CPUOperators(config)
        
        # Activation history for prefetching
        self.activation_history: List[LayerExpertActivations] = []
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache
    ) -> Dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            kv_cache: KV cache to populate
        
        Returns:
            Dict with 'logits', 'next_token_id', and other outputs
        """
        self.metrics.start_phase('prefill')
        
        # Move input to GPU
        input_ids_gpu = input_ids.cuda()
        
        # Embedding
        hidden_states = self._embed_tokens(input_ids_gpu)
        
        # Process each layer
        for layer_idx in range(self.config.num_hidden_layers):
            logger.debug(f"Processing layer {layer_idx}")
            
            hidden_states = self._process_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=kv_cache
            )
        
        # Final layer norm
        hidden_states = self._final_layernorm(hidden_states)
        
        # LM head
        logits = self._lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # Get next token
        next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        self.metrics.end_phase('prefill')
        
        return {
            'logits': logits,
            'next_token_id': next_token_id,
            'hidden_states': hidden_states
        }
    
    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply embedding layer"""
        embed_weight = self.parameter_loader.static_params_gpu['embed_tokens']
        return self.gpu_ops.embedding(input_ids, embed_weight)
    
    def _process_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: KVCache
    ) -> torch.Tensor:
        """
        Process a single transformer layer with MoE.
        
        Args:
            layer_idx: Layer index
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            kv_cache: KV cache
        
        Returns:
            Output hidden states
        """
        residual = hidden_states
        
        # 1. Pre-attention layer norm
        hidden_states = self._layernorm(hidden_states, layer_idx, 'input_layernorm')
        
        # 2. Self-attention
        hidden_states = self._self_attention(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            kv_cache=kv_cache
        )
        
        # 3. Residual connection
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        # 4. Post-attention layer norm
        hidden_states = self._layernorm(hidden_states, layer_idx, 'post_attention_layernorm')
        
        # 5. MoE FFN
        hidden_states = self._moe_forward(
            hidden_states=hidden_states,
            layer_idx=layer_idx
        )
        
        # 6. Residual connection
        hidden_states = residual + hidden_states
        
        # Update KV cache length
        kv_cache.update_length(kv_cache.current_length + hidden_states.shape[1])
        
        return hidden_states
    
    def _self_attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kv_cache: KVCache
    ) -> torch.Tensor:
        """Self-attention with KV caching"""
        # Get attention weights
        layer_prefix = f"layer_{layer_idx}.self_attn"
        q_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.q_proj"]
        k_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.k_proj"]
        v_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.v_proj"]
        o_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.o_proj"]
        
        # Apply attention
        output = self.gpu_ops.self_attention(
            hidden_states=hidden_states,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            kv_cache=kv_cache,
            layer_idx=layer_idx
        )
        
        return output
    
    def _moe_forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        MoE forward pass with CPU-GPU heterogeneous execution.
        
        Args:
            hidden_states: Input [batch, seq_len, hidden_size]
            layer_idx: Current layer index
        
        Returns:
            MoE output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Router: compute expert scores
        router_weight = self.parameter_loader.static_params_gpu[f"layer_{layer_idx}.router"]
        routing_scores = self.gpu_ops.router(hidden_states, router_weight)  # [batch*seq, num_experts]
        
        # 2. Select top-k experts
        top_k = self.config.num_experts_per_token
        topk_scores, topk_indices = torch.topk(routing_scores, top_k, dim=-1)
        topk_scores = torch.softmax(topk_scores, dim=-1)  # Normalize scores
        
        # 3. Collect expert activations
        layer_activations = self._collect_expert_activations(
            layer_idx=layer_idx,
            topk_indices=topk_indices,
            topk_scores=topk_scores
        )
        
        # 4. Prefetch experts for next layer
        self.prefetcher.prefetch_for_next_layer(
            current_layer_idx=layer_idx,
            current_activations=layer_activations,
            history=self.activation_history,
            expert_cache=self.expert_cache,
            parameter_loader=self.parameter_loader
        )
        
        # 5. Execute experts (CPU-GPU heterogeneous)
        expert_outputs = self._execute_experts_heterogeneous(
            hidden_states=hidden_states.view(batch_size * seq_len, hidden_size),
            layer_idx=layer_idx,
            topk_indices=topk_indices,
            topk_scores=topk_scores
        )
        
        # 6. Store activation history
        self.activation_history.append(layer_activations)
        if len(self.activation_history) > 20:  # Keep last 20 layers
            self.activation_history.pop(0)
        
        # 7. Reshape output
        output = expert_outputs.view(batch_size, seq_len, hidden_size)
        
        return output
    
    def _collect_expert_activations(
        self,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor
    ) -> LayerExpertActivations:
        """Collect expert activation information for analysis"""
        batch_seq, top_k = topk_indices.shape
        
        activations = []
        for k in range(top_k):
            for expert_idx in range(self.config.num_experts):
                # Find tokens that activate this expert at rank k
                mask = (topk_indices[:, k] == expert_idx)
                token_indices = torch.where(mask)[0]
                
                if len(token_indices) > 0:
                    expert_id = ExpertID(layer_idx, expert_idx)
                    scores = topk_scores[token_indices, k]
                    
                    activations.append(ExpertActivation(
                        expert_id=expert_id,
                        token_indices=token_indices,
                        scores=scores,
                        top_k_rank=k
                    ))
        
        return LayerExpertActivations(
            layer_idx=layer_idx,
            activations=activations,
            routing_scores=topk_scores
        )
    
    def _execute_experts_heterogeneous(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute experts across CPU and GPU based on availability.
        
        Args:
            hidden_states: [batch*seq, hidden_size]
            layer_idx: Current layer
            topk_indices: [batch*seq, top_k] - expert indices
            topk_scores: [batch*seq, top_k] - expert weights
        
        Returns:
            Weighted expert outputs [batch*seq, hidden_size]
        """
        batch_seq, hidden_size = hidden_states.shape
        top_k = topk_indices.shape[1]
        
        # Initialize output
        final_output = torch.zeros_like(hidden_states)
        
        # Group tokens by expert
        for expert_idx in range(self.config.num_experts):
            expert_id = ExpertID(layer_idx, expert_idx)
            
            # Find tokens that use this expert
            expert_mask = (topk_indices == expert_idx)
            token_expert_pairs = torch.where(expert_mask)
            token_indices = token_expert_pairs[0]
            k_indices = token_expert_pairs[1]
            
            if len(token_indices) == 0:
                continue
            
            # Get expert weights for these tokens
            weights = topk_scores[token_indices, k_indices]  # [num_tokens]
            
            # Get input for this expert
            expert_input = hidden_states[token_indices]  # [num_tokens, hidden_size]
            
            # Execute expert (GPU or CPU)
            if self.expert_cache.is_cached(expert_id):
                # Execute on GPU
                expert_params = self.expert_cache.get(expert_id)
                expert_output = self.gpu_ops.expert_forward(
                    expert_input, 
                    expert_params
                )
            else:
                # Check if available on CPU
                cpu_params = self.parameter_loader.get_expert_params(
                    expert_id,
                    device=DeviceType.CPU
                )
                
                if cpu_params:
                    # Execute on CPU and transfer result
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
    
    def _layernorm(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        norm_name: str
    ) -> torch.Tensor:
        """Apply layer normalization"""
        norm_weight = self.parameter_loader.static_params_gpu[
            f"layer_{layer_idx}.{norm_name}"
        ]
        return self.gpu_ops.layernorm(hidden_states, norm_weight)
    
    defdef _final_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization"""
        norm_weight = self.parameter_loader.static_params_gpu['final_layernorm']
        return self.gpu_ops.layernorm(hidden_states, norm_weight)
    
    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply language model head"""
        lm_head_weight = self.parameter_loader.static_params_gpu['lm_head']
        return self.gpu_ops.linear(hidden_states, lm_head_weight)