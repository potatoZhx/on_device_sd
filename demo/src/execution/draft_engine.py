from typing import Dict, List, Optional
import torch
import numpy as np
from ..core.types import ExpertID, LayerExpertActivations, DraftMetrics
from ..core.model import MoEConfig
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..memory.kv_cache import KVCache
from ..scheduling.draft_scheduler import DraftSchedulingStrategy
from ..operators.gpu_operators import GPUOperators
from ..operators.cpu_operators import CPUOperators
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


class DraftEngine:
    """
    Draft phase execution engine.
    Uses speculative decoding with CPU-GPU expert substitution.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
        expert_cache: ExpertCache,
        draft_scheduler: DraftSchedulingStrategy,
        metrics: MetricsCollector
    ):
        self.config = config
        self.parameter_loader = parameter_loader
        self.expert_cache = expert_cache
        self.draft_scheduler = draft_scheduler
        self.metrics = metrics
        
        # Operators
        self.gpu_ops = GPUOperators(config)
        self.cpu_ops = CPUOperators(config)
        
        # Track activations for cache updates
        self.draft_activations: List[LayerExpertActivations] = []
        
        # Cache hit tracking
        self.cache_hits = 0
        self.cache_misses = 0
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        max_draft_tokens: int,
        temperature: float = 1.0
    ) -> Dict:
        """
        Generate draft tokens using expert substitution strategy.
        
        Args:
            input_ids: Starting token ID [1] (last generated token)
            kv_cache: KV cache to extend
            max_draft_tokens: Maximum number of tokens to draft
            temperature: Sampling temperature
        
        Returns:
            Dict with 'drafted_tokens', 'metrics', etc.
        """
        self.metrics.start_phase('draft')
        
        drafted_tokens = []
        current_token = input_ids.cuda()
        
        self.cache_hits = 0
        self.cache_misses = 0
        self.draft_activations = []
        
        # Draft loop
        for step in range(max_draft_tokens):
            logger.debug(f"Draft step {step + 1}/{max_draft_tokens}")
            
            # Forward pass with substitution
            output = self._draft_forward_pass(
                input_ids=current_token,
                kv_cache=kv_cache
            )
            
            # Sample next token
            logits = output['logits'][:, -1, :] / temperature  # [1, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            
            drafted_tokens.append(next_token.item())
            current_token = next_token
        
        # Calculate metrics
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        perplexity = self._calculate_perplexity(output['logits'])
        
        metrics = DraftMetrics(
            num_drafted_tokens=len(drafted_tokens),
            perplexity=perplexity,
            expert_hit_rate=cache_hit_rate,
            cpu_compute_ratio=self.cache_misses / max(self.cache_hits + self.cache_misses, 1)
        )
        
        # Schedule expert transfers based on draft activations
        self._schedule_expert_transfers()
        
        self.metrics.end_phase('draft')
        
        logger.info(f"Drafted {len(drafted_tokens)} tokens, "
                   f"cache hit rate: {cache_hit_rate:.2%}, "
                   f"perplexity: {perplexity:.3f}")
        
        return {
            'drafted_tokens': drafted_tokens,
            'metrics': metrics,
            'activations': self.draft_activations
        }
    
    def _draft_forward_pass(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache
    ) -> Dict:
        """
        Single forward pass with expert substitution.
        
        Args:
            input_ids: Single token [1, 1]
            kv_cache: KV cache
        
        Returns:
            Dict with logits
        """
        # Embedding
        hidden_states = self._embed_tokens(input_ids)  # [1, 1, hidden_size]
        
        # Process each layer
        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states = self._process_layer_with_substitution(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=kv_cache
            )
        
        # Final layers
        hidden_states = self._final_layernorm(hidden_states)
        logits = self._lm_head(hidden_states)
        
        return {'logits': logits, 'hidden_states': hidden_states}
    
    def _process_layer_with_substitution(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: KVCache
    ) -> torch.Tensor:
        """
        Process layer with expert substitution strategy.
        """
        residual = hidden_states
        
        # Layer norm
        hidden_states = self._layernorm(hidden_states, layer_idx, 'input_layernorm')
        
        # Self-attention
        hidden_states = self._self_attention(
            hidden_states, layer_idx, kv_cache
        )
        
        # Residual
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        # Layer norm
        hidden_states = self._layernorm(hidden_states, layer_idx, 'post_attention_layernorm')
        
        # MoE with substitution
        hidden_states = self._moe_forward_with_substitution(
            hidden_states=hidden_states,
            layer_idx=layer_idx
        )
        
        # Residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _moe_forward_with_substitution(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        MoE forward with CPU execution and GPU substitution.
        
        Strategy:
        1. Compute routing scores
        2. For top-c experts: execute on CPU
        3. For remaining experts: substitute with GPU-cached experts
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]
        
        # 1. Router
        router_weight = self.parameter_loader.static_params_gpu[f"layer_{layer_idx}.router"]
        routing_scores = self.gpu_ops.router(hidden_states, router_weight)
        
        # 2. Select top-k experts
        top_k = self.config.num_experts_per_token
        topk_scores, topk_indices = torch.topk(routing_scores, top_k, dim=-1)
        topk_scores = torch.softmax(topk_scores, dim=-1)
        
        # 3. Collect activations
        layer_activations = self._collect_activations(
            layer_idx, topk_indices, topk_scores
        )
        self.draft_activations.append(layer_activations)
        
        # 4. Select top-c for CPU execution
        top_c = self.config.draft_top_c
        cpu_expert_ids = self.draft_scheduler.select_cpu_experts(
            layer_activations, top_c
        )
        
        # 5. Determine which experts need substitution
        all_activated_expert_ids = set(
            ExpertID(layer_idx, idx.item())
            for idx in topk_indices.flatten().unique()
        )
        
        cached_experts = set(
            eid for eid in all_activated_expert_ids
            if self.expert_cache.is_cached(eid)
        )
        
        needs_substitution = all_activated_expert_ids - cached_experts - set(cpu_expert_ids)
        
        # 6. Get substitution mapping
        all_layer_experts = [
            ExpertID(layer_idx, i) 
            for i in range(self.config.num_experts)
        ]
        
        substitution_map = self.draft_scheduler.select_gpu_substitutes(
            requested_experts=list(needs_substitution),
            cached_experts=cached_experts,
            all_experts=all_layer_experts
        )
        
        # 7. Execute experts with substitution
        output = self._execute_with_substitution(
            hidden_states=flat_hidden,
            layer_idx=layer_idx,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            cpu_experts=set(cpu_expert_ids),
            substitution_map=substitution_map
        )
        
        return output.view(batch_size, seq_len, hidden_size)
    
    def _execute_with_substitution(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        cpu_experts: set,
        substitution_map: Dict[ExpertID, ExpertID]
    ) -> torch.Tensor:
        """
        Execute experts with substitution strategy.
        
        CPU experts: Run on CPU
        Cached experts: Run on GPU
        Others: Use substitutes from GPU cache
        """
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
            
            # Determine execution strategy
            if expert_id in cpu_experts:
                # Execute on CPU
                cpu_params = self.parameter_loader.get_expert_params(expert_id)
                if cpu_params:
                    expert_input_cpu = expert_input.cpu()
                    expert_output_cpu = self.cpu_ops.expert_forward(
                        expert_input_cpu, cpu_params
                    )
                    expert_output = expert_output_cpu.cuda()
                    self.cache_misses += 1
                else:
                    logger.warning(f"CPU expert {expert_id} not found")
                    continue
                    
            elif self.expert_cache.is_cached(expert_id):
                # Execute on GPU (cache hit)
                gpu_params = self.expert_cache.get(expert_id)
                expert_output = self.gpu_ops.expert_forward(expert_input, gpu_params)
                self.cache_hits += 1
                
            elif expert_id in substitution_map:
                # Use substitute expert from GPU
                substitute_id = substitution_map[expert_id]
                substitute_params = self.expert_cache.get(substitute_id)
                
                if substitute_params:
                    expert_output = self.gpu_ops.expert_forward(
                        expert_input, substitute_params
                    )
                    self.cache_hits += 1  # Substitution counts as hit
                    logger.debug(f"Used substitute {substitute_id} for {expert_id}")
                else:
                    logger.warning(f"Substitute {substitute_id} not in cache")
                    continue
            else:
                # Fallback: skip this expert
                logger.warning(f"No execution path for {expert_id}")
                continue
            
            # Accumulate weighted output
            weighted_output = expert_output * weights.unsqueeze(1)
            final_output[token_indices] += weighted_output
        
        return final_output
    
    def _schedule_expert_transfers(self) -> None:
        """
        Schedule expert transfers based on draft activations.
        """
        if not self.draft_activations:
            return
        
        # Get currently cached experts
        cached_experts = set(self.expert_cache.cached_experts.keys())
        
        # Select experts to transfer
        experts_to_transfer = self.draft_scheduler.select_experts_to_transfer(
            recent_activations=self.draft_activations,
            cached_experts=cached_experts,
            cache_capacity=self.expert_cache.max_experts
        )
        
        # Trigger async transfers
        if experts_to_transfer:
            logger.info(f"Scheduling {len(experts_to_transfer)} expert transfers")
            
            for expert_id in experts_to_transfer:
                cpu_params = self.parameter_loader.get_expert_params(expert_id)
                if cpu_params:
                    self.expert_cache.put(expert_id, cpu_params)
    
    def _collect_activations(
        self,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor
    ) -> LayerExpertActivations:
        """Collect activation information for draft scheduling"""
        from ..core.types import ExpertActivation
        
        activations = []
        for k in range(topk_indices.shape[1]):
            for expert_idx in range(self.config.num_experts):
                mask = (topk_indices[:, k] == expert_idx)
                token_indices = torch.where(mask)[0]
                
                if len(token_indices) > 0:
                    activations.append(ExpertActivation(
                        expert_id=ExpertID(layer_idx, expert_idx),
                        token_indices=token_indices,
                        scores=topk_scores[token_indices, k],
                        top_k_rank=k
                    ))
        
        return LayerExpertActivations(
            layer_idx=layer_idx,
            activations=activations,
            routing_scores=topk_scores
        )
    
    def _calculate_perplexity(self, logits: torch.Tensor) -> float:
        """Calculate perplexity from logits"""
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=-1))
        perplexity = torch.exp(entropy).item()
        return perplexity
    
    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_weight = self.parameter_loader.static_params_gpu['embed_tokens']
        return self.gpu_ops.embedding(input_ids, embed_weight)
    
    def _layernorm(self, hidden_states: torch.Tensor, layer_idx: int, norm_name: str) -> torch.Tensor:
        norm_weight = self.parameter_loader.static_params_gpu[f"layer_{layer_idx}.{norm_name}"]
        return self.gpu_ops.layernorm(hidden_states, norm_weight)
    
    def _self_attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kv_cache: KVCache
    ) -> torch.Tensor:
        layer_prefix = f"layer_{layer_idx}.self_attn"
        q_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.q_proj"]
        k_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.k_proj"]
        v_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.v_proj"]
        o_proj = self.parameter_loader.static_params_gpu[f"{layer_prefix}.o_proj"]
        
        return self.gpu_ops.self_attention(
            hidden_states, q_proj, k_proj, v_proj, o_proj, kv_cache, layer_idx
        )
    
    def _final_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_weight = self.parameter_loader.static_params_gpu['final_layernorm']
        return self.gpu_ops.layernorm(hidden_states, norm_weight)
    
    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm_head_weight = self.parameter_loader.static_params_gpu['lm_head']
        return self.gpu_ops.linear(hidden_states, lm_head_weight)