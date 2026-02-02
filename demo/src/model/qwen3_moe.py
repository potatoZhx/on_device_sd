"""
Qwen3 MoE Model Implementation
完整的模型推理实现
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict

from ..core.model import MoEConfig
from ..core.types import ExpertID
from ..layers.decoder_layer import Qwen3DecoderLayer
from ..layers.layernorm import RMSNorm
from ..memory.parameter_loader import ParameterLoader
from ..memory.paged_kv_cache import PagedKVCache


class Qwen3MoEModel(nn.Module):
    """
    Qwen3 MoE 完整模型
    包含 Embedding + N * DecoderLayer + Final LayerNorm + LM Head
    """
    def __init__(
        self,
        config: MoEConfig,
        kv_cache_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = config.get_dtype()
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=self.dtype,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                moe_intermediate_size=config.moe_intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                layer_idx=layer_idx,
                qkv_bias=False,
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM Head (output projection)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=self.dtype,
        )
        
        # KV Cache
        if kv_cache_config is None:
            kv_cache_config = {
                'block_size': 256,
                'gpu_memory_utilization': 0.9,
            }
        
        self.kv_cache = PagedKVCache(
            config=config,
            block_size=kv_cache_config['block_size'],
            gpu_memory_utilization=kv_cache_config.get('gpu_memory_utilization', 0.9),
            dtype=self.dtype,
        )
        
        # Parameter loader (will be set externally)
        self.param_loader: Optional[ParameterLoader] = None
    
    def set_parameter_loader(self, param_loader: ParameterLoader):
        """设置参数加载器"""
        self.param_loader = param_loader
    
    def load_static_weights(self):
        """
        加载静态参数（Embedding, LayerNorm, Attention, Gate）
        Expert 权重在 forward 时动态加载
        """
        if self.param_loader is None:
            raise ValueError("Parameter loader not set. Call set_parameter_loader() first.")
        
        static_params = self.param_loader.static_params_gpu
        
        # Load embedding
        if 'embed_tokens' in static_params:
            self.embed_tokens.weight.data = static_params['embed_tokens'].to(self.dtype)
        
        # Load final LayerNorm
        if 'norm' in static_params:
            self.norm.weight.data = static_params['norm'].to(self.dtype)
        
        # Load LM head
        if 'lm_head' in static_params:
            self.lm_head.weight.data = static_params['lm_head'].to(self.dtype)
        
        # Load decoder layer weights
        for layer_idx, layer in enumerate(self.layers):
            layer_prefix = f"layer_{layer_idx}"
            
            layer.load_weights(
                input_layernorm_weight=static_params[f"{layer_prefix}.input_layernorm"],
                q_weight=static_params[f"{layer_prefix}.self_attn.q_proj"],
                k_weight=static_params[f"{layer_prefix}.self_attn.k_proj"],
                v_weight=static_params[f"{layer_prefix}.self_attn.v_proj"],
                o_weight=static_params[f"{layer_prefix}.self_attn.o_proj"],
                q_norm_weight=static_params.get(f"{layer_prefix}.self_attn.q_norm"),
                k_norm_weight=static_params.get(f"{layer_prefix}.self_attn.k_norm"),
                post_attention_layernorm_weight=static_params[f"{layer_prefix}.post_attention_layernorm"],
                gate_weight=static_params[f"{layer_prefix}.router"],
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        expert_weights: Optional[Dict[int, Dict[int, Dict[str, torch.Tensor]]]] = None,
        seq_ids: Optional[list[int]] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        模型前向传播
        
        Args:
            input_ids: [batch_size, seq_len] - 输入 token IDs
            positions: [batch_size, seq_len] - 位置索引（可选，默认自动生成）
            expert_weights: {layer_idx: {expert_idx: {'gate_proj': ..., 'up_proj': ..., 'down_proj': ...}}}
            seq_ids: sequence IDs for KV cache (optional)
            is_prefill: whether this is prefill phase
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Generate seq_ids if not provided
        if seq_ids is None:
            seq_ids = list(range(batch_size))
        
        # Initialize sequences in KV cache if needed
        for seq_id in seq_ids:
            if seq_id not in self.kv_cache.sequences:
                self.kv_cache.add_sequence(seq_id, prompt_len=seq_len)
        
        # 1. Embedding
        hidden_states = self.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # 2. Decoder layers
        for layer_idx, layer in enumerate(self.layers):
            # Get expert weights for this layer
            layer_expert_weights = None
            if expert_weights is not None and layer_idx in expert_weights:
                layer_expert_weights = expert_weights[layer_idx]
            
            # Get KV cache
            kv_cache = self.kv_cache
            
            hidden_states = layer(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                positions=positions,
                expert_weights_dict=layer_expert_weights,
                seq_ids=seq_ids,
                is_prefill=is_prefill,
            )
        
        # 3. Final LayerNorm
        hidden_states = self.norm(hidden_states)
        
        # 4. LM Head
        logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        简单的自回归生成（不使用 speculative decoding）
        
        Args:
            input_ids: [batch_size, seq_len] - 输入 prompt
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Nucleus 采样
            eos_token_id: EOS token ID（可选）
            
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始化生成序列
        generated = input_ids.clone()
        
        # TODO: 在实际使用时需要动态加载 expert 权重
        # 这里假设所有 experts 都已加载（用于测试）
        expert_weights = self._load_all_experts()
        
        for _ in range(max_new_tokens):
            # 1. Forward pass
            logits = self.forward(
                input_ids=generated,
                expert_weights=expert_weights,
            )  # [batch_size, seq_len, vocab_size]
            
            # 2. 只取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 3. Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 4. Top-K sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 5. Top-P (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 6. Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # 7. Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # 8. Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def _load_all_experts(self) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
        """
        加载所有 experts 到 GPU（用于测试）
        实际使用时应该按需加载
        """
        if self.param_loader is None:
            return {}
        
        expert_weights = {}
        for layer_idx in range(self.config.num_hidden_layers):
            expert_weights[layer_idx] = {}
            for expert_idx in range(self.config.num_experts):
                expert_id = ExpertID(layer_idx, expert_idx)
                params = self.param_loader.get_expert_params(expert_id, device='cuda')
                if params is not None:
                    expert_weights[layer_idx][expert_idx] = params
        
        return expert_weights
