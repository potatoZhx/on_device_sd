# src/model/qwen3_runner.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor

from ..core.model_runner import (
    ModelRunner,
    RoutingResult,
    ExpertPlacement,
    LayerOutput,
    AttentionOutput,
)
from ..core.model import MoEConfig
from ..core.types import ExpertID
from ..memory.parameter_loader import ParameterLoader
from ..layers import (
    Qwen3DecoderLayer,
    RMSNorm,
    expert_forward_with_weights,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Qwen3ModelRunner(ModelRunner):
    """
    Qwen3-30B-A3B 模型的 ModelRunner 实现。
    内部使用 layers/ 模块中的 nn.Module 实现。
    """

    def __init__(
        self,
        config: MoEConfig,
        parameter_loader: ParameterLoader,
    ):
        self.config = config
        self.parameter_loader = parameter_loader

        # ---- 构建模型结构（使用 layers/ 的实现）----
        self.dtype = config.get_dtype()
        self.device = torch.device("cuda")

        # Decoder layers（静态参数内部持有）
        self.layers: List[Qwen3DecoderLayer] = []
        for layer_idx in range(config.num_hidden_layers):
            layer = Qwen3DecoderLayer(
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
            ).to(device=self.device, dtype=self.dtype)
            self.layers.append(layer)

        # Final norm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_norm = self.final_norm.to(device=self.device, dtype=self.dtype)

        # 加载静态权重
        self._load_static_weights()
        logger.info(f"Qwen3ModelRunner initialized: {config.num_hidden_layers} layers")

    # ==============================================================
    # ModelRunner 接口实现
    # ==============================================================

    def get_config(self) -> MoEConfig:
        return self.config

    def get_num_layers(self) -> int:
        return self.config.num_hidden_layers

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_weight = self.parameter_loader.static_params_gpu['embed_tokens']
        return F.embedding(input_ids.to(self.device), embed_weight)

    def route_experts(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> RoutingResult:
        """
        使用 layers/ 中的 MoEGate 计算路由。
        此处输入的 hidden_states 应该是经过 post-attention layernorm 后的。
        """
        gate = self.layers[layer_idx].mlp.gate
        top_k = self.config.num_experts_per_token

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        expert_indices, expert_weights = gate(hidden_states, top_k)
        # expert_indices: [batch_size, seq_len, top_k]
        # expert_weights: [batch_size, seq_len, top_k]

        # 展平为 [batch*seq, top_k]
        flat_indices = expert_indices.view(-1, top_k)
        flat_weights = expert_weights.view(-1, top_k)

        # 收集所有被激活的 expert ID
        activated = set()
        for idx in flat_indices.flatten().unique().tolist():
            activated.add(ExpertID(layer_idx, idx))

        return RoutingResult(
            layer_idx=layer_idx,
            topk_indices=flat_indices,
            topk_scores=flat_weights,
            activated_expert_ids=activated,
        )

    def forward_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: Optional[torch.Tensor],
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
        is_verify: bool = False,
    ) -> AttentionOutput:
        """
        执行 attention 部分（input_norm + attn + residual + post_attn_norm）。
        """
        layer = self.layers[layer_idx]

        # 初始化 seq_ids
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
        else:
            batch_size, seq_len = 1, hidden_states.shape[0]
            hidden_size = hidden_states.shape[-1]

        if seq_ids is None:
            seq_ids = list(range(batch_size))

        # PagedKVCache 需要提前创建/扩展 sequence
        if hasattr(kv_cache, "sequences") and hasattr(kv_cache, "add_sequence"):
            if is_prefill:
                for seq_id in seq_ids:
                    if seq_id not in kv_cache.sequences:
                        kv_cache.add_sequence(seq_id, prompt_len=seq_len)
            else:
                for seq_id in seq_ids:
                    if seq_id not in kv_cache.sequences:
                        kv_cache.add_sequence(seq_id, prompt_len=1)

        # ---- 1. Input LayerNorm ----
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # ---- 2. Self-Attention ----
        # 适配维度
        if hidden_states.dim() == 3:
            hidden_states_flat = hidden_states.reshape(-1, hidden_size)
            if positions is None:
                if is_prefill:
                    positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
                elif hasattr(kv_cache, "sequences"):
                    pos_vals = [kv_cache.sequences[seq_id].num_tokens - 1 for seq_id in seq_ids]
                    positions = torch.tensor(pos_vals, device=hidden_states.device).unsqueeze(1)
                else:
                    positions = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.long)
            positions_flat = positions.reshape(-1) if positions is not None else None
            reshape_back = True
        else:
            hidden_states_flat = hidden_states
            if positions is None:
                if is_prefill:
                    positions_flat = torch.arange(hidden_states_flat.shape[0], device=hidden_states.device)
                elif hasattr(kv_cache, "sequences") and seq_ids:
                    positions_flat = torch.tensor([kv_cache.sequences[seq_ids[0]].num_tokens - 1], device=hidden_states.device)
                else:
                    positions_flat = torch.zeros(hidden_states_flat.shape[0], device=hidden_states.device, dtype=torch.long)
            else:
                positions_flat = positions
            reshape_back = False

        attn_output = layer.self_attn(
            hidden_states=hidden_states_flat,
            positions=positions_flat,
            kv_cache=kv_cache,
            seq_ids=seq_ids,
            is_prefill=is_prefill,
            is_verify=is_verify,
        )

        if reshape_back:
            attn_output = attn_output.reshape(batch_size, seq_len, -1)

        hidden_states = residual + attn_output

        # ---- 3. Post-Attention LayerNorm ----
        normed = layer.post_attention_layernorm(hidden_states)

        return AttentionOutput(
            hidden_states=hidden_states,
            post_attn_normed=normed,
            residual=hidden_states,
        )

    def forward_moe(
        self,
        layer_idx: int,
        attn_output: AttentionOutput,
        expert_placement: ExpertPlacement,
    ) -> torch.Tensor:
        """
        执行 MoE 部分（根据 placement 执行 expert + residual）。
        """
        moe_output = self._execute_moe_with_placement(
            hidden_states=attn_output.post_attn_normed,
            expert_placement=expert_placement,
        )

        return attn_output.residual + moe_output

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: Optional[torch.Tensor],
        expert_placement: ExpertPlacement,
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
        is_verify: bool = False,
    ) -> LayerOutput:
        """完整单层 forward，包含 metadata。"""
        attn_out = self.forward_attention(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            positions=positions,
            seq_ids=seq_ids,
            is_prefill=is_prefill,
            is_verify=is_verify,
        )
        hidden_states = self.forward_moe(layer_idx, attn_out, expert_placement)

        metadata = {
            "gpu_expert_count": len(expert_placement.gpu_expert_params),
            "cpu_expert_count": len(expert_placement.cpu_expert_params),
            "substituted_count": len(expert_placement.substitution_map),
        }

        return LayerOutput(hidden_states=hidden_states, metadata=metadata)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.final_norm(hidden_states)
        lm_head_weight = self.parameter_loader.static_params_gpu['lm_head']
        return F.linear(hidden_states, lm_head_weight)

    # ==============================================================
    # 内部实现
    # ==============================================================

    def _execute_moe_with_placement(
        self,
        hidden_states: torch.Tensor,
        expert_placement: ExpertPlacement,
    ) -> torch.Tensor:
        """
        根据 ExpertPlacement 执行 MoE 计算。
        使用 layers/mlp.py 中的 expert_forward_with_weights。
        """
        routing = expert_placement.routing_result
        if routing is None:
            return torch.zeros_like(hidden_states)

        # 展平
        if hidden_states.dim() == 3:
            b, s, h = hidden_states.shape
            flat = hidden_states.view(-1, h)
            need_reshape = True
        else:
            flat = hidden_states
            h = flat.shape[-1]
            need_reshape = False

        topk_indices = routing.topk_indices  # [batch*seq, top_k]
        topk_scores = routing.topk_scores

        final_output = torch.zeros(flat.shape[0], h, device=flat.device, dtype=flat.dtype)

        # 处理替换映射
        sub_map = expert_placement.substitution_map

        gpu_tasks = []
        cpu_tasks = []

        for expert_id in routing.activated_expert_ids:
            expert_idx = expert_id.expert_idx
            expert_mask = (topk_indices == expert_idx)
            token_expert_pairs = torch.where(expert_mask)
            token_indices = token_expert_pairs[0]
            k_indices = token_expert_pairs[1]

            if len(token_indices) == 0:
                continue

            weights = topk_scores[token_indices, k_indices]
            expert_input = flat[token_indices]

            if expert_idx in expert_placement.gpu_expert_params:
                params = expert_placement.gpu_expert_params[expert_idx]
                gpu_tasks.append((token_indices, weights, expert_input, params))
            elif expert_idx in expert_placement.cpu_expert_params:
                params = expert_placement.cpu_expert_params[expert_idx]
                cpu_tasks.append((expert_idx, token_indices, weights, expert_input, params))
            elif expert_idx in sub_map:
                sub_idx = sub_map[expert_idx]
                if sub_idx in expert_placement.gpu_expert_params:
                    params = expert_placement.gpu_expert_params[sub_idx]
                    gpu_tasks.append((token_indices, weights, expert_input, params))
                elif sub_idx in expert_placement.cpu_expert_params:
                    params = expert_placement.cpu_expert_params[sub_idx]
                    cpu_tasks.append((sub_idx, token_indices, weights, expert_input, params))
                else:
                    continue
            else:
                continue

        def _prepare_params(params: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
            if params["gate_proj"].device == device and params["gate_proj"].dtype == dtype:
                return params
            return {k: v.to(device=device, dtype=dtype) for k, v in params.items()}

        def _run_cpu_task(task):
            expert_idx, token_indices, weights, expert_input, params = task
            cpu_input = expert_input.to("cpu")
            if cpu_input.dtype != params["gate_proj"].dtype:
                cpu_input = cpu_input.to(params["gate_proj"].dtype)
            output_cpu = expert_forward_with_weights(
                cpu_input,
                params["gate_proj"], params["up_proj"], params["down_proj"],
            )
            output_gpu = output_cpu.to(flat.device, dtype=flat.dtype)
            return expert_idx, token_indices, weights, output_gpu

        if cpu_tasks and gpu_tasks and flat.is_cuda:
            with ThreadPoolExecutor(max_workers=min(len(cpu_tasks), 4)) as executor:
                futures = [executor.submit(_run_cpu_task, task) for task in cpu_tasks]
                for token_indices, weights, expert_input, params in gpu_tasks:
                    params = _prepare_params(params, flat.device, flat.dtype)
                    expert_output = expert_forward_with_weights(
                        expert_input,
                        params["gate_proj"], params["up_proj"], params["down_proj"],
                    )
                    if expert_output.dtype != flat.dtype:
                        expert_output = expert_output.to(flat.dtype)
                    final_output[token_indices] += expert_output * weights.unsqueeze(1)
                for fut in futures:
                    _, token_indices, weights, output_gpu = fut.result()
                    final_output[token_indices] += output_gpu * weights.unsqueeze(1)
        else:
            for token_indices, weights, expert_input, params in gpu_tasks:
                params = _prepare_params(params, flat.device, flat.dtype)
                expert_output = expert_forward_with_weights(
                    expert_input,
                    params["gate_proj"], params["up_proj"], params["down_proj"],
                )
                if expert_output.dtype != flat.dtype:
                    expert_output = expert_output.to(flat.dtype)
                final_output[token_indices] += expert_output * weights.unsqueeze(1)

            for task in cpu_tasks:
                _, token_indices, weights, output_gpu = _run_cpu_task(task)
                final_output[token_indices] += output_gpu * weights.unsqueeze(1)

        if need_reshape:
            final_output = final_output.view(b, s, h)

        return final_output

    def _load_static_weights(self) -> None:
        """从 ParameterLoader 加载静态权重到 layers"""
        static_params = self.parameter_loader.static_params_gpu
        for layer_idx, layer in enumerate(self.layers):
            prefix = f"layer_{layer_idx}"
            layer.load_weights(
                input_layernorm_weight=static_params[f"{prefix}.input_layernorm"],
                q_weight=static_params[f"{prefix}.self_attn.q_proj"],
                k_weight=static_params[f"{prefix}.self_attn.k_proj"],
                v_weight=static_params[f"{prefix}.self_attn.v_proj"],
                o_weight=static_params[f"{prefix}.self_attn.o_proj"],
                q_norm_weight=static_params.get(f"{prefix}.self_attn.q_norm"),
                k_norm_weight=static_params.get(f"{prefix}.self_attn.k_norm"),
                post_attention_layernorm_weight=static_params[f"{prefix}.post_attention_layernorm"],
                gate_weight=static_params[f"{prefix}.router"],
            )
        if "final_layernorm" in static_params:
            self.final_norm.weight.data = static_params["final_layernorm"].to(self.final_norm.weight.dtype)
        logger.info("Static weights loaded into Qwen3ModelRunner layers")
