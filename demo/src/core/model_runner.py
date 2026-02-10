from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import torch

from .types import ExpertID
from .model import MoEConfig


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class RoutingResult:
    """
    路由计算结果，由 ModelRunner.route_experts() 返回，
    由引擎用于调度决策。
    """
    layer_idx: int
    topk_indices: torch.Tensor     # [batch*seq, top_k] 选中的 expert 索引
    topk_scores: torch.Tensor      # [batch*seq, top_k] 归一化后的权重
    activated_expert_ids: Set[ExpertID]  # 本层被激活的所有 expert ID 集合

    @property
    def top_k(self) -> int:
        return self.topk_indices.shape[-1]


@dataclass
class ExpertPlacement:
    """
    引擎的调度决策结果，传递给 ModelRunner.forward_layer()。
    告诉 ModelRunner 每个 expert 应该在哪里执行。
    """
    # 在 GPU 上执行的 expert 及其参数
    gpu_expert_params: Dict[int, Dict[str, torch.Tensor]]
    # 在 CPU 上执行的 expert 及其参数
    cpu_expert_params: Dict[int, Dict[str, torch.Tensor]]
    # 替换映射：{原 expert_idx: 替代 expert_idx}（仅 draft 阶段使用）
    substitution_map: Dict[int, int] = field(default_factory=dict)
    # 路由结果（从 route_experts 透传）
    routing_result: Optional[RoutingResult] = None


@dataclass
class LayerOutput:
    """
    ModelRunner.forward_layer() 的返回结果。
    包含计算输出和元数据（供引擎收集 metrics）。
    """
    hidden_states: torch.Tensor
    # 元数据供引擎使用（metrics 收集、prefetch 决策等）
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata 可包含:
    #   "gpu_expert_count": int     - GPU 上执行的 expert 数量
    #   "cpu_expert_count": int     - CPU 上执行的 expert 数量
    #   "substituted_count": int    - 被替换的 expert 数量
    #   "layer_time_ms": float      - 本层计算耗时


@dataclass
class AttentionOutput:
    """forward_attention 的返回结果"""
    hidden_states: torch.Tensor      # attention 后加了 residual 的 hidden_states
    post_attn_normed: torch.Tensor   # post_attention_layernorm 后的结果（用于 routing 和 MoE）
    residual: torch.Tensor           # MoE 之前的 residual（= hidden_states）


# ============================================================
# ModelRunner 抽象接口
# ============================================================

class ModelRunner(ABC):
    """
    模型前向推理的统一抽象。

    职责：
      - 管理模型结构和权重
      - 执行各层计算（embedding、attention、MoE、norm、lm_head）
      - 根据引擎提供的 ExpertPlacement 在 CPU/GPU 上执行 expert

    不负责：
      - 调度决策（由引擎负责）
      - KV Cache 生命周期管理（由引擎/Memory 模块负责）
      - Prefetch 策略（由引擎/Scheduling 模块负责）
    """

    @abstractmethod
    def get_config(self) -> MoEConfig:
        """返回模型配置"""
        ...

    @abstractmethod
    def get_num_layers(self) -> int:
        """返回模型的 decoder layer 数量"""
        ...

    # ----------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------
    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Token embedding lookup。

        Args:
            input_ids: [batch_size, seq_len] 或 [seq_len]

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        ...

    # ----------------------------------------------------------
    # Attention（input_norm + attn + residual + post_attn_norm）
    # ----------------------------------------------------------
    @abstractmethod
    def forward_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: Optional[torch.Tensor],
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
    ) -> AttentionOutput:
        """
        执行注意力部分（不包含 MoE）。

        Args:
            layer_idx: 当前层索引
            hidden_states: 输入隐状态 [batch_size, seq_len, hidden_size]
            kv_cache: KV Cache 对象（KVCache 或 PagedKVCache）
            positions: 位置索引 [batch_size, seq_len] 或 [num_tokens]
            seq_ids: 序列 ID 列表（用于 PagedKVCache）
            is_prefill: 是否为 prefill 阶段

        Returns:
            AttentionOutput: attention 后输出与 post-attn norm
        """
        ...

    # ----------------------------------------------------------
    # Routing（路由计算，与 expert 执行分离）
    # ----------------------------------------------------------
    @abstractmethod
    def route_experts(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> RoutingResult:
        """
        计算 MoE 路由，返回路由结果供引擎做调度决策。

        注意：此方法 **仅计算路由**，不执行 expert forward。
        路由与执行分离是为了让引擎在两者之间插入调度逻辑
        （prefetch、CPU/GPU 分配、expert 替换等）。

        Args:
            layer_idx: 当前层索引
            hidden_states: 经过 post-attention layernorm 后的隐状态
                          [batch_size, seq_len, hidden_size]

        Returns:
            RoutingResult: 包含 topk_indices, topk_scores, activated_expert_ids
        """
        ...

    # ----------------------------------------------------------
    # MoE Forward（专家执行）
    # ----------------------------------------------------------
    @abstractmethod
    def forward_moe(
        self,
        layer_idx: int,
        attn_output: AttentionOutput,
        expert_placement: ExpertPlacement,
    ) -> torch.Tensor:
        """
        执行 MoE 部分（根据 placement 执行 expert + residual）。

        Args:
            layer_idx: 当前层索引
            attn_output: forward_attention 的输出
            expert_placement: 引擎提供的 expert 执行策略

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        ...

    # ----------------------------------------------------------
    # Compute Logits（Final Norm + LM Head）
    # ----------------------------------------------------------
    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        最后的 LayerNorm + LM Head 投影。

        Args:
            hidden_states: 最后一层的输出 [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        ...

    # ----------------------------------------------------------
    # 便捷方法：完整单层 forward（用于不需要中间调度的场景）
    # ----------------------------------------------------------
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
    ) -> LayerOutput:
        """
        完整的单层 forward（默认实现：组合三步调用）。
        可被子类 override 以实现 CUDA Graph 等优化。
        """
        attn_out = self.forward_attention(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            positions=positions,
            seq_ids=seq_ids,
            is_prefill=is_prefill,
        )
        moe_out = self.forward_moe(layer_idx, attn_out, expert_placement)
        return LayerOutput(hidden_states=moe_out)

    # ----------------------------------------------------------
    # 可选：CUDA Graph 支持接口
    # ----------------------------------------------------------
    def supports_cuda_graph(self) -> bool:
        """是否支持 CUDA Graph 加速"""
        return False

    def warmup_cuda_graph(
        self,
        batch_size: int,
        seq_len: int = 1,
        **kwargs
    ) -> None:
        """
        预热 CUDA Graph（capture 阶段）。
        仅在 supports_cuda_graph() 返回 True 时有意义。

        Args:
            batch_size: 预热时使用的 batch size
            seq_len: 预热时使用的 seq_len（decode 阶段通常为 1）
        """
        raise NotImplementedError("CUDA Graph not supported by this ModelRunner")