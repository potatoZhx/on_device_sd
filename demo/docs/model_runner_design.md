# 方案B：ModelRunner 抽象层设计文档

## 1. Target

### 1.1 核心问题

当前架构存在两个关键问题：

1. **layers/ 与 operators/ 接口不兼容**：`layers/` 中已实现的 Qwen3-30B-A3B 算子（`nn.Module` 风格、权重内部持有、设备无关）无法直接被推理引擎使用，因为引擎仅依赖 `operators/` 中的函数式无状态接口（`GPUOperators`、`CPUOperators`）。

2. **引擎与模型硬耦合**：所有引擎（`StandardDecodeEngine`、`PrefillEngine`、`DraftEngine`、`VerifyEngine`）直接实例化 `GPUOperators`/`CPUOperators`，并手动编排完整的 transformer forward 流程（embedding → layernorm → attention → router → expert → lm_head），硬编码参数键名格式（如 `f"layer_{layer_idx}.self_attn.q_proj"`）。适配不同模型需要修改所有引擎代码。

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| **解耦引擎与模型** | 引擎仅负责调度决策（prefetch、CPU/GPU 分配、speculative decoding），不关心模型内部结构 |
| **复用 layers/ 实现** | 已有的 `Qwen3Attention`、`Qwen3MoELayer`、`Qwen3DecoderLayer` 等可直接使用 |
| **支持多模型扩展** | 新增模型只需实现 `ModelRunner` 接口，无需修改任何引擎代码 |
| **保留异构执行控制** | 引擎仍能精确控制哪些 expert 在 CPU/GPU 执行 |
| **兼容算子优化** | 为 CUDA Graph、torch.compile、Triton kernel 等优化预留接口 |
| **渐进式迁移** | 可与现有代码共存，逐步迁移 |

### 1.3 Non-Goals（本阶段不处理）

- 完整的 OperatorBackend 抽象（方案C 范围）
- 自动模型发现/注册机制
- 分布式推理支持

---

## 2. Design Overview

### 2.1 架构变更

**变更前**：

```
Engine ──直接调用──> GPUOperators / CPUOperators
       ──直接读取──> ParameterLoader.static_params_gpu["layer_0.self_attn.q_proj"]
       ──手动编排──> embedding → layernorm → attn → router → expert → lm_head
```

**变更后**：

```
Engine ──调度决策──> ModelRunner（抽象接口）
                         │
              ┌──────────┴──────────┐
              │                     │
       Qwen3ModelRunner       FutureModelRunner
       (内部使用 layers/)     (内部使用其他实现)
              │
         可选: 内部使用
         operators/ 或 layers/
         或 CUDA Graph 等
```

### 2.2 核心设计原则

1. **引擎职责**：调度策略（何时 prefetch、哪些 expert 放 CPU/GPU、speculative decoding 策略、token acceptance）
2. **ModelRunner 职责**：模型计算（如何执行 attention、MoE routing、expert forward、norm）
3. **Expert 执行策略由引擎传递、由 ModelRunner 执行**：引擎通过 `ExpertPlacement` 告诉 ModelRunner "expert 3 在 CPU 执行"，ModelRunner 负责具体实现

### 2.3 分层对照

| 层级 | 现有实现 | 方案B |
|------|---------|-------|
| 引擎层 | 手写 forward + 调度 | **仅调度** |
| 模型抽象层 | 无（`MoELayer` 为占位符） | **新增 `ModelRunner` 接口** |
| 模型实现层 | `Qwen3MoEModel`（未被引擎使用） | **`Qwen3ModelRunner`（实现 `ModelRunner`）** |
| 算子层 | `operators/`（函数式） + `layers/`（Module 式） | **由 ModelRunner 内部自由选择** |

---

## 3. Use Cases

### UC-1: 标准自回归推理（Standard Decode）

**现有流程**：`StandardDecodeEngine` 手动调用 `gpu_ops.embedding()` → `gpu_ops.layernorm()` → `gpu_ops.self_attention()` → `gpu_ops.router()` → `gpu_ops/cpu_ops.expert_forward()` → `gpu_ops.linear()`

**方案B 流程**：
```
StandardDecodeEngine:
  1. hidden = model_runner.embed(input_ids)
  2. for layer_idx in range(num_layers):
       routing_result = model_runner.route_experts(layer_idx, hidden)
       placement = self._decide_placement(routing_result)    # 引擎的调度决策
       hidden, meta = model_runner.forward_layer(layer_idx, hidden, kv_cache, positions, placement)
       self._update_prefetch(meta)                           # 引擎的 prefetch 策略
  3. logits = model_runner.compute_logits(hidden)
```

### UC-2: Prefill 阶段（全模型推理 + prefetch）

**方案B 流程**：
```
PrefillEngine:
  1. hidden = model_runner.embed(input_ids)
  2. for layer_idx in range(num_layers):
       routing_result = model_runner.route_experts(layer_idx, hidden)
       placement = self._decide_placement_with_prefetch(routing_result, layer_idx)
       self.prefetcher.prefetch_for_next_layer(layer_idx, routing_result)   # 引擎的预取决策
       hidden, meta = model_runner.forward_layer(layer_idx, hidden, kv_cache, positions, placement)
  3. logits = model_runner.compute_logits(hidden)
```

### UC-3: Draft 阶段（专家替换 + CPU 执行）

**方案B 流程**：
```
DraftEngine:
  1. hidden = model_runner.embed(input_ids)
  2. for layer_idx in range(num_layers):
       routing_result = model_runner.route_experts(layer_idx, hidden)
       # 引擎的 draft 调度核心逻辑
       cpu_experts = draft_scheduler.select_cpu_experts(routing_result, top_c)
       substitution_map = draft_scheduler.select_gpu_substitutes(routing_result, cached_experts)
       placement = self._build_draft_placement(routing_result, cpu_experts, substitution_map)
       hidden, meta = model_runner.forward_layer(layer_idx, hidden, kv_cache, positions, placement)
       self._collect_draft_metrics(meta)
  3. logits = model_runner.compute_logits(hidden)
```

### UC-4: Verify 阶段

与 UC-2（Prefill）相同流程，验证引擎复用 prefill 逻辑。

### UC-5: 切换到新模型

```python
# 只需实现新的 ModelRunner，所有引擎零改动
class LlamaModelRunner(ModelRunner):
    def __init__(self, config, parameter_loader, expert_cache):
        # Llama 的 layers 实现
        ...

engine = StandardDecodeEngine(model_runner=LlamaModelRunner(...), ...)
```

### UC-6: CUDA Graph 优化

```python
class Qwen3CUDAGraphModelRunner(Qwen3ModelRunner):
    """在 Qwen3ModelRunner 基础上启用 CUDA Graph"""

    def __init__(self, ...):
        super().__init__(...)
        self._graph = None
        self._static_inputs = {}    # CUDA Graph 需要固定地址的 static input buffers
        self._static_outputs = {}

    def warmup_cuda_graph(self, batch_size: int, seq_len: int):
        """预热 CUDA Graph：Capture decode 阶段的 static 计算图"""
        # CUDA Graph 要求：
        # 1. 固定形状的 input/output buffer（decode 阶段 seq_len=1 天然满足）
        # 2. 没有动态控制流（if/for 依赖数据值的分支）
        # 3. 没有 CPU-GPU 同步点
        self._static_inputs = self._allocate_static_buffers(batch_size, seq_len)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_outputs = self._run_captured_forward(self._static_inputs)

    def forward_layer(self, layer_idx, hidden_states, kv_cache, positions,
                      expert_placement, **kwargs):
        """
        如果可以使用 CUDA Graph（decode、全 GPU expert），则 replay；
        否则 fallback 到父类的逐算子执行
        """
        if self._can_use_graph(expert_placement):
            # Copy input 到 static buffer → replay → copy output
            self._static_inputs['hidden'].copy_(hidden_states)
            self._graph.replay()
            return self._static_outputs['hidden'].clone(), {}
        else:
            return super().forward_layer(layer_idx, hidden_states, kv_cache,
                                         positions, expert_placement, **kwargs)
```

> **为什么方案B 兼容 CUDA Graph？**
>
> - `ModelRunner.forward_layer()` 封装了一层完整的 transformer layer 计算，内部无引擎干预 → 适合整体 capture
> - 引擎仅通过 `ExpertPlacement` 声明式地传递调度决策，不插入 CPU-GPU 同步点
> - `forward_layer()` 返回 metadata 而非中间状态 → 引擎不需要在 layer 内部插入同步逻辑
> - Decode 阶段 seq_len=1 + 全 GPU expert 时可以捕获为 static graph；有 CPU expert 时自动 fallback

---

## 4. API Design

### 4.1 核心抽象：`ModelRunner`

```python
# src/core/model_runner.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import torch

from .types import ExpertID, DeviceType
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
    # Layer Forward（单层前向传播）
    # ----------------------------------------------------------
    @abstractmethod
    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: torch.Tensor,
        expert_placement: ExpertPlacement,
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
    ) -> LayerOutput:
        """
        执行单个 transformer decoder layer 的完整前向传播。

        内部流程：
          1. Input LayerNorm
          2. Self-Attention + Residual
          3. Post-Attention LayerNorm
          4. MoE Expert Execution（根据 expert_placement）+ Residual

        Args:
            layer_idx: 当前层索引
            hidden_states: 输入隐状态 [batch_size, seq_len, hidden_size]
            kv_cache: KV Cache 对象（KVCache 或 PagedKVCache）
            positions: 位置索引 [batch_size, seq_len] 或 [num_tokens]
            expert_placement: 引擎提供的 expert 执行策略
            seq_ids: 序列 ID 列表（用于 PagedKVCache）
            is_prefill: 是否为 prefill 阶段

        Returns:
            LayerOutput: 包含 hidden_states 和 metadata
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
```

### 4.2 具体实现：`Qwen3ModelRunner`

```python
# src/model/qwen3_runner.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Set

from ..core.model_runner import ModelRunner, RoutingResult, ExpertPlacement, LayerOutput
from ..core.model import MoEConfig
from ..core.types import ExpertID
from ..memory.parameter_loader import ParameterLoader
from ..memory.expert_cache import ExpertCache
from ..layers import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MoELayer,
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

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: torch.Tensor,
        expert_placement: ExpertPlacement,
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
    ) -> LayerOutput:
        """
        执行单层 transformer decoder layer。
        Attention 部分使用 layers/ 的内部实现；
        MoE 部分根据 expert_placement 在 CPU/GPU 上执行。
        """
        layer = self.layers[layer_idx]

        # ---- 1. Input LayerNorm ----
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # ---- 2. Self-Attention ----
        # 适配维度
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states_flat = hidden_states.reshape(-1, hidden_size)
            positions_flat = positions.reshape(-1) if positions is not None else None
            reshape_back = True
        else:
            batch_size, seq_len = 1, hidden_states.shape[0]
            hidden_states_flat = hidden_states
            positions_flat = positions
            reshape_back = False

        if seq_ids is None:
            seq_ids = list(range(batch_size))

        attn_output = layer.self_attn(
            hidden_states=hidden_states_flat,
            positions=positions_flat,
            kv_cache=kv_cache,
            seq_ids=seq_ids,
            is_prefill=is_prefill,
        )

        if reshape_back:
            attn_output = attn_output.reshape(batch_size, seq_len, -1)

        hidden_states = residual + attn_output

        # ---- 3. Post-Attention LayerNorm ----
        residual = hidden_states
        normed = layer.post_attention_layernorm(hidden_states)

        # ---- 4. MoE Expert Execution（根据 placement）----
        moe_output = self._execute_moe_with_placement(
            hidden_states=normed,
            expert_placement=expert_placement,
        )

        hidden_states = residual + moe_output

        # ---- 收集 metadata ----
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

        # 合并 gpu + cpu + substitution 的所有 expert params
        all_experts = {}
        all_experts.update(expert_placement.gpu_expert_params)

        # 处理替换映射
        sub_map = expert_placement.substitution_map

        for expert_idx in range(self.config.num_experts):
            expert_mask = (topk_indices == expert_idx)
            token_expert_pairs = torch.where(expert_mask)
            token_indices = token_expert_pairs[0]
            k_indices = token_expert_pairs[1]

            if len(token_indices) == 0:
                continue

            weights = topk_scores[token_indices, k_indices]
            expert_input = flat[token_indices]

            # 决定执行路径
            if expert_idx in expert_placement.gpu_expert_params:
                # GPU 执行
                params = expert_placement.gpu_expert_params[expert_idx]
                expert_output = expert_forward_with_weights(
                    expert_input,
                    params['gate_proj'], params['up_proj'], params['down_proj'],
                )
            elif expert_idx in expert_placement.cpu_expert_params:
                # CPU 执行
                params = expert_placement.cpu_expert_params[expert_idx]
                expert_input_cpu = expert_input.cpu()
                expert_output_cpu = expert_forward_with_weights(
                    expert_input_cpu,
                    params['gate_proj'], params['up_proj'], params['down_proj'],
                )
                expert_output = expert_output_cpu.to(flat.device)
            elif expert_idx in sub_map:
                # 替换执行
                sub_idx = sub_map[expert_idx]
                if sub_idx in expert_placement.gpu_expert_params:
                    params = expert_placement.gpu_expert_params[sub_idx]
                    expert_output = expert_forward_with_weights(
                        expert_input,
                        params['gate_proj'], params['up_proj'], params['down_proj'],
                    )
                else:
                    continue  # 替代 expert 也不可用，跳过
            else:
                continue

            weighted_output = expert_output * weights.unsqueeze(1)
            final_output[token_indices] += weighted_output

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
        logger.info("Static weights loaded into Qwen3ModelRunner layers")
```

### 4.3 引擎侧改造：以 `PrefillEngine` 为例

```python
# src/execution/prefill_engine.py（改造后）

class PrefillEngine:
    def __init__(
        self,
        model_runner: ModelRunner,             # <-- 依赖注入
        expert_cache: ExpertCache,
        prefetcher: ExpertPrefetcher,
        metrics: MetricsCollector,
    ):
        self.model_runner = model_runner
        self.config = model_runner.get_config()
        self.expert_cache = expert_cache
        self.prefetcher = prefetcher
        self.metrics = metrics
        # 不再实例化 GPUOperators / CPUOperators
        # 不再持有 parameter_loader

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Any,
        positions: Optional[torch.Tensor] = None,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = True,
    ) -> Dict:
        self.metrics.start_phase('prefill')

        # 1. Embedding
        hidden_states = self.model_runner.embed(input_ids.cuda())

        # 2. Layer-by-layer
        for layer_idx in range(self.model_runner.get_num_layers()):
            # 2a. Post-attn norm 后的 hidden_states 用于 routing
            #     但 route_experts 需要在 forward_layer 之前调用
            #     因此先做 routing（使用当前 hidden_states 的 post-norm 版本）
            #     注意：这要求 route_experts 内部自行做 norm，
            #     或者我们传入 forward_layer 前的 hidden_states，
            #     由 forward_layer 内部先做 norm 再 route。
            #
            #     设计选择：route_experts 接受 raw hidden_states，
            #     内部自行做 input_norm + attn + post_norm + route。
            #     但这样会把 attention 也包进去，不利于分离。
            #
            #     最终选择：引擎先调用 forward_attention()，
            #     再调用 route_experts()，再调用 forward_moe()。
            #     但这会增加接口数量。
            #
            #     折中方案：forward_layer 内部自行调用 route（因为 routing
            #     需要 post-attn-norm 后的 hidden_states，而这在 layer 内部
            #     才能得到），但将 routing 结果通过 LayerOutput.metadata
            #     暴露给引擎。引擎在调用 forward_layer 之前就提供 placement，
            #     这意味着引擎需要"提前一步"的 routing 信息。
            #
            #     实际方案：将流程分为两步调用
            #       Step A: routing_result = model_runner.route_experts(layer_idx, hidden)
            #       Step B: placement = engine._decide(routing_result)
            #       Step C: output = model_runner.forward_layer(layer_idx, hidden, ..., placement)
            #     route_experts 内部做 input_norm + attention + post_norm + gate
            #     但不执行 MoE，将 attention 后的 hidden 缓存起来。
            #
            #     更简洁方案：route_experts 不做 attention，
            #     它接受的是"当前层 MoE 之前"的 hidden_states。
            #     forward_layer 先执行 attention 部分并缓存 post-attn hidden，
            #     然后引擎在外部调用 route_experts 和 forward_moe。
            #
            #     最终采用方案（平衡简洁与灵活）：
            #       forward_layer() 执行完整的一层（attn + moe）
            #       route_experts 由 forward_layer 内部调用，
            #       但 placement 是引擎在调用前基于 *上一层* routing 或
            #       prefetch 预测来构建的。对于当前层，引擎需要先获取
            #       routing 才能做 placement。
            #
            #     ★ 最终 API 设计：forward_layer 内部执行 attention，
            #       但 MoE 部分需要 placement。因此将 forward_layer 改为
            #       接受一个 `placement_fn` 回调，或者将 layer 拆为两步。
            #       为了简洁，我们采用"惰性 routing"方案：
            #       forward_layer 内部先做 attn 得到 post-norm hidden，
            #       然后调用 route_experts 计算 routing，
            #       再通过回调让引擎构建 placement。
            #
            #     ★★ 最终简化方案（推荐）：
            #       保持 forward_layer 为整层执行，
            #       引擎通过 "pre-routing" 策略构建 placement：
            #       - Prefill/Verify: 基于 prefetcher 预测 → 简单策略
            #       - Draft: 先 route → placement → forward_layer
            #       为此新增一个轻量 API：
            #         forward_attention() → post_attn_hidden
            #       这样引擎可以：
            #         post_attn = model_runner.forward_attention(layer_idx, hidden, kv, pos)
            #         routing = model_runner.route_experts(layer_idx, post_attn)
            #         placement = engine._decide(routing)
            #         hidden = model_runner.forward_moe(layer_idx, post_attn, placement)

            # --- 实际执行（简化后的两步方案）---
            # Step 1: Attention + Norm（不含 MoE）
            attn_output = self.model_runner.forward_attention(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                positions=positions,
                seq_ids=seq_ids,
                is_prefill=is_prefill,
            )

            # Step 2: Route
            routing_result = self.model_runner.route_experts(
                layer_idx=layer_idx,
                hidden_states=attn_output.post_attn_normed,
            )

            # Step 3: 引擎调度决策
            placement = self._build_placement(routing_result)

            # Step 4: Prefetch for next layer
            self.prefetcher.prefetch_for_next_layer(
                current_layer_idx=layer_idx,
                routing_result=routing_result,
            )

            # Step 5: MoE forward
            hidden_states = self.model_runner.forward_moe(
                layer_idx=layer_idx,
                attn_output=attn_output,
                expert_placement=placement,
            )

        # 3. Logits
        logits = self.model_runner.compute_logits(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        self.metrics.end_phase('prefill')
        return {'logits': logits, 'next_token_id': next_token_id}
```

> **注意**：上述推导过程保留在注释中以展示设计权衡。最终 API 将 `forward_layer` 拆为 `forward_attention` + `route_experts` + `forward_moe` 三步。

### 4.4 修订后的 ModelRunner 接口（最终版）

基于上述推导，最终的接口拆分为更细粒度的三步调用，以满足引擎在 routing 和 expert 执行之间插入调度逻辑的需求：

```python
class ModelRunner(ABC):
    """模型前向推理的统一抽象（最终版）"""

    # ---- 基本信息 ----
    @abstractmethod
    def get_config(self) -> MoEConfig: ...

    @abstractmethod
    def get_num_layers(self) -> int: ...

    # ---- Embedding ----
    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    # ---- Attention 部分（包含 input_norm + attn + residual + post_attn_norm）----
    @abstractmethod
    def forward_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: torch.Tensor,
        *,
        seq_ids: Optional[List[int]] = None,
        is_prefill: bool = False,
    ) -> "AttentionOutput": ...

    # ---- Routing（仅计算路由，不执行 expert）----
    @abstractmethod
    def route_experts(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> RoutingResult: ...

    # ---- MoE 部分（根据 placement 执行 expert + residual）----
    @abstractmethod
    def forward_moe(
        self,
        layer_idx: int,
        attn_output: "AttentionOutput",
        expert_placement: ExpertPlacement,
    ) -> torch.Tensor: ...

    # ---- Final Norm + LM Head ----
    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    # ---- 便捷方法：完整单层 forward（用于不需要中间调度的场景）----
    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        positions: torch.Tensor,
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
            layer_idx, hidden_states, kv_cache, positions,
            seq_ids=seq_ids, is_prefill=is_prefill
        )
        # 注意：forward_layer 中无法插入引擎调度，
        # routing 使用 placement 中已提供的 routing_result
        moe_out = self.forward_moe(layer_idx, attn_out, expert_placement)
        return LayerOutput(hidden_states=moe_out)

    # ---- CUDA Graph 支持 ----
    def supports_cuda_graph(self) -> bool:
        return False

    def warmup_cuda_graph(self, batch_size: int, seq_len: int = 1) -> None:
        raise NotImplementedError


@dataclass
class AttentionOutput:
    """forward_attention 的返回结果"""
    hidden_states: torch.Tensor      # attention 后加了 residual 的 hidden_states
    post_attn_normed: torch.Tensor   # post_attention_layernorm 后的结果（用于 routing 和 MoE）
    residual: torch.Tensor           # MoE 之前的 residual（= hidden_states）
```

---

## 5. API Call Dependency（调用依赖关系）

### 5.1 模块依赖图

```
                    ┌─────────────────────┐
                    │    Orchestrator      │
                    │  (调度协调)           │
                    └──────────┬──────────┘
                               │ 创建并调度
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │  Prefill    │  │   Draft     │  │   Verify    │
     │  Engine     │  │   Engine    │  │   Engine    │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │ 依赖（依赖注入）
                             ▼
                    ┌─────────────────────┐
                    │   ModelRunner       │
                    │   (抽象接口)         │
                    └──────────┬──────────┘
                               │ 实现
                               ▼
                    ┌─────────────────────┐
                    │ Qwen3ModelRunner    │
                    │ (具体实现)           │
                    └──────────┬──────────┘
                               │ 内部使用
                    ┌──────────┼──────────┐
                    │          │          │
                    ▼          ▼          ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐
              │ layers/  │ │Param  │ │ Expert   │
              │ (nn.Mod) │ │ Loader │ │ Cache    │
              └──────────┘ └────────┘ └──────────┘
```

### 5.2 构造依赖（初始化阶段）

```
# 构建顺序（从底向上）

1. MoEConfig.from_pretrained(model_path)
        │
2. ParameterLoader(model_path, config)
        │   └──> 加载 safetensors 到 CPU/GPU
        │
3. ExpertCache(max_cache_size_gb, expert_size_mb, replacement_strategy)
        │
4. Qwen3ModelRunner(config, parameter_loader)          # <-- 新增
        │   └──> 内部创建 Qwen3DecoderLayer[] 并 load_weights
        │
5. ExpertPrefetcher(strategy)
        │
6. DraftSchedulingStrategy(config)
        │
7. PrefillEngine(model_runner, expert_cache, prefetcher, metrics)    # <-- 改造
   DraftEngine(model_runner, expert_cache, draft_scheduler, metrics)  # <-- 改造
   VerifyEngine(model_runner, expert_cache, prefetcher, metrics)      # <-- 改造
   StandardDecodeEngine(model_runner, expert_cache, prefetcher, metrics)
        │
8. InferenceOrchestrator(config, engines..., schedulers..., strategies...)
```

### 5.3 运行时调用序列（Prefill 阶段）

```
PrefillEngine.forward(input_ids, kv_cache)
    │
    ├──> model_runner.embed(input_ids)
    │        └──> F.embedding(input_ids, static_params['embed_tokens'])
    │
    ├──> for layer_idx in range(num_layers):
    │    │
    │    ├──> model_runner.forward_attention(layer_idx, hidden, kv_cache, positions)
    │    │        ├──> layer.input_layernorm(hidden)
    │    │        ├──> layer.self_attn(hidden, positions, kv_cache, seq_ids, is_prefill)
    │    │        │        ├──> q_proj, k_proj, v_proj
    │    │        │        ├──> QK norm
    │    │        │        ├──> RoPE
    │    │        │        ├──> KV cache store
    │    │        │        └──> flash_attn / standard attn
    │    │        ├──> residual + attn_output
    │    │        └──> layer.post_attention_layernorm(hidden)
    │    │
    │    ├──> model_runner.route_experts(layer_idx, post_attn_normed)
    │    │        └──> layer.mlp.gate(hidden, top_k) → topk_indices, topk_scores
    │    │
    │    ├──> engine._build_placement(routing_result)       # 引擎调度决策
    │    │        ├──> expert_cache.is_cached(expert_id)
    │    │        └──> parameter_loader.get_expert_params(expert_id, device)
    │    │
    │    ├──> prefetcher.prefetch_for_next_layer(...)        # 引擎预取决策
    │    │
    │    └──> model_runner.forward_moe(layer_idx, attn_output, placement)
    │             ├──> for expert_idx: expert_forward_with_weights(...)
    │             │        └──> F.linear (gate, up, down) + SiLU
    │             └──> residual + moe_output
    │
    └──> model_runner.compute_logits(hidden)
             ├──> final_norm(hidden)
             └──> F.linear(hidden, lm_head_weight)
```

### 5.4 运行时调用序列（Draft 阶段）

```
DraftEngine.forward(input_ids, kv_cache, max_draft_tokens)
    │
    ├──> for step in range(max_draft_tokens):
    │    │
    │    ├──> model_runner.embed(current_token)
    │    │
    │    ├──> for layer_idx in range(num_layers):
    │    │    │
    │    │    ├──> model_runner.forward_attention(...)
    │    │    │
    │    │    ├──> model_runner.route_experts(...)
    │    │    │
    │    │    ├──> engine._build_draft_placement(routing_result)    # Draft 特有调度
    │    │    │        ├──> draft_scheduler.select_cpu_experts(routing, top_c)
    │    │    │        ├──> draft_scheduler.select_gpu_substitutes(...)
    │    │    │        ├──> expert_cache.is_cached(expert_id)
    │    │    │        └──> parameter_loader.get_expert_params(expert_id)
    │    │    │
    │    │    └──> model_runner.forward_moe(...)
    │    │
    │    ├──> logits = model_runner.compute_logits(hidden)
    │    └──> sample next token
    │
    └──> draft_scheduler.should_trigger_verify(metrics)
```

---

## 6. Test Cases

### 6.1 单元测试

#### TC-1: ModelRunner 接口合规性测试

```python
# tests/unit/test_model_runner_interface.py

class TestModelRunnerInterface:
    """验证 Qwen3ModelRunner 正确实现了 ModelRunner 接口"""

    def test_is_subclass(self):
        """Qwen3ModelRunner 是 ModelRunner 的子类"""
        assert issubclass(Qwen3ModelRunner, ModelRunner)

    def test_get_config(self, qwen3_runner):
        """get_config 返回 MoEConfig"""
        config = qwen3_runner.get_config()
        assert isinstance(config, MoEConfig)
        assert config.num_experts > 0
        assert config.num_hidden_layers > 0

    def test_get_num_layers(self, qwen3_runner):
        """get_num_layers 返回正确的层数"""
        assert qwen3_runner.get_num_layers() == qwen3_runner.get_config().num_hidden_layers

    def test_embed_output_shape(self, qwen3_runner):
        """embed 输出形状正确"""
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch, seq_len), device='cuda')
        output = qwen3_runner.embed(input_ids)
        config = qwen3_runner.get_config()
        assert output.shape == (batch, seq_len, config.hidden_size)

    def test_compute_logits_output_shape(self, qwen3_runner):
        """compute_logits 输出形状正确"""
        config = qwen3_runner.get_config()
        batch, seq_len = 2, 16
        hidden = torch.randn(batch, seq_len, config.hidden_size, device='cuda', dtype=config.get_dtype())
        logits = qwen3_runner.compute_logits(hidden)
        assert logits.shape == (batch, seq_len, config.vocab_size)
```

#### TC-2: RoutingResult 测试

```python
# tests/unit/test_routing.py

class TestRouting:
    """验证路由计算的正确性"""

    def test_route_experts_output_format(self, qwen3_runner):
        """route_experts 返回正确格式的 RoutingResult"""
        config = qwen3_runner.get_config()
        hidden = torch.randn(1, 4, config.hidden_size, device='cuda', dtype=config.get_dtype())
        result = qwen3_runner.route_experts(layer_idx=0, hidden_states=hidden)

        assert isinstance(result, RoutingResult)
        assert result.layer_idx == 0
        assert result.topk_indices.shape[-1] == config.num_experts_per_token
        assert result.topk_scores.shape == result.topk_indices.shape

    def test_routing_scores_sum_to_one(self, qwen3_runner):
        """路由权重归一化：每个 token 的 top-k 权重和为 1"""
        config = qwen3_runner.get_config()
        hidden = torch.randn(1, 4, config.hidden_size, device='cuda', dtype=config.get_dtype())
        result = qwen3_runner.route_experts(layer_idx=0, hidden_states=hidden)

        score_sums = result.topk_scores.sum(dim=-1)
        assert torch.allclose(score_sums, torch.ones_like(score_sums), atol=1e-5)

    def test_activated_expert_ids_consistency(self, qwen3_runner):
        """activated_expert_ids 与 topk_indices 一致"""
        config = qwen3_runner.get_config()
        hidden = torch.randn(1, 4, config.hidden_size, device='cuda', dtype=config.get_dtype())
        result = qwen3_runner.route_experts(layer_idx=0, hidden_states=hidden)

        expected_ids = set()
        for idx in result.topk_indices.flatten().unique().tolist():
            expected_ids.add(ExpertID(0, idx))

        assert result.activated_expert_ids == expected_ids
```

#### TC-3: ExpertPlacement 构建测试

```python
# tests/unit/test_expert_placement.py

class TestExpertPlacement:
    """验证引擎正确构建 ExpertPlacement"""

    def test_all_gpu_placement(self, expert_cache, parameter_loader):
        """所有 expert 在 GPU 时的 placement"""
        routing = RoutingResult(
            layer_idx=0,
            topk_indices=torch.tensor([[0, 1]]),
            topk_scores=torch.tensor([[0.6, 0.4]]),
            activated_expert_ids={ExpertID(0, 0), ExpertID(0, 1)},
        )
        # 模拟所有 expert 都在 cache 中
        placement = build_prefill_placement(routing, expert_cache, parameter_loader)
        assert len(placement.gpu_expert_params) == 2
        assert len(placement.cpu_expert_params) == 0

    def test_mixed_cpu_gpu_placement(self, expert_cache, parameter_loader):
        """CPU/GPU 混合 placement"""
        routing = RoutingResult(
            layer_idx=0,
            topk_indices=torch.tensor([[0, 5]]),
            topk_scores=torch.tensor([[0.6, 0.4]]),
            activated_expert_ids={ExpertID(0, 0), ExpertID(0, 5)},
        )
        # expert 0 在 GPU cache，expert 5 不在
        placement = build_prefill_placement(routing, expert_cache, parameter_loader)
        assert 0 in placement.gpu_expert_params
        assert 5 in placement.cpu_expert_params

    def test_draft_substitution_placement(self, expert_cache, draft_scheduler):
        """Draft 阶段的替换 placement"""
        routing = RoutingResult(
            layer_idx=0,
            topk_indices=torch.tensor([[0, 5, 10, 20]]),
            topk_scores=torch.tensor([[0.3, 0.3, 0.2, 0.2]]),
            activated_expert_ids={ExpertID(0, 0), ExpertID(0, 5), ExpertID(0, 10), ExpertID(0, 20)},
        )
        placement = build_draft_placement(routing, expert_cache, draft_scheduler)
        # 验证替换映射存在且指向 cached expert
        for orig, sub in placement.substitution_map.items():
            assert sub in placement.gpu_expert_params
```

#### TC-4: forward_attention 测试

```python
# tests/unit/test_forward_attention.py

class TestForwardAttention:
    """验证 attention 前向传播"""

    def test_output_shape(self, qwen3_runner, kv_cache):
        """forward_attention 输出形状正确"""
        config = qwen3_runner.get_config()
        batch, seq_len = 1, 8
        hidden = torch.randn(batch, seq_len, config.hidden_size, device='cuda', dtype=config.get_dtype())
        positions = torch.arange(seq_len, device='cuda').unsqueeze(0)

        attn_out = qwen3_runner.forward_attention(
            layer_idx=0, hidden_states=hidden, kv_cache=kv_cache,
            positions=positions, is_prefill=True,
        )

        assert isinstance(attn_out, AttentionOutput)
        assert attn_out.hidden_states.shape == hidden.shape
        assert attn_out.post_attn_normed.shape == hidden.shape
        assert attn_out.residual.shape == hidden.shape

    def test_residual_connection(self, qwen3_runner, kv_cache):
        """验证 residual 连接正确（attn_output.residual == attn_output.hidden_states）"""
        config = qwen3_runner.get_config()
        hidden = torch.randn(1, 4, config.hidden_size, device='cuda', dtype=config.get_dtype())
        positions = torch.arange(4, device='cuda').unsqueeze(0)

        attn_out = qwen3_runner.forward_attention(
            layer_idx=0, hidden_states=hidden, kv_cache=kv_cache,
            positions=positions, is_prefill=True,
        )

        # residual 就是 attention 后加了 skip connection 的结果
        assert torch.equal(attn_out.residual, attn_out.hidden_states)
```

#### TC-5: forward_moe 测试

```python
# tests/unit/test_forward_moe.py

class TestForwardMoE:
    """验证 MoE 前向传播"""

    def test_output_shape(self, qwen3_runner, mock_attn_output, mock_placement):
        """forward_moe 输出形状正确"""
        config = qwen3_runner.get_config()
        output = qwen3_runner.forward_moe(
            layer_idx=0, attn_output=mock_attn_output, expert_placement=mock_placement,
        )
        assert output.shape == mock_attn_output.hidden_states.shape

    def test_gpu_only_execution(self, qwen3_runner, mock_attn_output):
        """所有 expert 在 GPU 执行时结果非零"""
        # 构建全 GPU placement
        placement = ExpertPlacement(
            gpu_expert_params={0: gpu_params_0, 1: gpu_params_1},
            cpu_expert_params={},
            routing_result=routing_result,
        )
        output = qwen3_runner.forward_moe(0, mock_attn_output, placement)
        assert output.abs().sum() > 0  # 非零输出

    def test_cpu_execution_correctness(self, qwen3_runner, mock_attn_output):
        """CPU expert 执行结果与 GPU 执行结果接近（数值精度差异）"""
        # 同一 expert，分别用 GPU 和 CPU 参数构建 placement
        gpu_placement = ExpertPlacement(gpu_expert_params={0: params_gpu}, ...)
        cpu_placement = ExpertPlacement(cpu_expert_params={0: params_cpu}, ...)

        gpu_out = qwen3_runner.forward_moe(0, mock_attn_output, gpu_placement)
        cpu_out = qwen3_runner.forward_moe(0, mock_attn_output, cpu_placement)

        assert torch.allclose(gpu_out, cpu_out, atol=1e-3)

    def test_substitution_uses_substitute_params(self, qwen3_runner, mock_attn_output):
        """替换模式下使用替代 expert 的参数"""
        placement = ExpertPlacement(
            gpu_expert_params={2: sub_params},  # expert 2 在 GPU
            cpu_expert_params={},
            substitution_map={5: 2},            # expert 5 → expert 2
            routing_result=routing_with_expert_5,
        )
        output = qwen3_runner.forward_moe(0, mock_attn_output, placement)
        assert output.abs().sum() > 0
```

### 6.2 集成测试

#### TC-6: 端到端 Prefill 测试

```python
# tests/integration/test_prefill_with_model_runner.py

class TestPrefillIntegration:
    """端到端测试 PrefillEngine + Qwen3ModelRunner"""

    def test_prefill_produces_logits(self, model_runner, expert_cache, prefetcher, metrics):
        """Prefill 阶段能正常产生 logits"""
        engine = PrefillEngine(
            model_runner=model_runner,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=metrics,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
        kv_cache = create_test_kv_cache(model_runner.get_config())

        output = engine.forward(input_ids, kv_cache, is_prefill=True)

        assert 'logits' in output
        assert 'next_token_id' in output
        assert output['logits'].dim() == 3  # [batch, seq, vocab]

    def test_prefill_kv_cache_updated(self, model_runner, expert_cache, prefetcher, metrics):
        """Prefill 后 KV cache 被正确更新"""
        engine = PrefillEngine(model_runner=model_runner, ...)
        input_ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
        kv_cache = create_test_kv_cache(model_runner.get_config())

        engine.forward(input_ids, kv_cache, is_prefill=True)

        # KV cache 应该有内容
        assert kv_cache.current_length > 0
```

#### TC-7: 端到端 Draft-Verify 测试

```python
# tests/integration/test_draft_verify_with_model_runner.py

class TestDraftVerifyIntegration:
    """端到端测试 Draft + Verify"""

    def test_draft_produces_tokens(self, model_runner, expert_cache, draft_scheduler, metrics):
        """Draft 阶段能生成 draft tokens"""
        engine = DraftEngine(
            model_runner=model_runner,
            expert_cache=expert_cache,
            draft_scheduler=draft_scheduler,
            metrics=metrics,
        )
        input_ids = torch.tensor([1], device='cuda')
        kv_cache = create_test_kv_cache(model_runner.get_config())

        result = engine.forward(input_ids, kv_cache, max_draft_tokens=4)

        assert 'drafted_tokens' in result
        assert len(result['drafted_tokens']) > 0
        assert len(result['drafted_tokens']) <= 4

    def test_verify_accepts_correct_tokens(self, model_runner, expert_cache, prefetcher, metrics):
        """Verify 阶段能正确验证 draft tokens"""
        verify_engine = VerifyEngine(
            model_runner=model_runner,
            expert_cache=expert_cache,
            prefetcher=prefetcher,
            metrics=metrics,
        )
        # 使用与 prefill 相同的输入验证
        all_ids = torch.tensor([[1, 2, 3, 4, 5]], device='cuda')  # prompt + draft
        kv_cache = create_test_kv_cache(model_runner.get_config())

        output = verify_engine.forward(all_ids, kv_cache)
        assert 'logits' in output
```

#### TC-8: 模型可替换性测试

```python
# tests/integration/test_model_swappability.py

class TestModelSwappability:
    """验证不同 ModelRunner 实现可互换使用"""

    def test_engine_accepts_any_model_runner(self):
        """引擎接受任意 ModelRunner 实现"""
        class MockModelRunner(ModelRunner):
            """用于测试的最小 ModelRunner 实现"""
            def get_config(self): return mock_config
            def get_num_layers(self): return 2
            def embed(self, input_ids): return torch.randn(1, 4, 128)
            def forward_attention(self, **kw): return mock_attn_output
            def route_experts(self, **kw): return mock_routing
            def forward_moe(self, **kw): return torch.randn(1, 4, 128)
            def compute_logits(self, hidden): return torch.randn(1, 4, 1000)

        # 所有引擎都应该能使用 MockModelRunner
        engine = PrefillEngine(model_runner=MockModelRunner(), ...)
        # 应能正常初始化而不报错
        assert engine.model_runner is not None
```

### 6.3 性能测试

#### TC-9: CUDA Graph 可行性测试

```python
# tests/performance/test_cuda_graph.py

class TestCUDAGraphFeasibility:
    """验证 CUDA Graph 的可行性"""

    def test_decode_step_capturable(self, qwen3_runner):
        """Decode 阶段（seq_len=1, 全 GPU expert）可被 CUDA Graph capture"""
        config = qwen3_runner.get_config()

        # 准备 static input
        hidden = torch.randn(1, 1, config.hidden_size, device='cuda', dtype=config.get_dtype())
        positions = torch.tensor([[0]], device='cuda')

        # 构造全 GPU placement
        placement = make_all_gpu_placement(...)

        # 尝试 capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            attn_out = qwen3_runner.forward_attention(0, hidden, kv_cache, positions)
            out = qwen3_runner.forward_moe(0, attn_out, placement)

        # Replay 应该成功
        g.replay()
        assert out is not None

    def test_forward_layer_no_cpu_sync(self, qwen3_runner):
        """全 GPU 路径下 forward_layer 无 CPU-GPU 同步"""
        # 使用 torch profiler 验证无 cudaStreamSynchronize
        ...
```

---

## 7. Migration Plan（迁移计划）

### Phase 1：新增抽象层（不影响现有代码）

1. 新增 `src/core/model_runner.py`：定义 `ModelRunner`、`RoutingResult`、`ExpertPlacement`、`LayerOutput`、`AttentionOutput`
2. 新增 `src/model/qwen3_runner.py`：实现 `Qwen3ModelRunner`
3. 新增对应的单元测试

### Phase 2：引擎适配（渐进式）

1. 创建新版引擎文件（如 `prefill_engine_v2.py`），使用 `ModelRunner` 接口
2. 在 Orchestrator 中通过配置选择新旧引擎
3. 对比新旧引擎的输出一致性

### Phase 3：切换与清理

1. 确认新引擎通过所有集成测试
2. 将新引擎替换旧引擎
3. 清理废弃的直接 operator 调用代码

### Phase 4：优化（后续）

1. 实现 `Qwen3CUDAGraphModelRunner`
2. 接入 `torch.compile`
3. 实现新模型的 `ModelRunner`（如 `LlamaModelRunner`）

---

## 8. 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新增** | `src/core/model_runner.py` | `ModelRunner` 抽象接口 + 数据结构 |
| **新增** | `src/model/qwen3_runner.py` | `Qwen3ModelRunner` 实现 |
| **修改** | `src/execution/prefill_engine.py` | 依赖 `ModelRunner` 而非 `GPUOperators` |
| **修改** | `src/execution/draft_engine.py` | 同上 |
| **修改** | `src/execution/verify_engine.py` | 同上 |
| **修改** | `src/execution/standard_engine.py` | 同上 |
| **修改** | `src/execution/orchestrator.py` | 传入 `ModelRunner` 给各引擎 |
| **新增** | `tests/unit/test_model_runner_interface.py` | 接口合规性测试 |
| **新增** | `tests/unit/test_routing.py` | 路由测试 |
| **新增** | `tests/unit/test_expert_placement.py` | Placement 构建测试 |
| **新增** | `tests/unit/test_forward_attention.py` | Attention 测试 |
| **新增** | `tests/unit/test_forward_moe.py` | MoE 测试 |
| **新增** | `tests/integration/test_prefill_with_model_runner.py` | Prefill 集成测试 |
| **新增** | `tests/integration/test_draft_verify_with_model_runner.py` | Draft-Verify 集成测试 |
