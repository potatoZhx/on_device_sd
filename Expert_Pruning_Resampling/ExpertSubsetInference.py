import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
# ==========================================
# 1. 基类: 定义通用接口和核心选择逻辑
# ==========================================
class BaseExpertSubsetBlockWrapper(nn.Module):
    def __init__(self, original_block, use_top_m, mode="remove"):
        """
        Args:
            original_block: 原始 MoE 模块
            use_top_m: 
                - 在 mode="remove" 时，表示保留前 m 个专家
                - 在 mode="replace" 时，表示要替换的第 m 个专家 (Rank k)
            mode: "remove", "replace"
        """
        super().__init__()
        self.original_block = original_block
        self.use_top_m = use_top_m
        self.mode = mode
        
        # 属性由子类填充
        self.num_experts = getattr(original_block, "num_experts", None)
        self.top_k = getattr(original_block, "top_k", None)
        
    def _validate_params(self):
        """通用参数校验"""
        if self.top_k is None:
            raise ValueError("top_k is not initialized properly.")
        
 
        if self.use_top_m < 0:
            raise ValueError(f"use_top_m ({self.use_top_m}) cannot be negative")
        if self.use_top_m > self.top_k:
            raise ValueError(f"use_top_m ({self.use_top_m}) cannot be > top_k ({self.top_k})")

    def _process_subset_selection(self, weights, indices):
        """
        核心算法：根据 mode 修改 TopK 的权重和索引
        输入形状: [Batch_Size, Top_K]
        """
        device = weights.device
        batch_size_flat, top_k = weights.shape
        
        # 复制以避免修改原计算图
        new_weights = weights.clone()
        new_indices = indices.clone()

        if self.mode == "remove":
            # 模式: 只保留前 m 个
            # 截断 Tensor
            new_weights = weights[:, :self.use_top_m]
            new_indices = indices[:, :self.use_top_m]
        
        elif self.mode == "replace":
            # 模式: 仅替换第 k 个专家 (Rank k Replacement)
            # use_top_m 在这里代表 Rank k (1-based)
            target_rank = self.use_top_m 
            target_idx = target_rank - 1 # 0-based index
            
            # 1. 生成全量噪声
            noise = torch.rand(batch_size_flat, self.num_experts, device=device)
            
            # 2. 屏蔽掉当前所有的 Top-K 专家 (包括我们要替换的那个)
            # 这样保证新选出来的专家一定不在原来的 Top-K 列表中
            noise.scatter_(1, indices, -float('inf'))
            
            # 3. 选 1 个随机新专家
            _, random_expert = torch.topk(noise, k=1, dim=-1)
            
            # 4. 仅替换目标位置
            new_indices[:, target_idx] = random_expert.squeeze(-1)
            
        elif self.mode == "replace_remaining":
            # 模式: 保留前 m 个专家，其他专家随机替换
            # use_top_m 在这里代表要保留的专家数量
            keep_count = self.use_top_m
            
            # 只有当需要替换的数量大于0时才执行
            if keep_count < top_k:
                replace_count = top_k - keep_count
                
                # 1. 生成噪声池 [N, Num_Experts]
                noise = torch.rand(batch_size_flat, self.num_experts, device=device)
                
                # 2. 屏蔽掉要保留的前m个专家
                keep_experts = indices[:, :keep_count]
                noise.scatter_(1, keep_experts, -float('inf'))
                
                # 3. 随机选 replace_count 个新专家
                _, random_experts = torch.topk(noise, k=replace_count, dim=-1)
                
                # 4. 替换后面的专家
                new_indices[:, keep_count:] = random_experts
                
        elif self.mode == "replace_with_topp":
            # 模式: 保留前m个专家，其他专家在topp阈值内随机替换
            # use_top_m 在这里代表要保留的专家数量
            keep_count = self.use_top_m
            
            # 假设topp阈值是0.9，这个值可以根据需要调整
            topp_threshold = 0.9
            
            # 只有当需要替换的数量大于0时才执行
            if keep_count < top_k:
                # 1. 计算router logits的softmax概率
                # 注意：这里需要重新计算概率，因为weights可能已经是topk后的结果
                # 假设我们可以通过原始模型获取完整的logits
                # 这里我们使用一个近似方法，通过当前的topk权重和索引来模拟
                
                # 2. 生成噪声池，但只在topp阈值内的专家中选择
                # 首先计算所有专家的权重（近似）
                all_weights = torch.zeros(batch_size_flat, self.num_experts, device=device)
                all_weights.scatter_(1, indices, weights)
                
                # 3. 对权重进行排序，找到topp阈值内的专家
                sorted_weights, sorted_indices = torch.sort(all_weights, dim=-1, descending=True)
                cumulative_weights = torch.cumsum(sorted_weights, dim=-1)
                topp_mask = cumulative_weights <= topp_threshold
                
                # 至少保留一个专家
                topp_mask[:, 0] = True
                
                # 4. 将topp阈值外的专家权重设置为-无穷
                noise = torch.rand(batch_size_flat, self.num_experts, device=device)
                noise.masked_fill_(~topp_mask.scatter_(1, sorted_indices, topp_mask), -float('inf'))
                
                # 5. 屏蔽掉要保留的前m个专家
                keep_experts = indices[:, :keep_count]
                noise.scatter_(1, keep_experts, -float('inf'))
                
                # 6. 随机选需要替换数量的新专家
                replace_count = top_k - keep_count
                _, random_experts = torch.topk(noise, k=replace_count, dim=-1)
                
                # 7. 替换后面的专家
                new_indices[:, keep_count:] = random_experts
                
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return new_weights, new_indices

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")


# ==========================================
# 2. Qwen 专用包装器
# ==========================================
class QwenExpertSubsetBlockWrapper(BaseExpertSubsetBlockWrapper):
    def __init__(self, original_block, use_top_m, mode="remove"):
        super().__init__(original_block, use_top_m, mode)
        
        # 兼容不同模型的属性名
        self.norm_topk_prob = getattr(original_block, "norm_topk_prob", False) # Mixtral 默认无此属性
        self.gate = original_block.gate
        self.experts = original_block.experts
        
        # Mixtral/Phi 通常没有共享专家，设为 None
        self.shared_expert = getattr(original_block, "shared_expert", None)
        self.shared_expert_gate = getattr(original_block, "shared_expert_gate", None)
        
        self._validate_params()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. 计算 Logits
        router_logits_raw = self.gate(hidden_states_flat)
        
        # 2. TopK
        routing_weights = F.softmax(router_logits_raw, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 3. 核心修改逻辑
        routing_weights, selected_experts = self._process_subset_selection(routing_weights, selected_experts)
        
        # 4. 归一化 (仅当模型需要时)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # 5. 专家计算
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            expert_layer = self.experts[expert_idx]
            
            mask_slice = expert_mask[expert_idx]
            top_m_indices, sample_indices = torch.where(mask_slice)
            
            current_state = hidden_states_flat[sample_indices]
            current_weights = routing_weights[sample_indices, top_m_indices]
            
            # 兼容：Mixtral 的 Expert 可能也是一个 Block，输出可能包含 cache_loss
            # 但 standard transformers implementation 的 experts[i] 只是 MLP
            expert_out = expert_layer(current_state)
            
            # 处理可能的 Tuple 返回 (某些实现可能会返回 cache_loss)
            if isinstance(expert_out, tuple):
                expert_out = expert_out[0]
            
            current_hidden_states = expert_out * current_weights.unsqueeze(1)
            final_hidden_states.index_add_(0, sample_indices, current_hidden_states.to(hidden_states.dtype))

        # 6. Shared Expert (如果存在)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states_flat)
            if self.shared_expert_gate is not None:
                shared_out = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_out
            final_hidden_states += shared_out
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits_raw


# ==========================================
# 3. DeepSeek-V2 专用包装器
# ==========================================
class DeepseekExpertSubsetBlockWrapper(BaseExpertSubsetBlockWrapper):
    def __init__(self, original_block, use_top_m, mode="remove"):
        super().__init__(original_block, use_top_m, mode)
        
        self.config = original_block.config
        self.num_experts = self.config.n_routed_experts
        self.top_k = self.config.num_experts_per_tok
        
        self.gate = original_block.gate
        # 直接引用专家列表，避开分布式逻辑
        self.experts = original_block.experts 
        
        self._validate_params()

    def forward(self, hidden_states: torch.Tensor):
        residuals = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. Gate 计算
        gate_output = self.gate(hidden_states)
        topk_indices = gate_output[0] 
        topk_weights = gate_output[1]
        
        # 2. 排序 (DeepSeek TopK 默认未排序，需手动排序以支持截断)
        topk_weights_sorted, sort_idx = torch.sort(topk_weights, dim=-1, descending=True)
        topk_indices_sorted = torch.gather(topk_indices, -1, sort_idx)
        
        # 展平 [N, K]
        flat_weights = topk_weights_sorted.view(-1, self.top_k)
        flat_indices = topk_indices_sorted.view(-1, self.top_k)
        
        # 3. 修改选择 (调用基类逻辑)
        new_weights, new_indices = self._process_subset_selection(flat_weights, flat_indices)
        
        # 4. 手动 Scatter-Gather 计算 (替代 moe_infer)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # 生成 Mask [Experts, Current_K, Tokens]
        # 注意：new_indices 的列数可能变化（remove模式下），需要动态获取
        current_k = new_indices.shape[1]
        
        expert_mask = torch.nn.functional.one_hot(new_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            
            # DeepSeek-V2-Lite 通常 ep_size=1，直接索引
            if expert_idx < len(self.experts):
                expert_layer = self.experts[expert_idx]
            else:
                continue 

            mask_slice = expert_mask[expert_idx] # [Current_K, Tokens]
            top_m_indices, sample_indices = torch.where(mask_slice)
            
            if sample_indices.numel() == 0:
                continue

            current_state = hidden_states_flat[sample_indices]
            current_weights = new_weights[sample_indices, top_m_indices]
            
            # 计算
            current_hidden_states = expert_layer(current_state) * current_weights.unsqueeze(1)
            
            # 累加结果
            final_hidden_states.index_add_(0, sample_indices, current_hidden_states.to(hidden_states.dtype))

        # 5. Shared Experts
        output = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
        if self.original_block.config.n_shared_experts is not None:
            output = output + self.original_block.shared_experts(residuals)
            
        return output


# ==========================================
# 0. Phi 原生辅助类 (必须保留以支持 sparsemixer 逻辑)
# ==========================================
class mp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, multiplier, selected_experts, masked_gates, mask_for_one):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one
        
    @staticmethod
    def backward(ctx, grad_at_output):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors
        grad_at_output = grad_at_output * multiplier
        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(dim=-1, index=selected_experts, src=grad_at_output)
        return grad_at_scores_expaned, None, None, None, None

# ==========================================
# 1. 修改后的 SparseMixer (带干预逻辑)
# ==========================================
def sparsemixer_with_intervention(
    scores, 
    top_k, 
    jitter_eps, 
    training, 
    # 新增参数用于干预
    mode="remove",
    use_top_m=2,
    num_experts=None
):
    assert top_k == 2
    
    # ------------------------------------------------------------------
    # [第一部分] 原生逻辑：计算 Top-1 (包含 Jitter 和 Masking)
    # ------------------------------------------------------------------
    with torch.no_grad():
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts = (
            masked_gates - torch.empty_like(masked_gates).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1)
    else:
        selected_experts = max_ind
        
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
    
    if training:
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75
        ) 
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)
        multiplier = mp.apply(scores, multiplier_o, selected_experts, masked_gates, mask_for_one)
    else:
        multiplier = multiplier_o

    # ------------------------------------------------------------------
    # [第二部分] 原生逻辑：计算 Top-2 (屏蔽 Top-1 后再次 Jitter)
    # ------------------------------------------------------------------
    masked_scores = torch.scatter(scores, -1, selected_experts, float('-inf'))
    with torch.no_grad():
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts_top2 = (
            masked_gates_top2 - torch.empty_like(masked_gates_top2).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1)
    else:
        selected_experts_top2 = max_ind

    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
    
    if training: 
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75
        ) 
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)
        multiplier_top2 = mp.apply(scores, multiplier_top2_o, selected_experts_top2, masked_gates_top2, mask_for_one_top2)
    else:
        multiplier_top2 = multiplier_top2_o
    
    # ------------------------------------------------------------------
    # [第三部分] 拼接结果 (原生)
    # ------------------------------------------------------------------
    # 此时 final_multiplier: [Batch, 2], final_experts: [Batch, 2]
    # Column 0 是 Top-1, Column 1 是 Top-2
    final_multiplier = torch.cat((multiplier, multiplier_top2), dim=-1)
    final_experts = torch.cat((selected_experts, selected_experts_top2), dim=-1)
    
    # ==================================================================
    # 🟢 [关键修改] 在返回前插入干预逻辑 (Intervention)
    # ==================================================================
    
    # 克隆以避免 inplace 操作带来的潜在梯度问题（虽然推理时不求导）
    intervened_multiplier = final_multiplier.clone()
    intervened_experts = final_experts.clone()
    
    batch_size = scores.shape[0]
    device = scores.device

    if mode == "remove":
        # 逻辑：只保留前 use_top_m 个
        # Phi 只有 top_k=2，所以 m 只可能是 1 或 2 (m=0 另行处理)
        if use_top_m < top_k:
            intervened_multiplier = intervened_multiplier[:, :use_top_m]
            intervened_experts = intervened_experts[:, :use_top_m]
            
    elif mode == "replace":
        # 逻辑：仅替换第 k 个 (Target Rank = use_top_m)
        target_rank = use_top_m
        target_idx = target_rank - 1 # 0-based
        
        # 只有当目标索引有效时才替换
        if 0 <= target_idx < top_k:
            # 1. 生成噪声池 [N, Num_Experts]
            noise = torch.rand(batch_size, num_experts, device=device)
            
            # 2. 屏蔽掉当前已选的所有专家 (E1 和 E2 都不能选)
            # 这样保证新选出来的专家既不是 E1 也不是 E2
            noise.scatter_(1, final_experts, -float('inf'))
            
            # 3. 随机选 1 个新专家
            _, random_expert = torch.topk(noise, k=1, dim=-1) # [N, 1]
            
            # 4. 替换目标位置的索引
            intervened_experts[:, target_idx] = random_expert.squeeze(-1)
            
            # 权重保持不变 (继承原位置的权重)
    elif mode == "replace_remaining":
        # 逻辑：保留前 m 个专家，其他专家随机替换
        keep_count = use_top_m
        
        # 只有当需要替换的数量大于0时才执行
        if keep_count < top_k:
            replace_count = top_k - keep_count
            
            # 1. 生成噪声池 [N, Num_Experts]
            noise = torch.rand(batch_size, num_experts, device=device)
            
            # 2. 屏蔽掉要保留的前m个专家
            keep_experts = final_experts[:, :keep_count]
            noise.scatter_(1, keep_experts, -float('inf'))
            
            # 3. 随机选 replace_count 个新专家
            _, random_experts = torch.topk(noise, k=replace_count, dim=-1)
            
            # 4. 替换后面的专家
            intervened_experts[:, keep_count:] = random_experts
            
    elif mode == "replace_with_topp":
        # 逻辑：保留前m个专家，其他专家在topp阈值内随机替换
        keep_count = use_top_m
        
        # 假设topp阈值是0.9，这个值可以根据需要调整
        topp_threshold = 0.9
        
        # 只有当需要替换的数量大于0时才执行
        if keep_count < top_k:
            # 1. 计算router logits的softmax概率
            # 这里我们可以使用原始的scores，因为它包含了所有专家的logits
            all_weights = torch.softmax(scores, dim=-1)
            
            # 2. 对权重进行排序，找到topp阈值内的专家
            sorted_weights, sorted_indices = torch.sort(all_weights, dim=-1, descending=True)
            cumulative_weights = torch.cumsum(sorted_weights, dim=-1)
            topp_mask = cumulative_weights <= topp_threshold
            
            # 至少保留一个专家
            topp_mask[:, 0] = True
            
            # 3. 生成噪声池，但只在topp阈值内的专家中选择
            noise = torch.rand(batch_size, num_experts, device=device)
            noise.masked_fill_(~topp_mask, -float('inf'))
            
            # 4. 屏蔽掉要保留的前m个专家
            keep_experts = final_experts[:, :keep_count]
            noise.scatter_(1, keep_experts, -float('inf'))
            
            # 5. 随机选需要替换数量的新专家
            replace_count = top_k - keep_count
            _, random_experts = torch.topk(noise, k=replace_count, dim=-1)
            
            # 6. 替换后面的专家
            intervened_experts[:, keep_count:] = random_experts

    # 必须 contiguous 否则后续 view 可能报错
    return intervened_multiplier.contiguous(), intervened_experts.contiguous()


# ==========================================
# 4. Phi-3.5-MoE 专用包装器 (调用修改后的 Mixer)
# ==========================================
class PhiExpertSubsetBlockWrapper(BaseExpertSubsetBlockWrapper):
    def __init__(self, original_block, use_top_m, mode="remove"):
        super().__init__(original_block, use_top_m, mode)
        
        self.num_experts = original_block.num_experts 
        self.top_k = original_block.top_k
        self.gate = original_block.gate
        self.experts = original_block.experts
        
        self.router_jitter_noise = getattr(original_block, "router_jitter_noise", 0.0)
        self.input_jitter_noise = getattr(original_block, "input_jitter_noise", 0.0)
        
        self.shared_expert = None
        self._validate_params()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Input Jitter (Eval时不生效，保留结构)
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 
                1.0 + self.input_jitter_noise
            )
            
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        
        # 1. Gate 计算
        router_logits = self.gate(hidden_states_reshaped)

        # 2. 调用带干预的 SparseMixer
        # 🟢 这里我们将 mode 和 use_top_m 传进去
        routing_weights, selected_experts = sparsemixer_with_intervention(
            scores=router_logits,
            top_k=self.top_k,
            jitter_eps=self.router_jitter_noise,
            training=self.training,
            # --- 我们的干预参数 ---
            mode=self.mode,
            use_top_m=self.use_top_m,
            num_experts=self.num_experts
        )
        
        # 转换回输入精度
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 3. 专家计算 (Standard Scatter-Gather, 保持 Phi 原生写法)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            mask_slice = expert_mask[expert_idx]
            if mask_slice.sum() == 0: continue # 优化：跳过空专家

            idx, top_x = torch.where(mask_slice)
            expert_layer = self.experts[expert_idx]

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states_reshaped[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits



# ==========================================
# 4. 应用与更新工具函数
# ==========================================

def apply_expert_subset_to_model(model, use_top_m, mode="remove"):
    """
    智能判断模型类型并应用对应的包装器
    支持: Qwen2-MoE, DeepSeek-V2, Mixtral, Phi-MoE
    """
    config = model.config
    model_type = getattr(config, "model_type", "").lower()
    
    # 1. 识别模型家族
    is_deepseek = "deepseek" in model_type
    is_qwen = "qwen" in model_type
    is_mixtral = "mixtral" in model_type
    is_phi = "phi" in model_type
    
    # 2. 获取原始 Top-K (保持你原有的逻辑)
    original_top_k = None
    if is_deepseek:
        original_top_k = config.num_experts_per_tok
    elif is_qwen:
        original_top_k = getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', None))
    elif is_mixtral:
        original_top_k = config.num_experts_per_tok
    elif is_phi:
        original_top_k = getattr(config, 'num_experts_per_tok', 2) 
    
    # 兜底：从 Layer 属性获取
    if original_top_k is None:
        for layer in model.model.layers:
            modules = [getattr(layer, "mlp", None), getattr(layer, "block_sparse_moe", None)]
            for module in modules:
                if module:
                    if hasattr(module, "top_k"):
                        original_top_k = module.top_k
                        break
                    if hasattr(module, "num_experts_per_tok"):
                        original_top_k = module.num_experts_per_tok
                        break
            if original_top_k: break
                
    print(f"Applying Expert Subset: Model={model_type}, Original TopK={original_top_k}, Target M={use_top_m}, Mode={mode}")

    count = 0
    for layer in model.model.layers:
        target_moe = None
        attr_name = ""
        
        # ==========================================================
        # 🟢 修复点：填补之前的 pass 空缺
        # ==========================================================
        
        # 1. DeepSeek-V2/V3 (通常在 layer.mlp)
        if is_deepseek:
            # 只有当 mlp 具备 experts 属性时才认为是 MoE (V2 Lite 可能混合了 Dense 层)
            if hasattr(layer, "mlp") and (hasattr(layer.mlp, "experts") or hasattr(layer.mlp, "gate")):
                target_moe = layer.mlp
                attr_name = "mlp"

        # 2. Qwen1.5-MoE / Qwen2-MoE (通常在 layer.mlp)
        elif is_qwen:
            # Qwen 的 MoE 也是 layer.mlp
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                target_moe = layer.mlp
                attr_name = "mlp"

        # 3. Phi-3.5-MoE (在 layer.block_sparse_moe)
        elif is_phi: 
            if hasattr(layer, "block_sparse_moe"):
                target_moe = layer.block_sparse_moe
                attr_name = "block_sparse_moe"

        # 4. Mixtral (在 layer.block_sparse_moe)
        elif is_mixtral: 
            if hasattr(layer, "block_sparse_moe"):
                target_moe = layer.block_sparse_moe
                attr_name = "block_sparse_moe"
        
        # ==========================================================
        # 执行包装
        # ==========================================================
        if target_moe:
            # 防止重复包装
            if isinstance(target_moe, BaseExpertSubsetBlockWrapper):
                continue
            
            # 根据模型类型选择 Wrapper
            if is_deepseek:
                new_block = DeepseekExpertSubsetBlockWrapper(target_moe, use_top_m, mode)
            elif is_phi: 
                new_block = PhiExpertSubsetBlockWrapper(target_moe, use_top_m, mode)
            else: 
                # Qwen 和 Mixtral 使用通用的 QwenWrapper (结构相似)
                new_block = QwenExpertSubsetBlockWrapper(target_moe, use_top_m, mode)
                
            setattr(layer, attr_name, new_block)
            count += 1
            
    print(f"Successfully wrapped {count} MoE layers.")
    
    # 🔴 调试信息：如果还是 0，强制报错提示结构不对
    if count == 0:
        print("❌ 警告: 未找到任何 MoE 层！请检查以下层结构:")
        print(f"Layer 0 keys: {model.model.layers[0].__dict__.keys()}")
        
    return model

def update_use_top_m(model, use_top_m, mode=None):
    """
    动态更新参数
    """
    count = 0
    for layer in model.model.layers:
        modules = [getattr(layer, "mlp", None), getattr(layer, "block_sparse_moe", None)]
        for module in modules:
            if isinstance(module, BaseExpertSubsetBlockWrapper):
                module.use_top_m = use_top_m
                if mode is not None:
                    module.mode = mode
                count += 1
                    
    print(f"Updated {count} layers to use_top_m={use_top_m}" + (f", mode={mode}" if mode else ""))
    return model