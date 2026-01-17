import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基类
# ==========================================
class BaseExpertSubsetBlockWrapper(nn.Module):
    def __init__(self, original_block, use_top_m, mode="remove", p_threshold=0.9, replace_count=None):
        super().__init__()
        self.original_block = original_block
        self.use_top_m = use_top_m
        self.mode = mode
        self.p_threshold = p_threshold
        self.replace_count = replace_count # New parameter for controlling replacement count
        
        self.num_experts = getattr(original_block, "num_experts", None)
        if self.num_experts is None:
            config = getattr(original_block, "config", None)
            if config:
                self.num_experts = getattr(config, "n_routed_experts", None)
        
        self.top_k = getattr(original_block, "top_k", None)
        if self.top_k is None:
            config = getattr(original_block, "config", None)
            if config:
                self.top_k = getattr(config, "num_experts_per_tok", None)
        
        self.last_metadata = {
            # "dynamic_m": [],
            "original_ids": [],
            "modified_ids": [],
            "weights": []
        }

    def _process_subset_selection(self, router_logits):
        batch_size, num_experts = router_logits.shape
        top_k = self.top_k
        
        all_probs = F.softmax(router_logits, dim=-1)
        topk_val, topk_indices = torch.topk(router_logits, top_k, dim=-1) 
        
        original_logits_gathered = router_logits.gather(1, topk_indices)
        original_weights_normalized = F.softmax(original_logits_gathered, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(all_probs, descending=True, dim=-1)
        new_indices = topk_indices.clone()
        # dynamic_m_list = []

        # --- 计算 Top-P 截止位 m_idx ---
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # === 模式处理逻辑 ===
        if self.mode == "standard":
            self.last_metadata = {
                # "dynamic_m": [top_k] * batch_size,
                "original_ids": topk_indices.detach().cpu().tolist(),
                "original_weights": original_weights_normalized.detach().cpu().tolist(),
            }
            return original_weights_normalized, topk_indices

        # 统一处理三种 TopP 相关的随机替换模式，以及通用的 replace_with_topp 模式
        if self.mode in ["replace_with_topp", "replace_last_two_with_topp", "replace_last_one_with_topp"]:
            
            for b in range(batch_size):
                # 1. 计算当前样本的 Top-P 边界
                cutoff_mask = cum_probs[b] >= self.p_threshold
                m_idx = (cutoff_mask.nonzero(as_tuple=True)[0][0]).item() + 1 if cutoff_mask.any() else num_experts
                
                # 确定替换的数量 num_to_replace 和起始位置 start_replace_idx
                num_to_replace = 0
                start_replace_idx = top_k 

                # 优先使用显式指定的 replace_count
                if self.replace_count is not None:
                     num_to_replace = self.replace_count
                     start_replace_idx = top_k - num_to_replace
                else:
                    # 兼容旧模式名称
                    if self.mode == "replace_last_two_with_topp":
                        num_to_replace = 2
                        start_replace_idx = top_k - 2
                    elif self.mode == "replace_last_one_with_topp":
                        num_to_replace = 1
                        start_replace_idx = top_k - 1
                    else: # 原始模式：从 use_top_m 开始替换
                        num_to_replace = top_k - self.use_top_m
                        start_replace_idx = self.use_top_m

                # 确保 start_replace_idx 合法
                start_replace_idx = max(0, start_replace_idx)
                
                # 确保 m_idx 至少能覆盖到替换范围，防止索引越界
                m_idx = max(m_idx, start_replace_idx + 1)
                m_idx = min(m_idx, num_experts)
                # dynamic_m_list.append(m_idx)

                # 2. 执行随机替换
                if num_to_replace > 0:
                    # 候选池：从 start_replace_idx 开始到 m_idx 的专家
                    # 这样可以自动避开排在前面的 (0 ~ start_replace_idx-1) 的专家
                    candidate_pool = sorted_indices[b, start_replace_idx:m_idx]
                    
                    if len(candidate_pool) >= num_to_replace:
                        perm = torch.randperm(len(candidate_pool))[:num_to_replace]
                        selected_replacements = candidate_pool[perm]
                        new_indices[b, start_replace_idx:] = selected_replacements
                    else:
                        # 如果候选池太小，则尽量填充
                        take = len(candidate_pool)
                        new_indices[b, start_replace_idx : start_replace_idx + take] = candidate_pool
        
        # === 权重重新计算与归一化 ===
        selected_logits = router_logits.gather(1, new_indices)
        final_weights = F.softmax(selected_logits, dim=-1)

        self.last_metadata = {
            # "dynamic_m": dynamic_m_list,
            "original_ids": topk_indices.detach().cpu().tolist(),
            "original_weights": original_weights_normalized.detach().cpu().tolist(),
            "modified_ids": new_indices.detach().cpu().tolist(),
            "final_weights": final_weights.detach().cpu().tolist()
        }
        
        return final_weights, new_indices

# DeepseekExpertSubsetBlockWrapper 与 apply_expert_subset_to_model 等函数保持不变
# 它们会自动兼容新的 mode 字符串
# ==========================================
# 2. DeepSeek-V2 专用包装器
# ==========================================
class DeepseekExpertSubsetBlockWrapper(BaseExpertSubsetBlockWrapper):
    def __init__(self, original_block, use_top_m, mode="remove", p_threshold=0.9, replace_count=None):
        super().__init__(original_block, use_top_m, mode, p_threshold, replace_count)
        self.config = original_block.config
        self.gate = original_block.gate
        self.experts = original_block.experts
        
    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [Batch, Seq, Hidden]
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # 1. 手动计算 Logits (绕过 gate.forward 的 topk 逻辑)
        # DeepSeek MoEGate 通常是一个 Linear 层
        if hasattr(self.gate, "weight"):
            # 标准情况
            bias = getattr(self.gate, 'bias', None)
            router_logits = F.linear(hidden_states, self.gate.weight, bias)
        elif hasattr(self.gate, "wg"):
            # 某些旧版本
            bias = getattr(self.gate, 'bias', None)
            router_logits = F.linear(hidden_states, self.gate.wg.weight, bias)
        else:
            # 兜底：如果实在找不到权重，只能调用 gate()，但这通常只返回 TopK
            # 这会导致无法做真正的 random replacement
            # 为了防止崩溃，我们尝试调用 forward 但可能报错维度问题，或者逻辑不支持
            # 假设 DeepSeek V2 Lite 结构标准，gate.weight 应该存在
            raise AttributeError(f"Could not find weights in gate module: {type(self.gate)}")

        # 2. 展平 Logits 以便处理: [Batch*Seq, Num_Experts]
        router_logits_flat = router_logits.view(-1, self.num_experts)
        
        # 3. 调用核心逻辑
        final_weights, final_indices = self._process_subset_selection(router_logits_flat)
        
        # 4. 专家计算 (Manual Scatter-Gather)
        # 将输入也展平
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # 生成 Mask [Experts, Top_K, Tokens]
        expert_mask = torch.nn.functional.one_hot(final_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx < len(self.experts):
                expert_layer = self.experts[expert_idx]
            else:
                continue 

            # 找出哪些 token 选中了这个专家
            mask_slice = expert_mask[expert_idx]
            top_m_indices, sample_indices = torch.where(mask_slice)
            
            if sample_indices.numel() == 0: continue

            # 取出对应 token
            current_state = hidden_states_flat[sample_indices]
            # 取出对应权重
            current_weights = final_weights[sample_indices, top_m_indices]
            
            # 计算
            expert_out = expert_layer(current_state)
            # 加权累加
            current_hidden_states = expert_out * current_weights.unsqueeze(1)
            final_hidden_states.index_add_(0, sample_indices, current_hidden_states.to(hidden_states.dtype))

        # 5. Shared Experts (DeepSeek 特有)
        output = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        if self.original_block.config.n_shared_experts is not None:
            output = output + self.original_block.shared_experts(hidden_states)
            
        return output

# ==========================================
# 3. Qwen 专用包装器
# ==========================================
class QwenExpertSubsetBlockWrapper(BaseExpertSubsetBlockWrapper):
    def __init__(self, original_block, use_top_m, mode="remove", p_threshold=0.9, replace_count=None):
        super().__init__(original_block, use_top_m, mode, p_threshold, replace_count)
        # self.config = original_block.config # Qwen3 block might not have config
        self.config = getattr(original_block, "config", None)
        
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = getattr(original_block, "shared_expert", None)
        self.shared_expert_gate = getattr(original_block, "shared_expert_gate", None)
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. Compute Logits
        router_logits = self.gate(hidden_states_flat)
        
        # 2. Process Subset Selection
        final_weights, final_indices = self._process_subset_selection(router_logits)
        
        # 3. Expert Computation
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        expert_mask = torch.nn.functional.one_hot(final_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx < len(self.experts):
                expert_layer = self.experts[expert_idx]
            else:
                continue 

            mask_slice = expert_mask[expert_idx]
            top_m_indices, sample_indices = torch.where(mask_slice)
            
            if sample_indices.numel() == 0: continue

            current_state = hidden_states_flat[sample_indices]
            current_weights = final_weights[sample_indices, top_m_indices]
            
            expert_out = expert_layer(current_state)
            current_hidden_states = expert_out * current_weights.unsqueeze(1)
            final_hidden_states.index_add_(0, sample_indices, current_hidden_states.to(hidden_states.dtype))

        # 4. Shared Expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states_flat)
            if self.shared_expert_gate is not None:
                shared_out = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_out
            final_hidden_states += shared_out
            
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

# ==========================================
# 4. 工具函数
# ==========================================
def apply_expert_subset_to_model(model, use_top_m, mode="replace_with_topp", p_threshold=0.9, replace_count=None):
    count = 0
    
    # 简单的模型类型判断
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "").lower()
    is_qwen = "qwen" in model_type
    is_deepseek = "deepseek" in model_type

    for layer in model.model.layers:
        target_moe = None
        
        # DeepSeek
        if hasattr(layer, "mlp") and (hasattr(layer.mlp, "experts") or hasattr(layer.mlp, "gate")):
            target_moe = layer.mlp
        
        # Qwen
        elif is_qwen and hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            target_moe = layer.mlp
            
        if target_moe:
            if not isinstance(target_moe, BaseExpertSubsetBlockWrapper):
                if is_qwen:
                    new_block = QwenExpertSubsetBlockWrapper(target_moe, use_top_m, mode, p_threshold, replace_count)
                else:
                    # Default to Deepseek for now or existing logic
                    new_block = DeepseekExpertSubsetBlockWrapper(target_moe, use_top_m, mode, p_threshold, replace_count)
                layer.mlp = new_block
                count += 1
            else:
                # 更新参数
                target_moe.use_top_m = use_top_m
                target_moe.mode = mode
                target_moe.p_threshold = p_threshold
                target_moe.replace_count = replace_count
                
    print(f"Applied wrapper to {count} MoE layers. Mode={mode}, TopM={use_top_m}, P={p_threshold}, ReplaceCount={replace_count}")
    return model

def collect_moe_metadata(model):
    all_meta = []
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and isinstance(layer.mlp, BaseExpertSubsetBlockWrapper):
            all_meta.append(layer.mlp.last_metadata)
    return all_meta