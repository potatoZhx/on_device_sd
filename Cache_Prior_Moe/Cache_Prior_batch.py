import torch
import torch.nn as nn
import torch.nn.functional as F

class CachePriorBlockWrapper(nn.Module):
    """
    Wrapper for Qwen2MoeSparseMoeBlock.
    """
    # [现有] 全局静态变量，用于传递 Batch Mask
    CURRENT_MASK = None 
    
    # 🟢 [新增] 全局静态变量：是否仅在 Decode 阶段应用 Cache 逻辑
    # 如果设为 True，当 seq_len > 1 (Prefill) 时，将跳过 Bias 和 Cache Update
    ONLY_CACHE_ON_DECODE = False 

    def __init__(self, original_block, cache, lambda_val=0.5, top_j=2):
        super().__init__()
        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob
        
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate
        
        self.cache = cache
        self.lambda_val = lambda_val
        self.top_j = top_j 
        
        self.register_buffer("avg_range", torch.tensor(0.0))
        self.register_buffer("step_count", torch.tensor(0.0))
        self.register_buffer("persistent_mask", torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. 计算原始 Logits
        router_logits_raw = self.gate(hidden_states_flat)

        # 🟢 [新增] 判断是否处于 Prefill 阶段
        # 如果开启了"仅Decode模式" 且 当前序列长度 > 1，则视为 Prefill
        is_prefill_stage = self.ONLY_CACHE_ON_DECODE and (sequence_length > 1)

        # 如果是 Prefill 阶段，强制 lambda=0 (不加 Bias)，并且后续不更新 Cache
        # 这相当于临时禁用了 Cache-Prior 逻辑
        effective_lambda = 0.0 if is_prefill_stage else self.lambda_val

        # 2. 更新 CMA (Prefill 阶段通常也跳过统计，或者你可以选择统计)
        # 这里我们选择：Prefill 阶段不更新统计量，以免长 Prompt 稀释生成时的统计特征
        if not is_prefill_stage:
            with torch.no_grad():
                token_ranges = router_logits_raw.max(dim=-1).values - router_logits_raw.min(dim=-1).values
                valid_mask = None
                if CachePriorBlockWrapper.CURRENT_MASK is not None:
                    if CachePriorBlockWrapper.CURRENT_MASK.size(0) == router_logits_raw.size(0):
                        valid_mask = CachePriorBlockWrapper.CURRENT_MASK.bool()
                
                if valid_mask is not None:
                    valid_ranges = token_ranges[valid_mask]
                    current_batch_avg = valid_ranges.mean() if valid_ranges.numel() > 0 else self.avg_range
                else:
                    current_batch_avg = token_ranges.mean()

                if valid_mask is None or valid_ranges.numel() > 0:
                    self.step_count += 1.0
                    if self.step_count == 1:
                        self.avg_range.copy_(current_batch_avg)
                    else:
                        weight_new = 1.0 / self.step_count
                        new_avg = self.avg_range * (1 - weight_new) + current_batch_avg * weight_new
                        self.avg_range.copy_(new_avg)
                
                current_avg_range_val = self.avg_range.item()
        else:
            current_avg_range_val = 0.0 # Prefill 时不用

        # ==============================================================================
        # 分支逻辑
        # ==============================================================================
        
        if effective_lambda == 0:
            # [极速通道] Lambda=0 OR Prefill Stage
            
            # 1. Top-K
            routing_weights_temp = F.softmax(router_logits_raw, dim=1, dtype=torch.float)
            _, topk_indices = torch.topk(routing_weights_temp, self.top_k, dim=-1)
            
            # 2. Cache Update
            # 🟢 [关键修改] 如果是 Prefill 阶段，跳过 Cache Update
            if not is_prefill_stage:
                with torch.no_grad():
                    indices_to_update = topk_indices 
                    if valid_mask is not None:
                        indices_to_update = topk_indices[valid_mask]
                    if indices_to_update.size(0) > 0:
                        self.cache.update(indices_to_update)
            
            router_logits_final = router_logits_raw

        else:
            # [顺序通道] Decode Stage & Lambda > 0
            # ... (保留原有的串行逻辑) ...
            boosted_logits_list = []
            if self.persistent_mask.device != router_logits_raw.device:
                self.persistent_mask = self.persistent_mask.to(router_logits_raw.device)

            for t in range(router_logits_raw.size(0)):
                current_logit = router_logits_raw[t]
                
                # Mask 检查
                is_valid_token = True
                if CachePriorBlockWrapper.CURRENT_MASK is not None:
                    is_valid_token = CachePriorBlockWrapper.CURRENT_MASK[t].item()

                # 计算增强
                with torch.no_grad():
                    self.persistent_mask.zero_() 
                    current_cache_indices = list(self.cache.cache.keys())
                    if current_cache_indices:
                        idx_tensor = torch.tensor(current_cache_indices, device=current_logit.device)
                        self.persistent_mask[idx_tensor] = 1.0
                    if self.top_j > 0:
                        _, top_j_indices = torch.topk(current_logit, self.top_j)
                        self.persistent_mask[top_j_indices] = 1.0
                
                bias = self.lambda_val * current_avg_range_val * self.persistent_mask
                current_logit_boosted = current_logit + bias
                boosted_logits_list.append(current_logit_boosted)
                
                # 更新 Cache (仅有效 Token)
                if is_valid_token:
                    with torch.no_grad():
                        _, topk_indices = torch.topk(current_logit_boosted, self.top_k)
                        self.cache.update(topk_indices.view(-1, self.top_k))

            router_logits_final = torch.stack(boosted_logits_list)

        # ... (后续 MoE 计算保持不变) ...
        
        routing_weights = F.softmax(router_logits_final, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_output = self.shared_expert(hidden_states_flat)
        shared_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_output
        final_hidden_states += shared_output
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits_raw
    



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CachePriorBlockWrapper(nn.Module):
#     """
#     Wrapper for Qwen2MoeSparseMoeBlock.
#     Implements Cache-Prior routing with LRU simulation.
#     Supports Batch Parallelism and Padding Masking.
#     """
#     # Global static variable for passing batch mask (1=Valid, 0=Padding)
#     CURRENT_MASK = None 

#     def __init__(self, original_block, cache, lambda_val=0.5, top_j=2):
#         super().__init__()
#         # 1. Reference original components
#         self.num_experts = original_block.num_experts
#         self.top_k = original_block.top_k
#         self.norm_topk_prob = original_block.norm_topk_prob
        
#         self.gate = original_block.gate
#         self.experts = original_block.experts
#         self.shared_expert = original_block.shared_expert
#         self.shared_expert_gate = original_block.shared_expert_gate
        
#         # 2. Bind Cache & Params
#         self.cache = cache
#         self.lambda_val = lambda_val
#         self.top_j = top_j 
        
#         # 3. Stats
#         self.register_buffer("avg_range", torch.tensor(0.0))
#         self.register_buffer("step_count", torch.tensor(0.0))

#     def forward(self, hidden_states: torch.Tensor):
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
#         # 1. Compute Raw Logits (Parallel)
#         # router_logits: [Total_Tokens, Num_Experts]
#         router_logits_raw = self.gate(hidden_states_flat)

#         # 2. Update Global Statistics (CMA) - Filter Padding
#         with torch.no_grad():
#             token_ranges = router_logits_raw.max(dim=-1).values - router_logits_raw.min(dim=-1).values
            
#             # Check for global mask
#             valid_mask = None
#             if CachePriorBlockWrapper.CURRENT_MASK is not None:
#                 if CachePriorBlockWrapper.CURRENT_MASK.size(0) == router_logits_raw.size(0):
#                     valid_mask = CachePriorBlockWrapper.CURRENT_MASK.bool()
            
#             # Calculate mean range (only for valid tokens)
#             if valid_mask is not None:
#                 valid_ranges = token_ranges[valid_mask]
#                 if valid_ranges.numel() > 0:
#                     current_batch_avg = valid_ranges.mean()
#                 else:
#                     current_batch_avg = self.avg_range 
#             else:
#                 current_batch_avg = token_ranges.mean()

#             # Update CMA
#             if valid_mask is None or valid_ranges.numel() > 0:
#                 self.step_count += 1.0
#                 if self.step_count == 1:
#                     self.avg_range.copy_(current_batch_avg)
#                 else:
#                     weight_new = 1.0 / self.step_count
#                     new_avg = self.avg_range * (1 - weight_new) + current_batch_avg * weight_new
#                     self.avg_range.copy_(new_avg)
            
#             current_avg_range_val = self.avg_range.item()

#         # ==============================================================================
#         # 🚀 Branch Logic: Fast Path vs Bias Path
#         # ==============================================================================
        
#         if self.lambda_val == 0:
#             # [Fast Path] Lambda=0
#             # Parallel Top-K, Async Cache Update (Filtered)
            
#             # 1. Top-K
#             routing_weights_temp = F.softmax(router_logits_raw, dim=1, dtype=torch.float)
#             _, topk_indices = torch.topk(routing_weights_temp, self.top_k, dim=-1)
            
#             # 2. Batch Update Cache (Filter Padding)
#             with torch.no_grad():
#                 indices_to_update = topk_indices 
#                 if valid_mask is not None:
#                     indices_to_update = topk_indices[valid_mask]
                
#                 if indices_to_update.size(0) > 0:
#                     self.cache.update(indices_to_update)
            
#             router_logits_final = router_logits_raw

#         else:
#             # [Parallel Bias Path] Lambda > 0
#             # Use speculative parallelism (batch approximation) for speed
            
#             with torch.no_grad():
#                 # A. Get Base Cache Mask (Static for this batch)
#                 base_cache_mask = self.cache.get_mask() 
#                 if base_cache_mask.device != router_logits_raw.device:
#                     base_cache_mask = base_cache_mask.to(router_logits_raw.device)
                
#                 # B. Build Combined Mask (Parallel)
#                 # shape: [Total_Tokens, Num_Experts]
#                 combined_mask = torch.zeros_like(router_logits_raw, dtype=torch.float32)
                
#                 # Broadcast Base Mask
#                 combined_mask += base_cache_mask.unsqueeze(0) 
                
#                 # Add Top-J Mask
#                 if self.top_j > 0:
#                     _, top_j_indices = torch.topk(router_logits_raw, self.top_j, dim=-1)
#                     ones = torch.ones_like(top_j_indices, dtype=torch.float32)
#                     combined_mask.scatter_(1, top_j_indices, ones)
                
#                 combined_mask.clamp_(max=1.0)

#             # C. Apply Bias (Parallel Addition)
#             # Logits' = Logits + Lambda * Range * Mask
#             bias = self.lambda_val * current_avg_range_val * combined_mask
#             router_logits_final = router_logits_raw + bias
            
#             # D. Update Cache (Filter Padding)
#             with torch.no_grad():
#                 # Re-calculate Top-K on boosted logits
#                 routing_weights_temp = F.softmax(router_logits_final, dim=1, dtype=torch.float)
#                 _, topk_indices = torch.topk(routing_weights_temp, self.top_k, dim=-1)
                
#                 indices_to_update = topk_indices
#                 if valid_mask is not None:
#                     indices_to_update = topk_indices[valid_mask]
                
#                 if indices_to_update.size(0) > 0:
#                     self.cache.update(indices_to_update)

#         # ==============================================================================
#         # Downstream Computation (Standard MoE)
#         # ==============================================================================
        
#         routing_weights = F.softmax(router_logits_final, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

#         if self.norm_topk_prob:
#             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
#         expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

#         for expert_idx in expert_hit:
#             expert_idx = expert_idx[0]
#             expert_layer = self.experts[expert_idx]
#             idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

#             current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            
#             # Expert computation is always full-fledged (includes padding) for speed
#             current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#         shared_output = self.shared_expert(hidden_states_flat)
#         shared_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_output
#         final_hidden_states += shared_output
        
#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
#         return final_hidden_states, router_logits_raw