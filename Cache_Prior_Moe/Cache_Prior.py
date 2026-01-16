import torch
import torch.nn as nn
import torch.nn.functional as F

class CachePriorBlockWrapper(nn.Module):
    def __init__(self, original_block, cache, lambda_val=0.5, top_j=2):
        super().__init__()
        # 1. 引用原始组件
        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob
        
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate
        
        # 2. 绑定 Cache & 参数
        self.cache = cache
        self.lambda_val = lambda_val
        self.top_j = top_j 
        
        # 3. 统计量
        self.register_buffer("avg_range", torch.tensor(0.0))
        self.register_buffer("step_count", torch.tensor(0.0))

        self.register_buffer("persistent_mask", torch.zeros(self.num_experts, dtype=torch.float32))

    # 逐个计算mask和更新缓存专家
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. 计算原始 Logits (并行计算)
        # router_logits: [Total_Tokens, Num_Experts]
        router_logits_raw = self.gate(hidden_states_flat)

        # 2. 更新全局统计量 (CMA - 剔除 Padding 影响)
        with torch.no_grad():
            token_ranges = router_logits_raw.max(dim=-1).values - router_logits_raw.min(dim=-1).values
            # 简单处理：全量平均 (如果需要更严谨可加 mask 过滤)
            current_batch_avg = token_ranges.mean()
            
            self.step_count += 1.0
            if self.step_count == 1:
                self.avg_range.copy_(current_batch_avg)
            else:
                weight_new = 1.0 / self.step_count
                new_avg = self.avg_range * (1 - weight_new) + current_batch_avg * weight_new
                self.avg_range.copy_(new_avg)
                
            current_avg_range_val = self.avg_range.item()
        
        if self.lambda_val == 0:
            # 1. 并行 Top-K (GPU 满载运行)
            # 临时计算 Softmax 用于选 TopK (不影响最终输出，因为没改 logits)
            # 或者直接用 Logits 选 TopK (结果一样)
            _, topk_indices = torch.topk(router_logits_raw, self.top_k, dim=-1)
            
            # 2. 批量更新 Cache (统计 Miss Rate)
            # 我们直接把整个 Batch 的索引传给 update
            # update 内部的纯 Python 循环处理速度极快，且不需要 GPU 同步等待
            with torch.no_grad():
                # 传入 [Total_Tokens, TopK]
                self.cache.update(topk_indices)
            
            # 3. 直接复用原始 Logits
            router_logits_final = router_logits_raw

        else:

            boosted_logits_list = []
            
            # 确保 persistent_mask 在正确的设备上
            if self.persistent_mask.device != router_logits_raw.device:
                self.persistent_mask = self.persistent_mask.to(router_logits_raw.device)

            for t in range(router_logits_raw.size(0)):
                current_logit = router_logits_raw[t] # [Num_Experts]
                
                # --- A. 计算增强 Logits (使用预分配内存优化) ---
                with torch.no_grad():
                    # 1. 清零 (In-place)
                    self.persistent_mask.zero_() 
                    
                    # 2. 填充 Cache Index (In-place)
                    # 假设 cache.cache.keys() 返回的是 python list
                    current_cache_indices = list(self.cache.cache.keys())
                    if current_cache_indices:
                        # 转 tensor 会有一点点开销，但比 malloc 整个 mask 小
                        idx_tensor = torch.tensor(current_cache_indices, device=current_logit.device)
                        self.persistent_mask[idx_tensor] = 1.0

                    # 3. 填充 Top-J (In-place)
                    if self.top_j > 0:
                        _, top_j_indices = torch.topk(current_logit, self.top_j)
                        self.persistent_mask[top_j_indices] = 1.0
                
                # 4. 应用 Bias
                # Logits' = Logits + Lambda * Range * Mask
                bias = self.lambda_val * current_avg_range_val * self.persistent_mask
                current_logit_boosted = current_logit + bias
                
                boosted_logits_list.append(current_logit_boosted)
                
                # --- B. 更新 Cache ---
                with torch.no_grad():
                    _, topk_indices = torch.topk(current_logit_boosted, self.top_k)
                    # 传入 [TopK] 的 1D 张量，update 内部视为单步
                    self.cache.update(topk_indices.view(-1, self.top_k))

            # 重新堆叠
            router_logits_final = torch.stack(boosted_logits_list)
        
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
#     def __init__(self, original_block, cache, lambda_val=0.5, top_j=2):
#         super().__init__()
#         # 1. 引用原始组件
#         self.num_experts = original_block.num_experts
#         self.top_k = original_block.top_k
#         self.norm_topk_prob = original_block.norm_topk_prob
        
#         self.gate = original_block.gate
#         self.experts = original_block.experts
#         self.shared_expert = original_block.shared_expert
#         self.shared_expert_gate = original_block.shared_expert_gate
        
#         # 2. 绑定 Cache & 参数
#         self.cache = cache
#         self.lambda_val = lambda_val
#         self.top_j = top_j 
        
#         # 3. 统计量
#         self.register_buffer("avg_range", torch.tensor(0.0))
#         self.register_buffer("step_count", torch.tensor(0.0))
        
#         # 4. 预分配 Mask 内存 (用于并行操作)
#         # 我们不需要在这里分配巨大的 [Seq, Experts] 矩阵，随用随建即可，
#         # 因为并行模式下 malloc 次数很少 (每层 1 次 vs 1024 次)。

#     def forward(self, hidden_states: torch.Tensor):
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
#         # 1. 计算原始 Logits (并行)
#         # router_logits: [Total_Tokens, Num_Experts]
#         router_logits_raw = self.gate(hidden_states_flat)

#         # 2. 更新全局统计量 (CMA) - 只在 lambda > 0 时需要精确值，但为了统计连续性始终计算
#         with torch.no_grad():
#             token_ranges = router_logits_raw.max(dim=-1).values - router_logits_raw.min(dim=-1).values
#             current_batch_avg = token_ranges.mean()
            
#             self.step_count += 1.0
#             if self.step_count == 1:
#                 self.avg_range.copy_(current_batch_avg)
#             else:
#                 weight_new = 1.0 / self.step_count
#                 new_avg = self.avg_range * (1 - weight_new) + current_batch_avg * weight_new
#                 self.avg_range.copy_(new_avg)
                
#             current_avg_range_val = self.avg_range.item()
        
#         if self.lambda_val > 0:
#             with torch.no_grad():
#                 # A. 获取当前时刻的 Cache Mask (静态)
#                 # shape: [Num_Experts]
#                 # 这是 Batch 开始时的缓存状态
#                 base_cache_mask = self.cache.get_mask() 
#                 if base_cache_mask.device != router_logits_raw.device:
#                     base_cache_mask = base_cache_mask.to(router_logits_raw.device)
                
#                 # B. 构建全序列 Top-J Mask (并行)
#                 # 我们创建一个与 Logits 形状相同的 Mask 矩阵
#                 # shape: [Total_Tokens, Num_Experts]
#                 combined_mask = torch.zeros_like(router_logits_raw, dtype=torch.float32)
                
#                 # 先把 Cache Mask 广播填进去 (所有 Token 共享当前 Cache 状态)
#                 combined_mask += base_cache_mask.unsqueeze(0) 
                
#                 # 再把每个 Token 自己的 Top-J 填进去
#                 if self.top_j > 0:
#                     # 找出所有 Token 的 Top-J 索引
#                     # indices: [Total_Tokens, J]
#                     _, top_j_indices = torch.topk(router_logits_raw, self.top_j, dim=-1)
                    
#                     # 使用 scatter 将这些位置设为 1
#                     # src 全是 1
#                     ones = torch.ones_like(top_j_indices, dtype=torch.float32)
#                     combined_mask.scatter_(1, top_j_indices, ones)
                
#                 # 截断 (因为 += 可能会让重叠部分变成 2，虽然 bias x 2 也没大问题，但最好 clamp)
#                 combined_mask.clamp_(max=1.0)

#             # C. 应用 Bias (全矩阵并行加法)
#             # Logits' = Logits + Lambda * Range * Mask
#             bias = self.lambda_val * current_avg_range_val * combined_mask
#             router_logits_final = router_logits_raw + bias
            
#         else:
#             # Lambda = 0: 直接复用
#             router_logits_final = router_logits_raw
        
#         # 1. 计算权重 & 选 Top-K
#         routing_weights = F.softmax(router_logits_final, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

#         # 2. 更新 Cache (并行/批量更新)
#         # 我们直接把这一批所有 Token 选出的专家扔给 LRU 模拟器
#         # 注意：这里我们假设了一个 Batch 内的 LRU 更新是近似的，或者由 update 内部去模拟串行流
#         # (只要传入的是 [Total_Tokens, K] 的张量，你的 Moe_LRU.py 的 update 函数会自动处理串行统计)
#         with torch.no_grad():
#             self.cache.update(selected_experts)

#         # 3. 归一化 & 类型转换
#         if self.norm_topk_prob:
#             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         # 4. 专家计算 (Expert Computation Loop)
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
#             current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#         # (可选) 逻辑清理
#         # with torch.no_grad():
#         #     self.cache.enforce_limit()

#         # 5. 共享专家
#         shared_output = self.shared_expert(hidden_states_flat)
#         shared_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_output
#         final_hidden_states += shared_output
        
#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
#         return final_hidden_states, router_logits_raw