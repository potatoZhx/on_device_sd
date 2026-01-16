from collections import OrderedDict
import torch

class ExpertCache:
    def __init__(self, experts_list, cache_size, layer_id, device="cuda"):
        self.experts_list = experts_list
        self.num_experts = len(experts_list)
        self.cache_size = cache_size
        self.device = device
        self.layer_id = layer_id  # 方便日志显示是哪一层的
        
        self.cache = OrderedDict() 
        self.hits = 0
        self.total_queries = 0
        self.last_events = [] # 记录 ["Load #5", "Evict #2", "Hit #5"]
        
        self._initialize_offload()

    def _initialize_offload(self):
        cpu_offload_count = max(0, self.num_experts - self.cache_size)
        temp_cpu_count = cpu_offload_count
        for i, expert in enumerate(self.experts_list):
            if temp_cpu_count > 0:
                # expert.to("cpu")
                temp_cpu_count -= 1
            else:
                expert.to(self.device)
                self.cache[i] = True 

    def get_mask(self):
        mask = torch.zeros(self.num_experts, device=self.device)
        if self.cache:
            indices = torch.tensor(list(self.cache.keys()), device=self.device)
            mask[indices] = 1.0
        return mask
    
    def update(self, selected_experts):
        """
        [通用更新接口] 支持单行或多行输入，按 Token 组进行 LRU 更新。
        
        Args:
            selected_experts: 
                - 2D Tensor/List: [Batch*Seq, TopK] -> 处理多个 Token
                - 1D Tensor/List: [TopK] -> 处理单个 Token
        """
        self.last_events = [] 
        
        # --- 1. 输入标准化：统一转为 2D 列表 [[idx...], [idx...]] ---
        request_stream = []
        
        if isinstance(selected_experts, torch.Tensor):
            if selected_experts.dim() == 2:
                # 多行输入 [N, K] -> 保持原样
                request_stream = selected_experts.tolist()
            elif selected_experts.dim() == 1:
                # 单行输入 [K] -> 包装成 [[K]]
                request_stream = [selected_experts.tolist()]
            else:
                # 兜底：尝试展平并包装，或者报错
                print(f"Warning: Unexpected tensor shape {selected_experts.shape}")
                request_stream = [selected_experts.view(-1).tolist()]
                
        elif isinstance(selected_experts, list):
            if len(selected_experts) > 0 and isinstance(selected_experts[0], list):
                # 已经是 2D 列表
                request_stream = selected_experts
            else:
                # 1D 列表 -> 包装
                request_stream = [selected_experts]

        step_hits = 0
        step_total = 0
        
        # --- 2. 逐 Token (Group) 处理 ---
        for token_experts in request_stream:
            # token_experts: [Exp1, Exp2, Exp3, Exp4] (当前 Token 需要的专家)
            
            # A. 统计命中 (Statistics)
            # 不去重统计，以反映真实计算负载的命中率
            for expert_idx in token_experts:
                if expert_idx in self.cache:
                    step_hits += 1
            
            step_total += len(token_experts)
            
            # B1. 保护现有专家 (Mark as Recently Used)
            # 先把这一组里已经在缓存的，全部移到 LRU 队列尾部(最新)
            # 这样在下面的淘汰步骤中，它们绝对不会被踢走
            for expert_idx in token_experts:
                if expert_idx in self.cache:
                    self.cache.move_to_end(expert_idx)
            
            # B2. 加载缺失专家 (Load & Evict)
            for expert_idx in token_experts:
                if expert_idx not in self.cache:
                    # 检查容量
                    if len(self.cache) >= self.cache_size:
                        # 弹出头部 (最久未使用)
                        # 因为上面的循环已经把当前需要的移到了尾部，所以这里弹出的肯定是不需要的
                        evicted_idx, _ = self.cache.popitem(last=False)
                        # self.last_events.append(f"Evict {evicted_idx}")
                    
                    # 加载
                    self.cache[expert_idx] = True
                    # self.last_events.append(f"Load {expert_idx}")

        # --- 3. 更新全局统计 ---
        self.hits += step_hits
        self.total_queries += step_total
        
        if step_total > 0:
            self.step_hit_rate = step_hits / step_total
        else:
            self.step_hit_rate = 1.0

    # def update(self, selected_experts):
    #     '''
    #     先计算命中率再更新缓存状态,只支持单token专家选择输入
    #     '''
    #     self.last_events = [] 

    #     if isinstance(selected_experts, torch.Tensor):
    #         all_indices_list = selected_experts.view(-1).tolist()
    #     else:
    #         # 假设输入已经是 list 或 numpy array
    #         import numpy as np
    #         all_indices_list = np.array(selected_experts).flatten().tolist()
            
    #     step_total_queries = len(all_indices_list)
    #     step_hits = 0
        
    #     # 仅统计命中情况 (不修改 Cache 状态，避免影响后续的物理逻辑判断)
    #     for idx in all_indices_list:
    #         if idx in self.cache:
    #             step_hits += 1
                
    #     # 计算当前步的瞬时命中率
    #     if step_total_queries > 0:
    #         self.step_hit_rate = step_hits / step_total_queries
    #     else:
    #         self.step_hit_rate = 1.0
            
    #     # 更新全局累计统计
    #     self.hits += step_hits
    #     self.total_queries += step_total_queries

    #     current_needed_indices = list(set(all_indices_list))
    #     if len(current_needed_indices) > self.cache_size:
    #         print(f"警告: 选择的专家数量 {len(current_needed_indices)} 超过缓存限制 {self.cache_size}！请检查模型配置。")
        
    #     for expert_idx in current_needed_indices:
    #         if expert_idx in self.cache:
    #             self.cache.move_to_end(expert_idx)
    #         else:
    #             # self.experts_list[expert_idx].to(self.device)
    #             self.cache[expert_idx] = True
    #             self.last_events.append(f"Load {expert_idx}")
        
    #     # 2. 淘汰
    #     while len(self.cache) > self.cache_size:
    #         candidate_idx = next(iter(self.cache))
    #         # if candidate_idx in current_needed_indices:
    #         #     break 
            
    #         self.cache.popitem(last=False)
    #         # self.experts_list[candidate_idx].to("cpu")
    #         self.last_events.append(f"Evict {candidate_idx}")
    
    def enforce_limit(self):
        """
       强制清理：在计算完成后，将缓存大小压缩回 cache_size。
        此时不需要保护 '当前需要的专家'，严格按照 LRU 淘汰。
        """        
        while len(self.cache) > self.cache_size:
            # 1. 严格取出最久未使用的 (FIFO in OrderedDict)
            # last=False 表示弹出第一个插入的元素（最旧的）
            candidate_idx, _ = self.cache.popitem(last=False)
            
            # 2. 物理卸载
            # self.experts_list[candidate_idx].to("cpu")
            self.last_events.append(f"Post-Evict {candidate_idx}")

    def get_miss_rate(self):
        if self.total_queries == 0: return 0.0
        return 1.0 - (self.hits / self.total_queries)
    
    def reset_stats(self):
        self.hits = 0
        self.total_queries = 0
        self.last_events = []
        
        
