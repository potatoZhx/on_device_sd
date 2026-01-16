import torch
import time
from typing import Dict, List, Tuple
from .spec_sampling import speculative_sampling

class MOESpecDecoder:
    def __init__(self, original_model, modified_model, draft_length: int = 2):
        self.original_model = original_model
        self.modified_model = modified_model
        self.draft_length = draft_length
        
        # 统计变量
        self.total_draft_length = 0
        self.total_accept_length = 0
        self.accept_length_list = []
        
    def speculate_decode(self, input_ids: torch.Tensor, max_new_tokens: int = 1024, deterministic: bool = False) -> Dict:
        """
        MOE推测解码主循环
        返回格式兼容Spec-Bench的eval.py
        """
        # 初始化统计变量
        self.total_draft_length = 0
        self.total_accept_length = 0
        self.accept_length_list = []
        
        current_input = input_ids.clone()
        generated_tokens = []  # 仅统计新生成tokens
        
        # Prefill阶段（始终使用原始模型，不使用修改后的路由）
        try:
            prefill_logits, kv_cache = self.original_model.prefill(input_ids)
            # 从prefill的logits中采样第一个token作为last_token
            last_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).item()
            
            # 将first_token加入generated_tokens（作为输出的一部分）
            generated_tokens.append(last_token)
            current_input = torch.cat([
                current_input,
                torch.tensor([[last_token]], device=input_ids.device)
            ], dim=1)
            # 注意：不把last_token加入KV cache，因为在verify阶段会输入[last_token] + draft_tokens
        except AttributeError as e:
            if "'tuple' object has no attribute 'get_seq_length'" in str(e):
                print("\n警告: 在prefill阶段遇到Qwen3模型的KV cache问题，尝试不使用KV cache")
                kv_cache = None
            else:
                raise e
        
        step_count = 0
        
        # 推测解码主循环
        while len(generated_tokens) < max_new_tokens:
            step_start_time = time.time()
            
            # 3.1 修改模型生成draft（传入last_token和verify的kv_cache）
            try:
                draft_tokens, draft_logits = self._generate_draft(
                    last_token,
                    kv_cache
                )
            except Exception as e:
                print(f"\n生成draft时出错: {str(e)}")
                break
            
            if len(draft_tokens) == 0:
                break
                
            # 累加draft长度
            self.total_draft_length += len(draft_tokens)
            
            # 3.2 原模型验证draft（传入last_token和draft_tokens）
            try:
                accepted_tokens, accepted_length, kv_cache = self._verify_draft(
                    last_token,
                    draft_tokens,
                    draft_logits,
                    kv_cache,
                    deterministic=deterministic
                )
            except Exception as e:
                print(f"\n验证draft时出错: {str(e)}")
                break
            
            # 累加接受长度（accepted_length是被接受的draft token数，用于统计接受率）
            self.total_accept_length += accepted_length
            self.accept_length_list.append(accepted_length)
            
            # 若超过max_new_tokens，进行截断
            remaining = max_new_tokens - len(generated_tokens)
            if len(accepted_tokens) > remaining:
                accepted_tokens = accepted_tokens[:remaining]
                # 截断时，将accepted_length也裁剪到不超过已追加的数量
                accepted_length = min(accepted_length, len(accepted_tokens))
                # 同步裁剪KV cache（_verify_draft内部会基于accepted_tokens长度裁剪，这里无需再次裁剪）

            # 更新状态（accepted_tokens包含所有新生成的tokens：接受的draft + 新采样的token）
            # 注意：即使accepted_length=0（draft被拒绝），accepted_tokens也至少包含1个新采样的token
            generated_tokens.extend(accepted_tokens)
            current_input = torch.cat([
                current_input, 
                torch.tensor([accepted_tokens], device=current_input.device)
            ], dim=1)
            
            # 更新last_token为本轮输出的最后一个token（用于下一轮draft）
            last_token = accepted_tokens[-1]
            # KV cache已经在_verify_draft中正确更新，这里不需要额外处理
            
            step_count += 1
            step_time = time.time() - step_start_time
            
            # 检查终止条件
            if self.original_model.tokenizer.eos_token_id in accepted_tokens:
                break
        
        # 构建最终输出序列
        if generated_tokens:
            final_output = torch.cat([input_ids, torch.tensor([generated_tokens], device=input_ids.device)], dim=1)
        else:
            final_output = input_ids
        
        return {
            'output_ids': final_output,
            'new_token': len(generated_tokens),
            'step': step_count,
            'accept_length_list': self.accept_length_list,
            'total_draft_length': self.total_draft_length,
            'total_accept_length': self.total_accept_length,
            'acceptance_rate': self.total_accept_length / self.total_draft_length if self.total_draft_length > 0 else 0.0
        }
    
    def _generate_draft(self, last_token: int, verify_kv_cache: Dict) -> Tuple[List[int], torch.Tensor]:
        """
        使用Draft模型生成draft_length个tokens（当前为2）
        
        KV Cache管理策略：
        - 第1次decode: 使用verify_kv_cache
        - 第2次decode: 使用verify_kv_cache + draft生成的kv_cache拼接
        - 最后丢弃draft生成的所有kv_cache，只保留verify_kv_cache
        
        参数：
            last_token: 上一轮输出的最后一个token（int）
            verify_kv_cache: Verify模型的KV cache
            
        返回：
            draft_tokens: 生成的draft tokens列表，长度为self.draft_length
            draft_logits: 每个draft token的logits，shape [1, draft_length, vocab_size]
        """
        try:
            # 临时启用路由修改
            self.modified_model.routing_modifier.enable_routing_modification(self.modified_model.model)
            
            draft_tokens = []
            all_draft_logits = []
            
            # 重要：创建verify_kv_cache的深拷贝，避免修改原始cache
            import copy
            current_kv_cache = copy.deepcopy(verify_kv_cache)
            
            current_token = last_token
            
            with torch.no_grad():
                for i in range(self.draft_length):
                    # 准备输入token
                    input_token = torch.tensor([[current_token]], device=self.original_model.device)
                    
                    # Draft模型前向传播
                    outputs = self.modified_model.model(
                        input_token,
                        past_key_values=current_kv_cache,
                        use_cache=True
                    )
                    
                    # 获取logits
                    logits = outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
                    all_draft_logits.append(logits)
                    
                    # 贪婪采样下一个token
                    next_token = torch.argmax(logits[:, 0, :], dim=-1).item()
                    draft_tokens.append(next_token)
                    
                    # 更新状态（用于下一次draft循环）
                    current_token = next_token
                    # 更新KV cache（累积draft生成的token）
                    # 注意：这个kv_cache包含 verify部分 + draft部分
                    current_kv_cache = outputs.past_key_values
            
            # 合并所有logits
            draft_logits = torch.cat(all_draft_logits, dim=1)  # [1, draft_length, vocab_size]
            
            # 返回时丢弃draft生成的kv_cache，只保留原始的verify_kv_cache
            # （实际上我们不返回kv_cache，verify阶段会自己生成）
            
            return draft_tokens, draft_logits
            
        finally:
            # 确保禁用路由修改
            self.modified_model.routing_modifier.disable_routing_modification(self.modified_model.model)
    
    def _verify_draft(self, last_token: int, draft_tokens: List[int], 
                     draft_logits: torch.Tensor, verify_kv_cache: Dict, deterministic: bool = False) -> Tuple[List[int], int, Dict]:
        """
        使用Verify模型验证draft tokens，应用推测采样算法
        
        新的KV Cache策略：
        - 输入tokens: [last_token] + draft_tokens（例如：[e, f, g]）
        - 使用verify_kv_cache进行前向传播
        - 获取所有位置的logits用于验证和采样
        
        参数：
            last_token: 上一轮输出的最后一个token（int）
            draft_tokens: Draft模型生成的tokens列表，长度为draft_length
            draft_logits: Draft模型的logits，shape [1, draft_length, vocab_size]
            verify_kv_cache: Verify模型的KV cache
            
        返回：
            accepted_tokens: 实际生成的所有tokens（包括接受的draft + 新采样的）
            n_matches: 被接受的draft token数量（0到draft_length）
            new_kv_cache: 更新后的Verify KV cache
        """
        device = self.original_model.device
        
        # 构建verify输入序列：[last_token] + draft_tokens
        # 例如：[e] + [f, g] = [e, f, g]
        verify_input_tokens = [last_token] + draft_tokens
        verify_input_ids = torch.tensor([verify_input_tokens], device=device)
        
        # 构建候选序列（用于speculative_sampling）
        # 注意：candidate_input_ids应该只包含draft_tokens，不包含last_token
        draft_tokens_tensor = torch.tensor([draft_tokens], device=device)
        candidate_input_ids = draft_tokens_tensor
        candidate_length = len(draft_tokens)
        
        # 使用Verify模型进行前向传播
        with torch.no_grad():
            outputs = self.original_model.model(
                verify_input_ids,
                past_key_values=verify_kv_cache,
                use_cache=True
            )
            # outputs.logits shape: [1, len(verify_input_tokens), vocab_size]
            # 包含所有位置的logits
            verify_logits = outputs.logits
            updated_kv_cache = outputs.past_key_values
        
        # 提取用于验证和采样的logits
        # verify_logits[:, 0, :] 对应last_token位置，用于验证draft_tokens[0]
        # verify_logits[:, 1, :] 对应draft_tokens[0]位置，用于验证draft_tokens[1]
        # verify_logits[:, 2, :] 对应draft_tokens[1]位置，用于采样新token
        # 我们需要前(draft_length+1)个位置的logits
        new_logits = verify_logits[:, :candidate_length+1, :]  # [1, draft_length+1, vocab_size]
        
        # 使用推测采样算法决定接受/拒绝
        # 比较draft_logits和new_logits的前draft_length个位置
        valid_tokens, n_matches = speculative_sampling(
            candidate_input_ids=candidate_input_ids,
            candidate_logits=draft_logits,
            candidate_length=candidate_length,
            new_logits=new_logits,
            last_assistant_token_is_eos=(draft_tokens[-1] == self.original_model.tokenizer.eos_token_id),
            max_matches=candidate_length,
            deterministic=deterministic
        )
        
        # 提取最终接受的tokens
        # valid_tokens的格式：
        # - 如果n_matches > 0: [draft_tokens[:n_matches]] + [新采样的token]
        # - 如果n_matches = 0: [新采样的token]
        # 但我们需要排除last_token（因为它在下一步不需要了）
        accepted_tokens = valid_tokens[0].tolist()
        
        # 注意：valid_tokens可能包含last_token，我们需要检查
        # 根据speculative_sampling的实现，valid_tokens应该只包含新生成的部分
        # 但为了安全起见，我们确保只取新生成的tokens
        # accepted_tokens已经是我们需要的新tokens
        
        # 更新KV cache：需要裁剪到实际接受的长度
        # 如果n_matches < draft_length，说明有些draft token被拒绝了
        # 我们需要裁剪KV cache
        actual_new_tokens = len(accepted_tokens)
        
        # updated_kv_cache现在包含：原verify_kv + [last_token] + draft_tokens的信息
        # 我们需要裁剪到：原verify_kv + [last_token] + accepted_tokens
        # 计算最终的cache大小
        original_cache_size = self._get_kv_cache_length(verify_kv_cache)
        final_cache_size = original_cache_size + 1 + actual_new_tokens  # +1是last_token
        
        # 裁剪KV cache
        if self._get_kv_cache_length(updated_kv_cache) > final_cache_size:
            final_kv_cache = self._crop_kv_cache(updated_kv_cache, final_cache_size)
        else:
            final_kv_cache = updated_kv_cache
        
        # 返回：
        # - accepted_tokens: 实际生成的所有tokens（可能包括部分draft + 新采样的）
        # - n_matches: 被接受的draft token数量（用于统计接受率）
        # - final_kv_cache: 更新后的KV cache
        return accepted_tokens, n_matches, final_kv_cache
    
    def _get_kv_cache_length(self, kv_cache: Dict) -> int:
        """获取KV cache的长度（包含多少个tokens）"""
        if kv_cache is None:
            return 0
        
        from transformers.cache_utils import DynamicCache
        if isinstance(kv_cache, DynamicCache):
            # DynamicCache有get_seq_length方法
            return kv_cache.get_seq_length()
        else:
            # 对于元组类型的cache，获取第一层的key的长度
            if len(kv_cache) > 0 and kv_cache[0] is not None:
                return kv_cache[0][0].shape[2]  # shape: [batch, num_heads, seq_len, head_dim]
            return 0
    
    def _crop_kv_cache(self, kv_cache: Dict, new_cache_size: int) -> Dict:
        """
        裁剪KV缓存到指定长度
        基于Spec-Bench的_crop_past_key_values实现
        """
        if kv_cache is None:
            return None
        
        # 对于DynamicCache类型的特殊处理
        from transformers.cache_utils import DynamicCache
        if isinstance(kv_cache, DynamicCache):
            # 使用DynamicCache自带的crop方法
            return kv_cache.crop(new_cache_size)
        
        # 对于传统的元组类型缓存
        cropped_cache = []
        for layer_cache in kv_cache:
            if layer_cache is None:
                cropped_cache.append(None)
            else:
                # 裁剪到new_cache_size长度
                cropped_cache.append((
                    layer_cache[0][:, :, :new_cache_size, :] if layer_cache[0] is not None else None,
                    layer_cache[1][:, :, :new_cache_size, :] if layer_cache[1] is not None else None
                ))
        return tuple(cropped_cache)
        
    def _deep_copy_cache(self, kv_cache: Dict) -> Dict:
        """
        深拷贝KV缓存，用于在拒绝时回退
        """
        if kv_cache is None:
            return None
        
        copied_cache = []
        for layer_cache in kv_cache:
            if layer_cache is None:
                copied_cache.append(None)
            else:
                copied_cache.append((
                    layer_cache[0].clone() if layer_cache[0] is not None else None,
                    layer_cache[1].clone() if layer_cache[1] is not None else None
                ))
        return tuple(copied_cache)
