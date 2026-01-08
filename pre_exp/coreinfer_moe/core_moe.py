import time
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

try:
    from . import config
except ImportError:
    import config


class CoreMoE:
    def __init__(self):
        self.dtype = torch.bfloat16
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        self.device = self.model.device
        self.lm_head = self.model.lm_head
        self.model = self.model.model

        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.vram_limit = config.vram_limit * 1024 * 1024 * 1024

        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.expert_loc = np.zeros(
            (self.n_layer, self.n_expert), dtype=int
        )
        self.expert_popularity = np.zeros(
            (self.n_layer, self.n_expert), dtype=float
        )

        # n_expert_on_gpu = 40
        # print(f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}")
        
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.bring_fixed_layer_to_gpu(config.start_layer, config.end_layer)
        # self.left_expert_on_gpu = n_expert_on_gpu - (self.n_layer - (config.end_layer - config.start_layer)) * self.n_expert
        self.n_expert_per_layer = 2  # self.left_expert_on_gpu // (config.end_layer - config.start_layer)

        # print("Model is ready. GPU memory usage: ", torch.cuda.memory_allocated(torch.device) / (1024 * 1024 * 1024))

    
    def bring_fixed_layer_to_gpu(self, start_layer, end_layer):
        """Bring fixed experts to GPU"""
        for i_layer in range(0, start_layer):
            self.expert_loc[i_layer] = np.ones(self.n_expert, dtype=int)

        for i_layer in range(end_layer, self.n_layer):
            self.expert_loc[i_layer] = np.ones(self.n_expert, dtype=int)

    def set_and_bring_expert_to_gpu(self, n_expert_on_gpu, start_layer, end_layer):
        """Set the expert on GPU"""

        popular_experts_of_layer = {}
        for i_layer, i_expert in self.popular_experts:
            if i_layer not in popular_experts_of_layer:
                popular_experts_of_layer[i_layer] = []
            popular_experts_of_layer[i_layer].append(i_expert)

        if n_expert_on_gpu < (end_layer - start_layer - 1) * 2:
            # 返回模型加载失败信息，程序退出
            print(f"Model load failed: n_expert_on_gpu < < (end_layer - start_layer - 1) * 2")
            exit(1)
        
        # 每层选取2个未放入gpu的热门expert载入gpu，保证每层至少有2个expert在gpu上
        for i_layer in range(start_layer, end_layer):
            for i_expert in popular_experts_of_layer[i_layer][:2]:
                    self.expert_loc[i_layer][i_expert] = 1
                    self.model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.device)

        cnt = (end_layer - start_layer - 1) * 2
        expert_i = 0
        while cnt < n_expert_on_gpu:
            i_layer, i_expert = self.popular_experts[expert_i]
            while self.expert_loc[i_layer][i_expert] == 1:
                expert_i += 1
                i_layer, i_expert = self.popular_experts[expert_i]
            self.expert_loc[i_layer][i_expert] = 1
            self.model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.device)
            cnt += 1
            expert_i += 1
    
    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids):
        # print(f"expert_loc: {self.expert_loc}")
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        hidden_dim = self.model.config.hidden_size
        inps = self.model.embed_tokens(input_ids)
        inputs_embeds = inps

        for i_layer, layer in enumerate(self.model.layers):
           original_inps_shape = inps.shape

           inps_residual = inps
           inps = layer.input_layernorm(inps)
           inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask = None,
                    input_shape=input_ids.shape,
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=self.past_key_values_length,
                )
            )

           # inps.shape: (batch_size, seq_len, hidden_dim)
           inps = inps + inps_residual.to(inps.device)
           inps_residual = inps
           inps = layer.post_attention_layernorm(inps)
           inps = inps.view(-1, hidden_dim)
           # inps.shape: (batch_size*seq_len, hidden_dim)
           router_logits = layer.block_sparse_moe.gate(inps)
           routing_weights = F.softmax(router_logits, dim=-1)
           routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
           # routing_weights.shape: (batch_size*seq_len, num_experts)
           # 从在gpu中的expert中选取top2个
           expert_loc_i = torch.tensor(
               self.expert_loc[i_layer], 
               device=routing_weights.device,
               dtype=self.dtype,
           ).unsqueeze(0).repeat(routing_weights.shape[0], 1)
           routing_weights = routing_weights * expert_loc_i
           # routing_weights.shape: (batch_size*seq_len, num_experts)
           # selected_experts.shape: (batch_size*seq_len, 2)
           routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)

           inps_after_experts = torch.zeros_like(inps, device=self.device)
           experts = layer.block_sparse_moe.experts
           # expert_mask.shape: (num_experts, 2, batch_size*seq_len)
           expert_mask = torch.nn.functional.one_hot(
               selected_experts, num_classes=self.n_expert
           ).permute(2, 1, 0)
           for i_expert in range(len(experts)):
               if self.expert_loc[i_layer][i_expert] == 0:
                   continue
               top_i, token_i = torch.where(expert_mask[i_expert])

               if(token_i.shape[0] == 0):
                    continue # Expert_i has no tokens
           
               token_i_list = token_i.tolist()
               top_i_list = top_i.tolist()

               current_state = inps[None, token_i_list].reshape(-1, hidden_dim)
               current_state = experts[i_expert](
                    current_state, routing_weights[token_i_list, top_i_list, None]
               )
               inps_after_experts.index_add_(
                    0, token_i.to(inps_after_experts.device), current_state.to(inps.dtype).to(inps_after_experts.device)
               )

           inps = inps_residual.to(inps.device) + inps_after_experts.reshape(original_inps_shape).to(inps.device)
        
        inps = self.model.norm(inps)
        lm_logits = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logits

    def __call__(self, input_ids):
        self.model.eval()
        return self.model(input_ids)
    
    def update_experts_loc(self):
        for i_layer in range(self.start_layer, self.end_layer):
            # print(f"expert popularity for layer {i_layer}: {self.expert_popularity[i_layer]}")
            _, on_gpu_experts = torch.topk(
                torch.tensor(self.expert_popularity[i_layer]), 
                k=self.n_expert_per_layer
            )

            self.expert_loc[i_layer] = torch.nn.functional.one_hot(
                on_gpu_experts, num_classes=self.n_expert
            ).sum(dim=0).cpu().numpy()

            # print(f"expert location for layer {i_layer}: {self.expert_loc[i_layer]}")

    @torch.no_grad()
    def prefill_forward(self, input_ids, position_ids):
        ''' 
        prefill forward pass
        记录expert的route weights, 更新expert_popularity
        更新in gpu experts
        '''
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        hidden_dim = self.model.config.hidden_size
        inps = self.model.embed_tokens(input_ids) 
        inputs_embeds = inps

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape
            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask = None,
                    input_shape=input_ids.shape,
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=self.past_key_values_length,
                )
            )

            inps = inps + inps_residual.to(inps.device)
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            self.expert_popularity[i_layer] = routing_weights.cpu().sum(axis=0).float().numpy() / routing_weights.shape[0]
            # print(f"expert popularity for layer {i_layer}: {self.expert_popularity[i_layer]}")

            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)

            inps_after_experts = torch.zeros_like(inps, device=self.device)
            experts = layer.block_sparse_moe.experts
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.n_expert
            ).permute(2, 1, 0)
            for i_expert in range(len(experts)):
                top_i, token_i = torch.where(expert_mask[i_expert])
                if(token_i.shape[0] == 0):
                    continue
                token_i_list = token_i.tolist()
                top_i_list = top_i.tolist()

                current_state = inps[None, token_i_list].reshape(-1, hidden_dim)
                current_state = experts[i_expert](
                    current_state, routing_weights[token_i_list, top_i_list, None]
                )
                inps_after_experts.index_add_(
                    0, token_i.to(inps_after_experts.device), current_state.to(inps.dtype).to(inps_after_experts.device)
                )

            inps = inps_residual.to(inps.device) + inps_after_experts.reshape(original_inps_shape).to(inps.device)

        inps = self.model.norm(inps)
        lm_logits = self.lm_head(inps)
        self.present_key_value = present_key_value
        self.past_key_values_length += input_ids.shape[1]

        self.update_experts_loc()

        return lm_logits
    
    @torch.no_grad()
    def decode_forward(self, decode_ids, decode_position_ids):
        # should be called after prefill_forward
        hidden_dim = self.model.config.hidden_size
        inps = self.model.embed_tokens(decode_ids)
        inputs_embeds = inps

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape
            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=decode_position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask = None,
                    input_shape=decode_ids.shape,
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=self.past_key_values_length,
                )
            )

            inps = inps + inps_residual.to(inps.device)
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            expert_loc_i = torch.tensor(
                self.expert_loc[i_layer],
                device=routing_weights.device,
                dtype=torch.bfloat16
            ).unsqueeze(0).repeat(routing_weights.shape[0], 1)
            routing_weights = routing_weights * expert_loc_i
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)

            inps_after_experts = torch.zeros_like(inps, device=self.device)
            experts = layer.block_sparse_moe.experts
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.n_expert
            ).permute(2, 1, 0)
            
            for i_expert in range(len(experts)):
                if self.expert_loc[i_layer][i_expert] == 0:
                    continue
                top_i, token_i = torch.where(expert_mask[i_expert])
                if(token_i.shape[0] == 0):
                        continue
                token_i_list = token_i.tolist()
                top_i_list = top_i.tolist()

                current_state = inps[None, token_i_list].reshape(-1, hidden_dim)
                current_state = experts[i_expert](
                    current_state, routing_weights[token_i_list, top_i_list, None]
                )
                inps_after_experts.index_add_(
                    0, token_i.to(inps_after_experts.device), current_state.to(inps.dtype).to(inps_after_experts.device)
                )
            
            inps = inps_residual.to(inps.device) + inps_after_experts.reshape(original_inps_shape).to(inps.device)

        inps = self.model.norm(inps)
        lm_logits = self.lm_head(inps)

        self.present_key_value = present_key_value
        self.past_key_values_length += decode_ids.shape[1]

        return lm_logits


if __name__ == "__main__":
    model = CoreMoE()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model)
    text = "What is the spiciest part of a chili pepper? The"
    # text = "The Tower Building of the Little Rock Arsenal , also known as U.S. Arsenal Building , is a building located in MacArthur Park in downtown Little Rock , Arkansas ."
    inputs = tokenizer(text, return_tensors="pt")
    inputs_ids = inputs.input_ids.to(model.device)
    position_ids = torch.arange(
        0, inputs_ids.shape[-1], dtype=torch.long, device=model.device
    ).unsqueeze(0).view(-1, inputs_ids.shape[-1])

    # inputs_ids对应的token
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs_ids[0].tolist())}")

    n_tokens_decode = 3 # 以len - n_tokens_decode个token为prefill更新core

    logits_ref = model.prefill_forward(inputs_ids, position_ids)[:, -n_tokens_decode:, :]


    prefill_ids = inputs_ids[:, :-n_tokens_decode]
    prefill_position_ids = position_ids[:, :-n_tokens_decode]
    decode_ids = inputs_ids[:, -n_tokens_decode:]
    decode_position_ids = position_ids[:, -n_tokens_decode:]

    model.prefill_forward(prefill_ids, prefill_position_ids)
    logits = model.decode_forward(decode_ids, decode_position_ids)
    # print(logits)

    for i in range(logits_ref.shape[1]):
        # i = logits_ref.shape[1] - 1
        logits_core = logits[0, i, :]
        logits_full = logits_ref[0, i, :]
        probs_core = torch.nn.functional.softmax(logits_core, dim=-1).cpu().tolist()
        probs_full = torch.nn.functional.softmax(logits_full, dim=-1).cpu().tolist()
        acc_rate = np.sum(np.minimum(probs_core, probs_full)) 
        print(f"Accuracy rate of token {i}: {acc_rate:.4f}")


       



