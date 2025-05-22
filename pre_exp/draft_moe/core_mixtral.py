import time
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers

import config


class CoreMixtral:
    def __init__(self):
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda:0")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=self.dtype,
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.beam_width = config.beam_width
        
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.vram_limit = config.vram_limit * 1024 * 1024 * 1024
        self.popular_experts = config.popular_experts

        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        self.export_loc = np.zeros(
            (self.n_layer, self.n_expert), dtype=int
        )
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}")
        
        self.bring_fixed_expert_to_gpu(config.start_layer, config.end_layer)
        left_expert_on_gpu = n_expert_on_gpu - (self.n_layer - (config.end_layer - config.start_layer)) * self.n_expert
        self.set_and_bring_expert_to_gpu(left_expert_on_gpu, config.start_layer, config.end_layer)

        print("Model is ready. GPU memory usage: ", torch.cuda.memory_allocated() / (1024 * 1024 * 1024))

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.device)
        self.model.embed_tokens.to(self.device)
        self.model.norm.to(self.device)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.device)
            self.model.layers[i].input_layernorm.to(self.device)
            self.model.layers[i].block_sparse_moe.gate.to(self.device)
            self.model.layers[i].post_attention_layernorm.to(self.device)
            # only model.layers[i].block_sparse_moe.experts is on CPU

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel() # 获取tensor的元素个数
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        total_vram = self.vram_limit
        free_vram = total_vram * 0.95 - torch.cuda.memory_allocated(self.device) # TODO: 保留kv cache的vram
        return int((free_vram) //( n_param * 2))
    
    def bring_fixed_expert_to_gpu(self, start_layer, end_layer):
        """Bring fixed experts to GPU"""
        for i_layer in itertools.chain(range(0, start_layer), range(end_layer, self.n_layer)):
            for i_expert in range(self.n_expert):
                self.export_loc[i_layer][i_expert] = 1
                self.model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.device)

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
                    self.export_loc[i_layer][i_expert] = 1
                    self.model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.device)

        cnt = (end_layer - start_layer - 1) * 2
        expert_i = 0
        while cnt < n_expert_on_gpu:
            i_layer, i_expert = self.popular_experts[expert_i]
            while self.export_loc[i_layer][i_expert] == 1:
                expert_i += 1
                i_layer, i_expert = self.popular_experts[expert_i]
            self.export_loc[i_layer][i_expert] = 1
            self.model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.device)
            cnt += 1
            expert_i += 1
    
    def tokenize(self, text):
        input_ids = []
        encodings = self.tokenizer(text, return_tensors="pt")
        input_id = encodings.input_ids.to(self.device)
        for i in range(self.beam_width): # TODO: 删除beam_width
            input_ids.append(input_id[0])

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.device)
        inps = self.model.embed_tokens(inps)

        for i_layer, layer in enumerate(self.model.layers):
           original_inps_shape = inps.shape

           inps_residual = inps
           inps = layer.input_layernorm(inps)
           inps, self_attn_weights, present_key_value = layer.self_attn(
               inps,
               position_ids=position_ids,
               past_key_value=self.past_key_value,
               use_cache=True,
           )

           # inps.shape: (batch_size, seq_len, hidden_dim)
           inps = inps + inps_residual
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
               self.export_loc[i_layer], 
               device=self.device,
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
               if self.export_loc[i_layer][i_expert] == 0:
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
                    0, token_i, current_state.to(inps.dtype)
               )

           inps = inps_residual + inps_after_experts.reshape(original_inps_shape)
        
        inps = self.model.norm(inps)
        lm_logits = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logits

    def initial_beam_tensor(self, input_tensor):
        # transpose tensor of shape (beam_width, seq_len, beam_width) to (beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16) # TODO: set appropriately
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        input_ids, position_ids = self.tokenize(text)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]
    
        tick = time.time()
        prefill_time, decode_time = 0, 0
        is_decode = False
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False # TODO: 删除beam_search
        probs = torch.full((input_ids.shape[0], 1), 1.0)

        for i_token in range(output_token):
            if self.beam_width == 1:
                print(self.tokenizer.decode(input_ids[0]))
            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])
            
            logits = self.mixtral_forward(input_ids, position_ids)

            # logits.shape: (batch_size, seq_len, vocab_size)
            logits = logits.to("cpu")
            # normalize logits
            logits = F.softmax(logits, dim=-1)

            # greedy search
            # output = torch.argmax(logits, dim=-1)

            # beam search
            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.device)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0).view(-1, 1)
            )

            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")

        return (
            prefill_time,
            decode_time
        )

           
               
               
           
model = CoreMixtral()
prefill_time, decode_time = model.generate(
    config.input, output_token=20
)
print(
    f"prefill_time: {prefill_time}, decode_time: {decode_time}"
)




