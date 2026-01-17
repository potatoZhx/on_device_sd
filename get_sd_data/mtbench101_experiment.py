import os
import json
import torch
import time
import random
import traceback
import gc 
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === DynamicCache 补丁 ===
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, input_seq_len, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length

# === 导入核心逻辑 ===
# 假设 ExpertSubsetInference.py 在同级目录
from ExpertSubsetInference import apply_expert_subset_to_model, collect_moe_metadata

def generate_request_id():
    return int(time.time() * 1000000) + random.randint(0, 1000)

def prepare_mtbench_data(tokenizer):
    """加载 MTBench 数据并格式化"""
    DATA_FILE = "/data2/group_谈海生/lagin/data/mtbench101/mtbench101.jsonl"
    print(f"正在加载 MTBench 数据: {DATA_FILE}")
    
    samples = []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                history = item.get("history", [])
                
                # Format history
                text = ""
                for turn in history:
                    if "user" in turn:
                        text += f"User: {turn['user']}\n"
                    if "bot" in turn:
                        text += f"Bot: {turn['bot']}\n"
                
                if not text: continue
                
                # Tokenize
                encodings = tokenizer(text, return_tensors="pt")
                samples.append(encodings.input_ids)
                
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

    print(f"加载了 {len(samples)} 个样本。")
    return samples

def manual_decode_step(model, input_ids, attention_mask, past_key_values):
    """
    执行单步解码，同时返回 Logits 和 Embedding
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True  # [关键] 请求返回所有层的 hidden states
    )
    
    next_token_logits = outputs.logits[:, -1, :]
    
    # 获取最后一层的输出 Embedding (Last Hidden State)
    # outputs.hidden_states 是一个 tuple，最后一个元素是最后一层的输出
    # Shape: [Batch, Seq_Len, Hidden_Dim] -> 取最后一个 Step: [Batch, Hidden_Dim]
    last_embedding = outputs.hidden_states[-1][:, -1, :]
    
    return next_token_logits, last_embedding, outputs.past_key_values

def run_experiment():
    # === 配置参数 ===
    MODEL_PATH = "/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base"
    MODEL_NAME = "Qwen3-30B-A3B-Base"
    BASE_MAX_LEN = 1024 
    NUM_DECODE_STEPS = 10
    TOP_M = 2
    P_THRESHOLD = 0.9
    REPLACE_COUNT = 4
    TASK_MODE = 'replace_with_topp' # ["replace_with_topp", "replace_last_two_with_topp", "replace_last_one_with_topp"]
    
    MAX_SAMPLES = 300
    
    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    # 修改结果目录以区分
    RESULT_DIR = f"./get_sd_data/data/mtbench_results_{REPLACE_COUNT}_with_{MODEL_NAME}"
    os.makedirs(RESULT_DIR, exist_ok=True)

    SUMMARY_FILE_PATH = f"{RESULT_DIR}/experiment_summary_{TIMESTAMP}.jsonl"
    print(f"📄 汇总数据将追加写入: {SUMMARY_FILE_PATH}")

    print(f"加载模型: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 应用 MoE 干预
    model = apply_expert_subset_to_model(model, use_top_m=TOP_M, mode=TASK_MODE, p_threshold=P_THRESHOLD, replace_count=REPLACE_COUNT)

    # 加载数据
    raw_batches = prepare_mtbench_data(tokenizer)
    if not raw_batches: return
    
    total_samples = 0
    data_to_process = raw_batches[:MAX_SAMPLES]
    pbar = tqdm(data_to_process, total=len(data_to_process), desc="Processing", unit="sample")

    for raw_chunk in pbar:
        # 随机截取长度，但不能超过实际长度
        actual_len = raw_chunk.size(1)
        target_len = random.randint(1, min(actual_len, BASE_MAX_LEN))
        
        sample_input = raw_chunk[:, :target_len].to(model.device)
        
        initial_attention_mask = torch.ones(sample_input.shape, device=model.device, dtype=torch.long)
        
        req_id = generate_request_id()
        pbar.set_description(f"ReqID: {req_id} | Len: {target_len}")

        try:
            # A. Prefill
            for layer in model.model.layers: 
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "mode"):
                    layer.mlp.mode = "standard"
            
            with torch.no_grad():
                outputs = model(input_ids=sample_input, attention_mask=initial_attention_mask, use_cache=True)
                past_key_values = outputs.past_key_values
                prefill_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            prefill_id = prefill_token.item()

            # B. Intervention
            curr_input = prefill_token
            # 使用 deepcopy 避免污染原始 Cache
            curr_kv = copy.deepcopy(past_key_values)
            curr_mask = torch.cat([initial_attention_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)
            intervention_out, intervention_data = [], []
            
            for step in range(NUM_DECODE_STEPS):
                for layer in model.model.layers: 
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "mode"):
                        layer.mlp.mode = TASK_MODE
                
                with torch.no_grad():
                    logits, embedding, new_kv = manual_decode_step(model, curr_input, curr_mask, curr_kv)
                    next_token = logits.argmax(dim=-1)
                
                meta = collect_moe_metadata(model)
                intervention_data.append({
                    "step": step + 1,
                    # "dynamic_m": [m.get('dynamic_m', []) for m in meta],
                    "router_original": {"ids": [m.get('original_ids', []) for m in meta], "weights": [m.get('original_weights', []) for m in meta]},
                    "router_modified": {"ids": [m.get('modified_ids', []) for m in meta], "weights": [m.get('final_weights', []) for m in meta]},
                    "full_logits": logits[0].float().cpu().numpy().tolist(),
                    "final_embedding": embedding[0].float().cpu().numpy().tolist()
                })
                intervention_out.append(next_token.item())
                curr_input, curr_kv = next_token.unsqueeze(0), new_kv
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)

            # C. Baseline
            for layer in model.model.layers: 
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "mode"):
                    layer.mlp.mode = "standard"
            
            with torch.no_grad():
                # 复用 Prefill 的 KV Cache，避免重新计算
                curr_kv = copy.deepcopy(past_key_values)
                curr_input = prefill_token
            
            # 重置 Mask
            curr_mask = torch.cat([initial_attention_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)
            baseline_out, baseline_data = [], []

            for step in range(NUM_DECODE_STEPS):
                for layer in model.model.layers: 
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "mode"):
                        layer.mlp.mode = "standard"
                
                with torch.no_grad():
                    logits, embedding, new_kv = manual_decode_step(model, curr_input, curr_mask, curr_kv)
                    next_token = logits.argmax(dim=-1)
                
                meta = collect_moe_metadata(model)
                baseline_data.append({
                    "step": step + 1,
                    "router_standard": {"ids": [m.get('original_ids', []) for m in meta], "weights": [m.get('original_weights', []) for m in meta]},
                    "full_logits": logits[0].float().cpu().numpy().tolist(),
                    "final_embedding": embedding[0].float().cpu().numpy().tolist()
                })
                baseline_out.append(next_token.item())
                curr_input, curr_kv = next_token.unsqueeze(0), new_kv
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)

            # D. Save Data
            match_rate = sum(1 for a, b in zip(intervention_out, baseline_out) if a == b) / len(intervention_out)
            
            record = {
                "metadata": {"req_id": req_id, "len": target_len, "params": {"topm": TOP_M, "p": P_THRESHOLD, "replace_count": REPLACE_COUNT}},
                "data": {"input": sample_input[0].tolist(), "prefill": [prefill_id]},
                "intervention": {"output": intervention_out, "steps": intervention_data},
                "baseline": {"output": baseline_out, "steps": baseline_data},
                "analysis": {"match_rate": match_rate}
            }
            
            with open(SUMMARY_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            total_samples += 1
            
            del record, intervention_data, baseline_data, embedding, logits
            if total_samples % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        except Exception as e:
            tqdm.write(f"❌ Error processing {req_id}: {e}")
            traceback.print_exc()
            continue

    print("\n" + "="*50)
    print("✅ 实验完成！")
    print(f"📂 结果目录 (绝对路径): {os.path.abspath(RESULT_DIR)}")
    print(f"📄 汇总文件 (JSONL): {os.path.abspath(SUMMARY_FILE_PATH)}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
