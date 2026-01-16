import os
import json
import torch
import time
import random
import traceback
import gc 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# === DynamicCache 补丁 ===
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, input_seq_len, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length

# === 导入核心逻辑 ===
from ExpertSubsetInference import apply_expert_subset_to_model, collect_moe_metadata

def generate_request_id():
    return int(time.time() * 1000000) + random.randint(0, 1000)

def prepare_wikitext_data(tokenizer, seq_len=1024):
    """加载本地 Parquet 并切分为标准长度 Chunks"""
    LOCAL_TEST_FILE = "/data2/group_谈海生/lagin/data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    print(f"正在加载 WikiText-2 测试集: {LOCAL_TEST_FILE}")
    
    try:
        test_data = load_dataset("parquet", data_files={"test": LOCAL_TEST_FILE}, split="test")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

    print("处理数据...")
    full_text = "\n\n".join(test_data["text"])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids
    total_length = input_ids.size(1)
    
    batch_input_ids = []
    stride = seq_len
    for i in range(0, total_length, stride):
        end_loc = min(i + seq_len, total_length)
        chunk = input_ids[:, i:end_loc]
        if chunk.size(1) == seq_len:
            batch_input_ids.append(chunk)
            
    print(f"生成了 {len(batch_input_ids)} 个样本。")
    return batch_input_ids

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
    MODEL_PATH = "/data2/group_谈海生/lagin/models/DeepSeek-V2-Lite"
    BASE_MAX_LEN = 1024 
    NUM_DECODE_STEPS = 10
    TOP_M = 2
    P_THRESHOLD = 0.9
    TASK_MODE = 'replace_last_one_with_topp' # ["replace_with_topp", "replace_last_two_with_topp", "replace_last_one_with_topp"]
    
    MAX_SAMPLES = 500
    
    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    RESULT_DIR = f"./get_sd_data/data/results_{TASK_MODE}"
    os.makedirs(RESULT_DIR, exist_ok=True)

    SUMMARY_FILE_PATH = f"{RESULT_DIR}/experiment_summary_{TIMESTAMP}.jsonl"
    print(f"📄 汇总数据将追加写入: {SUMMARY_FILE_PATH}")

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    model = apply_expert_subset_to_model(model, use_top_m=TOP_M, mode=TASK_MODE, p_threshold=P_THRESHOLD)

    raw_batches = prepare_wikitext_data(tokenizer, seq_len=BASE_MAX_LEN)
    if not raw_batches: return
    
    total_samples = 0
    data_to_process = raw_batches[:MAX_SAMPLES]
    pbar = tqdm(data_to_process, total=MAX_SAMPLES, desc="Processing", unit="sample")

    for raw_chunk in pbar:
        target_len = random.randint(1, BASE_MAX_LEN)
        sample_input = raw_chunk[:, :target_len].to(model.device)
        
        initial_attention_mask = torch.ones(sample_input.shape, device=model.device, dtype=torch.long)
        
        req_id = generate_request_id()
        pbar.set_description(f"ReqID: {req_id} | Len: {target_len}")

        try:
            # A. Prefill
            for layer in model.model.layers: layer.mlp.mode = "standard"
            with torch.no_grad():
                outputs = model(input_ids=sample_input, attention_mask=initial_attention_mask, use_cache=True)
                past_key_values = outputs.past_key_values
                prefill_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            prefill_id = prefill_token.item()

            # B. Intervention
            curr_input, curr_kv = prefill_token, past_key_values 
            curr_mask = torch.cat([initial_attention_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)
            intervention_out, intervention_data = [], []
            
            for step in range(NUM_DECODE_STEPS):
                for layer in model.model.layers: layer.mlp.mode = TASK_MODE
                with torch.no_grad():
                    # [修改] 接收 embedding 返回值
                    logits, embedding, new_kv = manual_decode_step(model, curr_input, curr_mask, curr_kv)
                    next_token = logits.argmax(dim=-1)
                
                meta = collect_moe_metadata(model)
                intervention_data.append({
                    "step": step + 1,
                    "dynamic_m": [m['dynamic_m'][0] for m in meta],
                    "router_original": {"ids": [m['original_ids'][0] for m in meta], "weights": [m['original_weights'][0] for m in meta]},
                    "router_modified": {"ids": [m['modified_ids'][0] for m in meta], "weights": [m['final_weights'][0] for m in meta]},
                    "full_logits": logits[0].float().cpu().numpy().tolist(),
                    # [新增] 保存 Embedding (转为 list)
                    "final_embedding": embedding[0].float().cpu().numpy().tolist()
                })
                intervention_out.append(next_token.item())
                curr_input, curr_kv = next_token.unsqueeze(0), new_kv
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)

            # C. Baseline
            curr_input, curr_kv = prefill_token, past_key_values
            curr_mask = torch.cat([initial_attention_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)
            baseline_out, baseline_data = [], []

            for step in range(NUM_DECODE_STEPS):
                for layer in model.model.layers: layer.mlp.mode = "standard"
                with torch.no_grad():
                    # [修改] 接收 embedding 返回值
                    logits, embedding, new_kv = manual_decode_step(model, curr_input, curr_mask, curr_kv)
                    next_token = logits.argmax(dim=-1)
                
                meta = collect_moe_metadata(model)
                baseline_data.append({
                    "step": step + 1,
                    "router_standard": {"ids": [m['original_ids'][0] for m in meta], "weights": [m['original_weights'][0] for m in meta]},
                    "full_logits": logits[0].float().cpu().numpy().tolist(),
                    # [新增] 保存 Embedding
                    "final_embedding": embedding[0].float().cpu().numpy().tolist()
                })
                baseline_out.append(next_token.item())
                curr_input, curr_kv = next_token.unsqueeze(0), new_kv
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)

            # D. Save Data
            match_rate = sum(1 for a, b in zip(intervention_out, baseline_out) if a == b) / len(intervention_out)
            
            record = {
                "metadata": {"req_id": req_id, "len": target_len, "params": {"topm": TOP_M, "p": P_THRESHOLD}},
                "data": {"input": sample_input[0].tolist(), "prefill": [prefill_id]},
                "intervention": {"output": intervention_out, "steps": intervention_data},
                "baseline": {"output": baseline_out, "steps": baseline_data},
                "analysis": {"match_rate": match_rate}
            }
            
            # 追加写入大文件
            with open(SUMMARY_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            total_samples += 1
            
            # 显式释放内存
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