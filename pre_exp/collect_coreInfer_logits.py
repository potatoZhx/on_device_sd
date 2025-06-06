import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import os
import tqdm
from draft_moe.core_moe import CoreMoE

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# dataset_dir = "/zx_data1/models/datasets/wikitext"
save_dir = "/zx_data1/sparsity/on_device_sd/log/logits"
model_dir = "/zx_data1/models/mixtral/models--mistralai--Mixtral-8x7B-v0.1"
model_name = "Mixtral-8x7B"
max_examples = 10000

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = load_dataset(path=dataset_dir, name="wikitext-103-raw-v1", split="train")
dataset = load_dataset('parquet', data_files="/zx_data1/models/datasets/wikitext-tmp/train-00000-of-00002.parquet", split="train")
dataset = dataset.filter(lambda x: len(x["text"]) > 5) # 过滤空行和短文本

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
model.eval()

model_coreinfer = CoreMoE()

all_probs_origin = []
all_probs_coreinfer = []
num_collected = 0

os.makedirs(save_dir, exist_ok=True)
print(f"开始收集 probs，保存到 {save_dir}...")

with torch.no_grad():
    for example_idx, example in enumerate(tqdm.tqdm(dataset)):
        if num_collected >= max_examples:
            break

        text = example["text"]
        if not text or not text.strip():
            continue
        
        # 使用 truncation 和 max_length 防止过长序列导致内存不足 model.config.max_position_embeddings 通常是模型支持的最大长度
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
        inputs_ids = inputs.input_ids.to(model.device)
        if inputs_ids.shape[1] == 0:
            continue

        outputs = model(inputs_ids)
        logits_tensor = outputs.logits # logits.shape: (batch_size=1, sequence_length, vocab_size)
        for i in range(2, logits_tensor.shape[1]):
            logits = logits_tensor[0, i, :] # logits_tensor[0, i, :] 是在给定前面 0...i-1 个token的条件下，预测第 i 个token的logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().tolist()

            prefix_ids = inputs_ids[0, :i-1]
            prefix_tokens = inputs_ids[0, :i-1].cpu().tolist()

            all_probs_origin.append({
                "prefix_tokens": prefix_tokens,
                "probs": probs,
            })
            all_probs_coreinfer.append({
                "prefix_tokens": prefix_tokens,
                "probs": model_coreinfer.next_prob(prefix_ids),
            })
            
            num_collected += 1
        
        if (example_idx + 1) % 100 == 0:
            print(f"已处理 {example_idx + 1} 个样本， 收集到 {num_collected} 个 logits 分布")

if not all_probs:
    print("没有收集到任何 probs 分布")
else:
    all_prefix_tokens = [item["prefix_tokens"] for item in all_probs]
    all_probs_array = np.array([item["probs"] for item in all_probs], dtype=np.float16)
    all_prefix_tokens_array = np.array(all_prefix_tokens, dtype=object)
    np.savez_compressed(
        os.path.join(save_dir, f"{model_name}_probs.npz"),
        prefix_tokens=all_prefix_tokens_array,
        probs=all_probs_array
    )        
print("收集 probs 完成")



