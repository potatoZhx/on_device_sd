import os
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

def prepare_wikitext_data(tokenizer, seq_len=1024):
    """
    加载并预处理 WikiText-2 数据集。
    处理方式：拼接全文 -> 按 \n\n 分隔(可选) -> 切分为固定长度的 Chunk。
    """
    # 路径应该指向你终端 ls 看到的那个文件
    LOCAL_TEST_FILE = "/data2/group_谈海生/lagin/data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    
    print("正在加载 WikiText-2 测试集 (Local Parquet Mode)...")
    
    try:
        # 核心：使用 "parquet" 驱动，并提供文件的映射
        test_data = load_dataset(
            "parquet", 
            data_files={"test": LOCAL_TEST_FILE}, 
            split="test"
        )
    except Exception as e:
        print(f"❌ 错误：加载本地 Parquet 文件失败，请检查路径。")
        print(f"详细错误: {e}")
        return []

    print("正在处理数据 (Tokenizing & Chunking)...")
    # 论文中通常将所有文本拼接，然后用 sliding window 或 fixed stride 切分
    # 这里简单起见，拼接所有文本
    full_text = "\n\n".join(test_data["text"])
    encodings = tokenizer(full_text, return_tensors="pt")
    
    input_ids = encodings.input_ids
    total_length = input_ids.size(1)
    print(f"总 Token 数: {total_length}")
    
    batch_input_ids = []
    stride = seq_len
    
    # 切分 Chunk
    for i in range(0, total_length, stride):
        end_loc = min(i + seq_len, total_length)
        input_id = input_ids[:, i:end_loc]
        
        # 丢弃最后不足长度的片段，保证 batch shape 一致
        if input_id.size(1) == seq_len:
            batch_input_ids.append(input_id)
            
    print(f"生成了 {len(batch_input_ids)} 个长度为 {seq_len} 的样本。")
    return batch_input_ids


def evaluate_perplexity(model, batches, batch_size=8):
    """
    执行评测循环：计算 PPL。
    支持 Batch 并行。
    
    Args:
        batch_size: 批次大小 (默认 8)
    """
    model.eval()
    nlls = []
    
    print(f"开始评测 (PPL) | Batch={batch_size}...")

    # 1. 定义 Collate Function (用于 Batch Padding)
    # 假设 batches 里的元素是 [1, Seq_Len] 的 Tensor
    # 我们需要 tokenizer 的 pad_token_id。由于这里没传入 tokenizer，
    # 尝试从 model.config 或者默认值获取。
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", 0) # 兜底

    def collate_fn(batch_list):
        # batch_list 是 List[Tensor]，每个 Tensor shape [1, Seq]
        # 先 squeeze 掉 batch 维
        tensors = [t.squeeze(0) for t in batch_list]
        
        # Padding (右填充)
        padded_input = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_token_id
        )
        
        # 生成 Mask (1=有效, 0=Padding)
        mask = (padded_input != pad_token_id).long()
        return padded_input, mask

    # 2. 创建 DataLoader
    data_loader = DataLoader(batches, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    pbar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_input_ids, batch_mask in pbar:
            batch_input_ids = batch_input_ids.to("cuda")
            batch_mask = batch_mask.to("cuda")
            target_ids = batch_input_ids.clone()
            
            # 设置 Loss 计算的 Label Mask (忽略 Padding)
            # CrossEntropyLoss 默认忽略 -100
            target_ids[batch_mask == 0] = -100

            # --- Forward ---            
            # 自动计算 Loss。注意：labels 中的 -100 会被自动忽略
            # 设置 use_cache=False 以避免 DeepSeek V2 Lite 模型的 DynamicCache 兼容性问题
            outputs = model(batch_input_ids, labels=target_ids, use_cache=False)
            
            # 收集负对数似然 (NLL)
            nlls.append(outputs.loss)
            
            pbar.set_postfix({
                "Loss": f"{outputs.loss.item():.4f}"
            })

    # 计算最终 PPL
    ppl = torch.exp(torch.stack(nlls).mean())
    
    return ppl.item()