import os
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from Moe_LRU import ExpertCache
# from Cache_Prior import CachePriorBlockWrapper
from Cache_Prior_batch import CachePriorBlockWrapper
from torch.utils.data import DataLoader

def move_static_modules_to_gpu(model, device="cuda"):
    """
    将非稀疏专家（Static）组件搬运到 GPU，为 Expert Offloading 腾出 CPU 空间。
    """
    print(f"正在将静态组件移动到 {device}...")
    
    # 1. 移动外层组件
    if hasattr(model.model, "embed_tokens"): model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"): model.model.norm.to(device)
    if hasattr(model, "lm_head"): model.lm_head.to(device)
        
    # 2. 移动每一层中的 Attention 和非稀疏部分
    for layer in model.model.layers:
        layer.self_attn.to(device)
        layer.input_layernorm.to(device)
        layer.post_attention_layernorm.to(device)
        
        target_moe = None
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            target_moe = layer.mlp
        elif hasattr(layer, "block_sparse_moe"):
            target_moe = layer.block_sparse_moe
            
        if target_moe:
            target_moe.gate.to(device)
            target_moe.shared_expert.to(device)
            target_moe.shared_expert_gate.to(device)
            # 注意：experts 列表保留在 CPU，由 Cache 接管
            
    print("静态组件移动完成。")

def apply_cache_prior_to_model(model, cache_ratio=0.5, lambda_val=0.5, top_j=2):
    """
    向模型注入 Cache-Prior 路由逻辑和 LRU 缓存管理器。
    """
    caches = []
    num_experts = model.config.num_experts
    cache_limit = int(num_experts * cache_ratio)
    
    # 获取 Top-K 配置
    if hasattr(model.config, 'num_experts_per_tok'):
        router_top_k = model.config.num_experts_per_tok
    elif hasattr(model.config, 'top_k'):
        router_top_k = model.config.top_k
    else:
        router_top_k = 4

    print(f"注入 Cache 逻辑: Top-K={router_top_k}, Limit={cache_limit}/{num_experts}, Lambda={lambda_val}")

    for layer_idx, layer in enumerate(model.model.layers):
        target_moe = None
        attr_name = ""
        
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            target_moe = layer.mlp
            attr_name = "mlp"
        elif hasattr(layer, "block_sparse_moe"):
            target_moe = layer.block_sparse_moe
            attr_name = "block_sparse_moe"
            
        if target_moe:
            # 实例化 Cache (初始化时会自动 Warm-up 部分专家到 GPU)
            layer_cache = ExpertCache(
                experts_list=target_moe.experts,
                cache_size=cache_limit,
                layer_id=layer_idx,
                device="cuda"
            )
            caches.append(layer_cache)
            
            # 使用 Block Wrapper 替换原始 Block
            new_block = CachePriorBlockWrapper(
                original_block=target_moe,
                cache=layer_cache,
                lambda_val=lambda_val,
                top_j=top_j
            )
            setattr(layer, attr_name, new_block)
            
    return caches

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

# def evaluate_perplexity(model, batches, caches):
#     """
#     执行评测循环：计算 PPL 和 Cache Miss Rate。
#     """
#     model.eval()
#     nlls = []
    
#     # 重置缓存统计
#     for c in caches:
#         c.reset_stats()
    
#     print("开始评测 (PPL & Miss Rate)...")
#     pbar = tqdm(batches, desc="Evaluating")
    
#     with torch.no_grad():
#         for i, input_ids in enumerate(pbar):
#             input_ids = input_ids.to("cuda")
#             target_ids = input_ids.clone()
            
#             # 前向传播 (自动计算 Loss)
#             # 在这一步，ExpertCache.update 已经被触发，统计数据已更新
#             outputs = model(input_ids, labels=target_ids)
            
#             # 收集负对数似然
#             nlls.append(outputs.loss)
            
#             # --- 🟢 新增：实时计算累计 Miss Rate ---
#             # 统计所有层的累计命中数和请求数
#             current_total_hits = sum(c.hits for c in caches)
#             current_total_queries = sum(c.total_queries for c in caches)
            
#             if current_total_queries > 0:
#                 current_miss_rate = 1.0 - (current_total_hits / current_total_queries)
#             else:
#                 current_miss_rate = 0.0
            
#             # --- 🟢 新增：更新进度条显示 ---
#             # 这会在进度条后面显示： "Miss Rate: 15.20%, Loss: 3.4512"
#             pbar.set_postfix({
#                 "Miss Rate": f"{current_miss_rate:.2%}",
#                 "Loss": f"{outputs.loss.item():.4f}"
#             })
            
#             # 如果你想在终端硬打印每一行（不推荐，会刷屏），取消下面注释：
#             print(f"Step {i}: Miss Rate = {current_miss_rate:.2%}")

#     # 计算最终 Perplexity
#     ppl = torch.exp(torch.stack(nlls).mean())
    
#     # 最终再次计算（确保精度）
#     total_hits = sum(c.hits for c in caches)
#     total_queries = sum(c.total_queries for c in caches)
#     miss_rate = 1.0 - (total_hits / total_queries) if total_queries > 0 else 0.0
    
#     return ppl.item(), miss_rate


def evaluate_perplexity(model, batches, caches, batch_size=8, use_mask=True):
    """
    执行评测循环：计算 PPL 和 Cache Miss Rate。
    支持 Batch 并行，并可通过 use_mask 控制是否在 Cache 统计中忽略 Padding。
    
    Args:
        batch_size: 批次大小 (默认 1)
        use_mask: bool, 是否使用 Mask 过滤 Padding Token 的 Cache 更新 (默认 True)
    """
    model.eval()
    nlls = []
    
    # 重置缓存统计
    for c in caches:
        c.reset_stats()
        
    print(f"开始评测 (PPL & Miss Rate) | Batch={batch_size} | Use Mask={use_mask}...")

    # 1. 定义 Collate Function (用于 Batch Padding)
    # 假设 batches 里的元素是 [1, Seq_Len] 的 Tensor
    # 我们需要 tokenizer 的 pad_token_id。由于这里没传入 tokenizer，
    # 我们尝试从 model.config 或者默认值获取。
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
            
            # --- 🟢 [核心逻辑] 设置全局 Mask ---
            if use_mask:
                # 如果开启，将 Mask 展平并传给 Wrapper
                # Wrapper 内部会根据这个 Mask 决定哪些 Token 更新 Cache
                CachePriorBlockWrapper.CURRENT_MASK = batch_mask.view(-1)
                
                # 设置 Loss 计算的 Label Mask (忽略 Padding)
                # CrossEntropyLoss 默认忽略 -100
                target_ids[batch_mask == 0] = -100
            else:
                # 如果关闭，Cache 更新将包含 Padding (模拟无差别处理)
                CachePriorBlockWrapper.CURRENT_MASK = None
                # 注意：即使不 Mask Cache，计算 PPL 时通常还是应该忽略 Padding 的 Loss
                # 这里为了严谨，Loss 计算始终忽略 Padding
                target_ids[batch_input_ids == pad_token_id] = -100

            # --- Forward ---
            # 自动计算 Loss。注意：labels 中的 -100 会被自动忽略
            outputs = model(batch_input_ids, labels=target_ids)
            
            # 收集负对数似然 (NLL)
            # outputs.loss 是标量 (当前 Batch 的平均 Loss)
            # 为了计算全局 PPL，我们需要加权平均 (乘以有效 Token 数) 
            # 或者简单处理：假设每个 Batch 有效长度差不多，直接存 mean loss
            nlls.append(outputs.loss)
            
            # --- 实时统计 ---
            current_total_hits = sum(c.hits for c in caches)
            current_total_queries = sum(c.total_queries for c in caches)
            
            if current_total_queries > 0:
                current_miss_rate = 1.0 - (current_total_hits / current_total_queries)
            else:
                current_miss_rate = 0.0
            
            pbar.set_postfix({
                "Miss": f"{current_miss_rate:.2%}",
                "Loss": f"{outputs.loss.item():.4f}"
            })

    # 清理全局状态
    CachePriorBlockWrapper.CURRENT_MASK = None

    # 计算最终 PPL
    ppl = torch.exp(torch.stack(nlls).mean())
    
    # 最终统计
    total_hits = sum(c.hits for c in caches)
    total_queries = sum(c.total_queries for c in caches)
    miss_rate = 1.0 - (total_hits / total_queries) if total_queries > 0 else 0.0
    
    return ppl.item(), miss_rate

# ----------------------------------------------------------------------------------------------------------------
# mmlu 相关辅助函数
# ----------------------------------------------------------------------------------------------------------------


def format_mmlu_example(example, include_answer=True):
    """
    辅助函数：将单条数据格式化为 MMLU 标准问答格式。
    格式参考：
    Question: ...
    A. ...
    B. ...
    C. ...
    D. ...
    Answer: A
    """
    prompt = f"Question: {example['question']}\n"
    choices = ["A", "B", "C", "D"]
    for i, choice in enumerate(example['choices']):
        prompt += f"{choices[i]}. {choice}\n"
    
    prompt += "Answer:"
    if include_answer:
        # example['answer'] 是 0-3 的整数，转换为 A-D
        prompt += f" {choices[example['answer']]}\n\n"
    else:
        # 如果是测试问题，Answer: 后面留空，等待模型预测
        prompt += " " 
        
    return prompt

def prepare_mmlu_data(tokenizer, subset="global_facts", num_shots=5):
    """
    加载本地 MMLU 数据并构建 5-shot Prompt。
    返回: List[ (input_ids, label_index) ]
    """
    # 1. 构造本地路径
    # 假设你的目录结构是 /data/mmlu/{subset}/...parquet
    base_path = f"/data2/group_谈海生/lagin/data/mmlu/{subset}"
    
    # 自动寻找 dev 和 test 的具体文件名
    dev_file = os.path.join(base_path, "dev-00000-of-00001.parquet")
    test_file = os.path.join(base_path, "test-00000-of-00001.parquet")
    
    # 兼容性检查：有些下载方式可能没有子文件夹，直接在 mmlu 目录下
    if not os.path.exists(dev_file):
        print(f"⚠️ 在 {base_path} 未找到标准文件，尝试搜索...")
        # 这里你可以根据实际下载的文件名调整
        # 如果是用 git clone cais/mmlu 下载的，结构可能是 data/{subset}/...
        pass 

    print(f"正在加载 MMLU [{subset}] (Local Mode)...")
    print(f"  - Dev  (Shots): {dev_file}")
    print(f"  - Test (Eval) : {test_file}")

    try:
        # 同时加载 dev 和 test
        dataset = load_dataset(
            "parquet", 
            data_files={
                "dev": dev_file,
                "test": test_file
            }
        )
    except Exception as e:
        print(f"❌ 错误：加载本地 Parquet 文件失败。")
        print(f"请检查路径是否正确: {base_path}")
        print(f"详细错误: {e}")
        return []

    # 2. 构建 5-shot 头部 (使用 'dev' 集)
    # 论文：MMLU 采用 5-shot 方法 [cite: 497, 596]
    if num_shots > 0:
        # 选取前 num_shots 个例子
        shot_examples = dataset['dev'].select(range(min(num_shots, len(dataset['dev']))))
        few_shot_prompt = ""
        for ex in shot_examples:
            few_shot_prompt += format_mmlu_example(ex, include_answer=True)
    else:
        few_shot_prompt = ""

    print(f"Few-shot header length: {len(few_shot_prompt)} chars")
    # print(f"Header Example:\n{few_shot_prompt[:200]}...\n")

    # 3. 处理测试集 (使用 'test' 集)
    processed_samples = []
    eval_data = dataset['test']
    
    print(f"正在构建 {len(eval_data)} 个测试样本...")
    
    for ex in eval_data:
        # 拼接: [5个示例] + [当前问题]
        full_prompt = few_shot_prompt + format_mmlu_example(ex, include_answer=False)
        
        # Tokenize
        # 论文提到 MMLU 应用于整个序列 [cite: 597]，通常不需要截断，Qwen 上下文足够
        inputs = tokenizer(text=full_prompt, return_tensors="pt", add_special_tokens=False)
        
        # 记录正确答案的索引 (0=A, 1=B, 2=C, 3=D)
        # 这个 label_idx 将用于后续与 logits 的 argmax 比较
        label_idx = ex['answer']
        
        processed_samples.append((inputs.input_ids, label_idx))
        
    return processed_samples

# def evaluate_mmlu(model, samples, tokenizer, caches):
#     """
#     MMLU 评测主循环：计算 Accuracy 和 Miss Rate
#     """
#     model.eval()
#     for c in caches: c.reset_stats()
    
#     correct = 0
#     total = 0
    
#     # 获取 A, B, C, D 在词表中的 Token ID
#     # 注意：Qwen 的 Tokenizer 可能会在前面加空格，需仔细核对
#     # 这里假设是单纯的字母。更稳妥的方法是 encode(" A")[-1] 或 encode("A")[-1]
#     # Qwen通常: "A" -> [32], " A" -> [220] 等，取决于前面是否有空格。
#     # MMLU 格式是 "Answer:", 所以后面应该接 " A" (带空格) 或 "A"
    
#     # 我们取出这四个候选 token 的 ID
#     candidate_tokens = [" A", " B", " C", " D"] # 常用格式
#     candidate_ids = [tokenizer.encode(t)[-1] for t in candidate_tokens]
    
#     # 映射表: 0->ID(A), 1->ID(B)...
#     id_map = {i: tid for i, tid in enumerate(candidate_ids)}
    
#     print(f"Candidate Token IDs (A,B,C,D): {candidate_ids}")
    
#     pbar = tqdm(samples, desc="MMLU Eval")
    
#     with torch.no_grad():
#         for input_ids, label_idx in pbar:
#             input_ids = input_ids.to("cuda")
            
#             # Forward
#             outputs = model(input_ids)
            
#             # 取最后一个 token 的 logits
#             next_token_logits = outputs.logits[:, -1, :] # [1, Vocab]
            
#             # 提取 A,B,C,D 的分数
#             option_logits = next_token_logits[0, candidate_ids] # [4]
            
#             # 选分数最大的那个
#             pred_idx = torch.argmax(option_logits).item() # 0,1,2,3
            
#             if pred_idx == label_idx:
#                 correct += 1
#             total += 1
            
#             # 更新进度条
#             curr_acc = correct / total
            
#             total_hits = sum(c.hits for c in caches)
#             total_queries = sum(c.total_queries for c in caches)
#             miss_rate = 1.0 - (total_hits/total_queries) if total_queries>0 else 0
            
#             pbar.set_postfix({"Acc": f"{curr_acc:.2%}", "Miss": f"{miss_rate:.2%}"})
            
#     final_acc = correct / total
    
#     total_hits = sum(c.hits for c in caches)
#     total_queries = sum(c.total_queries for c in caches)
#     final_miss = 1.0 - (total_hits/total_queries) if total_queries>0 else 0
    
#     return final_acc, final_miss


def evaluate_mmlu(model, samples, tokenizer, caches, batch_size=8, use_mask=True):
    """
    MMLU 评测主循环
    
    Args:
        batch_size: 并行批次大小
        use_mask: bool, 是否使用 Mask 过滤 Padding Token 的 Cache 更新
                  True = 仅统计有效 Token (推荐)
                  False = Padding 也算 Cache Hit/Miss
    """
    model.eval()
    for c in caches: c.reset_stats()
    
    correct = 0
    total = 0
    
    candidate_tokens = [" A", " B", " C", " D"]
    candidate_ids = [tokenizer.encode(t)[-1] for t in candidate_tokens]
    
    # --- Collate Function ---
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def collate_fn(batch):
        input_ids_list = [item[0].squeeze(0) for item in batch]
        labels = [item[1] for item in batch]
        
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        
        # 生成 Mask (1=有效, 0=Padding)
        attention_mask = (padded_inputs != tokenizer.pad_token_id).long()
        lengths = torch.tensor([x.size(0) for x in input_ids_list])
        
        return padded_inputs, torch.tensor(labels), attention_mask, lengths

    data_loader = DataLoader(samples, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    print(f"开始评测 (Batch Size={batch_size}, Use Mask={use_mask})...")
    
    # 清理状态
    CachePriorBlockWrapper.CURRENT_MASK = None
    
    pbar = tqdm(data_loader, desc="MMLU Eval")
    
    with torch.no_grad():
        for batch_input_ids, batch_labels, batch_mask, batch_lengths in pbar:
            batch_input_ids = batch_input_ids.to("cuda")
            batch_mask = batch_mask.to("cuda")
            
            # --- 🟢 [核心控制] 设置全局 Mask ---
            if use_mask:
                # 开启过滤：只让有效 Token 更新 Cache
                CachePriorBlockWrapper.CURRENT_MASK = batch_mask.view(-1)
            else:
                # 关闭过滤：Padding 也算
                CachePriorBlockWrapper.CURRENT_MASK = None
            
            # Forward (注意：attention_mask 用于计算，始终需要传)
            outputs = model(batch_input_ids, attention_mask=batch_mask)
            
            # --- Accuracy 计算 (始终忽略 Padding) ---
            batch_indices = torch.arange(batch_input_ids.size(0), device="cuda")
            last_token_indices = (batch_lengths - 1).to("cuda")
            
            target_logits = outputs.logits[batch_indices, last_token_indices, :]
            option_logits = target_logits[:, candidate_ids]
            pred_indices = torch.argmax(option_logits, dim=-1).cpu().tolist()
            
            for pred, true_label in zip(pred_indices, batch_labels):
                if pred == true_label:
                    correct += 1
                total += 1
            
            # 实时更新统计
            curr_acc = correct / total
            total_hits = sum(c.hits for c in caches)
            total_queries = sum(c.total_queries for c in caches)
            miss_rate = 1.0 - (total_hits/total_queries) if total_queries>0 else 0
            
            pbar.set_postfix({"Acc": f"{curr_acc:.2%}", "Miss": f"{miss_rate:.2%}"})
            
    # 清理全局变量
    CachePriorBlockWrapper.CURRENT_MASK = None
    
    final_acc = correct / total
    total_hits = sum(c.hits for c in caches)
    total_queries = sum(c.total_queries for c in caches)
    final_miss = 1.0 - (total_hits/total_queries) if total_queries>0 else 0
    
    return final_acc, final_miss




# --- 追加到 moe_eval_utils.py 末尾 ---

def extract_answer_gsm8k(text):
    """
    从模型生成的文本中提取最终数字答案。
    GSM8K 标准格式通常以 '#### <number>' 结尾。
    """
    # 1. 尝试寻找标准的 "####" 标记 (GSM8K 训练集格式)
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        # 移除逗号 (例如 1,234 -> 1234) 和结尾句号
        answer_part = answer_part.replace(",", "").replace(".", "")
        # 提取第一个连续的数字
        matches = re.findall(r'-?\d+\.?\d*', answer_part)
        if matches:
            return matches[0]
    
    # 2. 兜底：如果没有标记，尝试提取文本中出现的最后一个数字
    # (针对模型可能没生成 #### 的情况)
    text_clean = text.replace(",", "")
    matches = re.findall(r'-?\d+\.?\d*', text_clean)
    if matches:
        return matches[-1]
    
    return None

def prepare_gsm8k_data(tokenizer, num_shots=8):
    """
    加载本地 GSM8K 数据并构建 8-shot CoT Prompt。
    """
    # 🚨 请确认你的本地路径
    BASE_PATH = "/data2/group_谈海生/lagin/data/gsm8k/main"
    TRAIN_FILE = f"{BASE_PATH}/train-00000-of-00001.parquet"
    TEST_FILE = f"{BASE_PATH}/test-00000-of-00001.parquet"
    
    print(f"正在加载 GSM8K (Local Mode: {BASE_PATH})...")
    try:
        # 加载训练集用于构建 8-shot 示例
        train_data = load_dataset("parquet", data_files={"train": TRAIN_FILE}, split="train")
        # 加载测试集用于评估
        test_data = load_dataset("parquet", data_files={"test": TEST_FILE}, split="test")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

    # 1. 构建 8-shot 头部 (Prompt Header)
    # 格式:
    # Question: ...
    # Answer: ...
    shot_examples = train_data.select(range(min(num_shots, len(train_data))))
    few_shot_prompt = ""
    for ex in shot_examples:
        few_shot_prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        
    print(f"Few-shot header length: {len(few_shot_prompt)} chars")

    # 2. 构建测试样本
    processed_samples = []
    print(f"正在构建 {len(test_data)} 个测试样本...")
    
    for ex in test_data:
        # 拼接 Prompt: Header + Current Question
        full_prompt = few_shot_prompt + f"Question: {ex['question']}\nAnswer:"
        
        # 提取正确答案 (Ground Truth)
        ground_truth = extract_answer_gsm8k(ex['answer'])
        
        # 保存 (prompt_text, ground_truth_str)
        # 注意：这里存文本，而不是 input_ids，方便后续 batch 处理时做 left padding
        processed_samples.append((full_prompt, ground_truth))
        
    return processed_samples

def evaluate_gsm8k(model, samples, tokenizer, caches, batch_size=4, max_new_tokens=256):
    """
    GSM8K 评测主循环 (生成模式)
    """
    model.eval()
    for c in caches: c.reset_stats()
    
    correct = 0
    total = 0
    
    # 设置 Padding (Qwen 默认没有 pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 🟢 1. Collate Function (关键：左填充)
    def collate_fn(batch):
        prompts = [item[0] for item in batch]
        answers = [item[1] for item in batch]
        
        # 生成任务必须使用 Left Padding，否则 Pad Token 会干扰生成
        tokenizer.padding_side = "left" 
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        
        return inputs, answers

    # 🟢 2. DataLoader
    data_loader = DataLoader(samples, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    print(f"开始 GSM8K 生成评测 (BS={batch_size}, Max Tokens={max_new_tokens})...")
    
    # 清理全局 Mask (生成过程中很难动态维护 Mask，暂且置空)
    # 这意味着 Padding Token 也会触发 Cache 更新，但这在生成任务中影响较小
    CachePriorBlockWrapper.CURRENT_MASK = None 
    
    pbar = tqdm(data_loader, desc="GSM8K Gen")
    
    for batch_inputs, batch_truths in pbar:
        input_ids = batch_inputs.input_ids.to("cuda")
        attention_mask = batch_inputs.attention_mask.to("cuda")
        
        # 🟢 3. 执行生成 (Generate)
        # Cache-Prior 逻辑会在 generate 内部调用的 forward 中自动生效
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, # 贪婪解码 (复现标准)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True   # 必须开启 KV Cache
            )
        
        # 🟢 4. 解码与评估
        # 只解码新生成的部分
        input_len = input_ids.shape[1]
        new_tokens = generated_ids[:, input_len:]
        decoded_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        for pred_text, true_ans in zip(decoded_texts, batch_truths):
            pred_ans = extract_answer_gsm8k(pred_text)
            
            # 比较逻辑 (数字比较)
            if pred_ans is not None and true_ans is not None:
                try:
                    # 处理 1000.0 == 1000 的情况
                    if float(pred_ans) == float(true_ans):
                        correct += 1
                except ValueError:
                    pass # 解析失败算错
            
            total += 1
            
        # 更新进度
        curr_acc = correct / total
        total_hits = sum(c.hits for c in caches)
        total_queries = sum(c.total_queries for c in caches)
        miss_rate = 1.0 - (total_hits/total_queries) if total_queries>0 else 0
        
        pbar.set_postfix({"Acc": f"{curr_acc:.2%}", "Miss": f"{miss_rate:.2%}"})
        
    final_acc = correct / total
    total_hits = sum(c.hits for c in caches)
    total_queries = sum(c.total_queries for c in caches)
    final_miss = 1.0 - (total_hits/total_queries) if total_queries>0 else 0
    
    return final_acc, final_miss