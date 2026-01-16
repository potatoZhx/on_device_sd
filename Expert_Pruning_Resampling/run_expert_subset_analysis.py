import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_utils import prepare_wikitext_data, evaluate_perplexity
from ExpertSubsetInference import apply_expert_subset_to_model, update_use_top_m

# 支持多个模型进行比较
MODELS = [
    {"id": "/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B", "name": "Qwen1.5-MoE-A2.7B", "color": "red", "marker": "o"},
    {"id": "/data2/group_谈海生/lagin/models/DeepSeek-V2-Lite", "name": "DeepSeek-V2-Lite", "color": "purple", "marker": "o"},
    {"id": "/data2/group_谈海生/lagin/models/Phi-3.5-MoE-instruct", "name": "Phi-3.5-MoE-instruct", "color": "orange", "marker": "o"},
    {"id": "/data2/group_谈海生/lagin/models/Mixtral-8x7B-v0.1", "name": "Mixtral-8x7B-v0.1", "color": "yellow", "marker": "o"},
]

# 全局配置
SEQ_LEN = 1024      # WikiText 评测的上下文长度
MAX_USE_TOP_M = 7   # 最大使用的top_m专家数量
BATCH_SIZE = 8      # 批量大小

# 任务配置
CONFIG = {
    "task_remove": {
        "enabled": True,
        "mode": "remove",
        "results_file": "expert_remove_results.json"
    },
    "task_replace": {
        "enabled": True,
        "mode": "replace",
        "results_file": "expert_replace_results.json"
    }
}

def get_original_top_k(model):
    """
    获取模型的原始top_k配置
    
    Args:
        model: 模型对象
        
    Returns:
        original_top_k: 原始top_k配置
    """
    if hasattr(model.config, 'num_experts_per_tok'):
        return model.config.num_experts_per_tok
    elif hasattr(model.config, 'top_k'):
        return model.config.top_k
    else:
        # 从模型中推断top_k
        for layer in model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "top_k"):
                return layer.mlp.top_k
            elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "top_k"):
                return layer.block_sparse_moe.top_k
        
        raise ValueError("无法确定模型的top_k配置")

def run_expert_subset_task(mode, results_file):
    print(f"\n=== 🚀 开始专家子集推理分析 (模式: {mode}) ===")
   
    # ------------------------------------------------------------------
    # 🟢 改进 1: 断点续传逻辑 - 尝试加载已有结果
    # ------------------------------------------------------------------
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f" 📂 发现现有结果文件，已加载 {len(all_results)} 个模型的数据")
        except Exception as e:
            print(f" ⚠️ 读取现有文件失败 ({e})，将重新开始")
            all_results = {}
    else:
        all_results = {}
    
    for model_config in MODELS:
        MODEL_ID = model_config["id"]
        MODEL_NAME = model_config["name"]

        # 可选：如果该模型已经有结果，可以选择跳过
        # if MODEL_NAME in all_results:
        #     print(f" ⏩ 模型 {MODEL_NAME} 已存在结果，跳过...")
        #     continue
        
        print(f"\n2. 加载模型: {MODEL_NAME}...")

        # ------------------------------------------------------------------
        # 显存清理 & 加载
        # ------------------------------------------------------------------
        gc.collect()
        torch.cuda.empty_cache()

        # 显存分配策略: 限制 GPU 0 使用 60G，给推理留空间
        max_memory_mapping = {0: "60GiB", 1: "75GiB"}

        # 初始化变量，防止 finally 中报错
        model = None
        subset_model = None
        tokenizer = None

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                device_map="auto", 
                dtype=torch.bfloat16,
                max_memory=max_memory_mapping,
                trust_remote_code=True
            )
            print(f"   ✅ 模型加载成功 (Memory: {model.hf_device_map})")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            eval_batches = prepare_wikitext_data(tokenizer, seq_len=SEQ_LEN)
            
            original_top_k = get_original_top_k(model)
            print(f"   模型 {MODEL_NAME} 原始 top_k={original_top_k}")
            
            model_results = []
            max_m = min(MAX_USE_TOP_M, original_top_k)
            
            # ------------------------------------------------------------------
            # 🟢 改进 2: 范围逻辑修正
            # replace 模式建议从 1 开始 (Rank 1)，防止索引 -1 问题
            # ------------------------------------------------------------------
            if mode == "remove":
                m_range = range(1, max_m + 1)
            elif mode == "replace":
                m_range = range(1, max_m + 1) # 修改为 1 起步
            else:
                m_range = range(1, max_m + 1)

            # 应用包装器 (初始状态)
            subset_model = apply_expert_subset_to_model(model, use_top_m=1, mode=mode)
            
            for m in m_range:
                print(f"\n   🔄 测试 m={m} ({mode})...")
                update_use_top_m(subset_model, use_top_m=m, mode=mode)
                
                # 评测
                ppl = evaluate_perplexity(subset_model, eval_batches, batch_size=BATCH_SIZE)
                
                # ------------------------------------------------------------------
                # 🟢 改进 3: 类型安全转换 (Tensor -> Float)
                # 防止 JSON 序列化报错
                # ------------------------------------------------------------------
                if torch.is_tensor(ppl):
                    ppl_val = ppl.item()
                else:
                    ppl_val = float(ppl)
                
                model_results.append({ "m": m, "ppl": ppl_val })
                print(f"   👉 结果: m={m}, PPL={ppl_val:.4f}")
            
            # 更新总结果
            all_results[MODEL_NAME] = model_results

            # ------------------------------------------------------------------
            # 🟢 改进 4: 实时保存 (Checkpointing)
            # 每跑完一个模型就保存一次，防止长时间运行后崩盘导致数据全丢
            # ------------------------------------------------------------------
            print(f"   💾 [Checkpoint] 正在保存 {MODEL_NAME} 的结果...")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ 模型 {MODEL_NAME} 评测失败: {e}")
            traceback.print_exc() # 打印完整堆栈信息，便于调试
            # 不 continue，让 finally 块正常执行清理

        finally:
            print(f"   🧹 清理模型 {MODEL_NAME}...")
            
            # 删除变量引用
            if subset_model is not None: del subset_model
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
            print("   ✅ 显存已释放\n")
    
    print(f"\n✅ 所有任务执行完毕，最终结果保存在: {results_file}")
    return all_results

    
def main():
    print("=== 🚀 专家子集推理分析主程序 ===")
    
    # 执行所有启用的任务
    for task_name, task_config in CONFIG.items():
        if task_config["enabled"]:
            run_expert_subset_task(
                mode=task_config["mode"],
                results_file=task_config["results_file"]
            )
        else:
            print(f"\n⚠️  任务 {task_name} 已禁用，跳过执行")
    
    print("\n=== 🎉 所有任务执行完成 ===")

if __name__ == "__main__":
    main()