import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from moe_utils import (
    prepare_wikitext_data, 
    evaluate_perplexity
)
from ExpertSubsetInference import apply_expert_subset_to_model

# --- 配置参数 ---
# 支持多个模型进行比较
MODELS = [
    {"id": "/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B", "name": "Qwen1.5-MoE-A2.7B", "color": "red"},
    # {"id": "/data2/group_谈海生/lagin/models/DeepSeek-V2-Lite", "name": "DeepSeek-V2-Lite", "color": "purple"},
    # {"id": "/data2/group_谈海生/lagin/models/Phi-3.5-MoE-instruct", "name": "Phi-3.5-MoE-instruct", "color": "orange"},
    # {"id": "/data2/group_谈海生/lagin/models/Mixtral-8x7B-v0.1", "name": "Mixtral-8x7B-v0.1", "color": "yellow"},
]

SEQ_LEN = 1024      # WikiText 评测的上下文长度
MAX_USE_TOP_M = 7   # 最大使用的top_m专家数量
BATCH_SIZE = 8      # 批量大小

# 获取模型的原始top_k
def get_original_top_k(model):
    """
    获取模型的原始top_k配置
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

def main():
    print(f"=== 🚀 开始专家子集推理 PPL 检测 ===")
    
    # 准备数据
    print(f"\n1. 准备 WikiText 数据...")
    # 使用第一个模型的分词器准备数据（假设所有模型使用兼容的分词器）
    tokenizer = AutoTokenizer.from_pretrained(MODELS[0]["id"])
    eval_batches = prepare_wikitext_data(tokenizer, seq_len=SEQ_LEN)
    
    # 结果存储
    all_results = {}
    
    # 对每个模型执行评估
    for model_config in MODELS:
        MODEL_ID = model_config["id"]
        MODEL_NAME = model_config["name"]
        
        print(f"\n2. 加载模型: {MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map="cuda", 
            dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 获取模型的原始top_k
        original_top_k = get_original_top_k(model)
        print(f"   模型 {MODEL_NAME} 原始 top_k={original_top_k}")
        
        model_results = []
        
        # 确定要测试的m值范围（从1到original_top_k）
        max_m = min(MAX_USE_TOP_M, original_top_k)
        
        # 测试不同的use_top_m值
        for m in range(1, max_m + 1):
            print(f"\n   🔄 测试使用前 {m} 个专家进行推理...")
            
            # 重新加载模型以确保每次测试都是独立的
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                device_map="cuda", 
                dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 应用专家子集推理
            subset_model = apply_expert_subset_to_model(model, use_top_m=m)
            
            # 运行评测
            ppl = evaluate_perplexity(subset_model, eval_batches, batch_size=BATCH_SIZE)
            
            # 记录结果
            model_results.append({
                "m": m,
                "ppl": ppl
            })
            
            print(f"   👉 结果: use_top_m={m}, PPL={ppl:.4f}")
        
        all_results[MODEL_NAME] = model_results
    
    # 输出汇总表
    print(f"\n" + "="*100)
    print(f"专家子集推理 PPL 检测结果汇总")
    print(f"="*100)
    
    # 构建表头
    header = f"{'m':<5} | "
    for model_name in all_results.keys():
        header += f"{model_name:<25} | "
    print(header)
    print(f"-"*100)
    
    # 构建数据行
    max_m = max([max(r['m'] for r in results) for results in all_results.values()])
    for m in range(1, max_m + 1):
        row = f"{m:<5} | "
        for model_name in all_results.keys():
            ppl = None
            for result in all_results[model_name]:
                if result["m"] == m:
                    ppl = result["ppl"]
                    break
            if ppl is not None:
                row += f"{ppl:<25.4f} | "
            else:
                row += f"{'N/A':<25} | "
        print(row)
    print(f"="*100)
    
    # 保存结果到文件
    with open("expert_subset_eval_results.txt", "w") as f:
        f.write(f"专家子集推理 PPL 检测结果 (date={__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        f.write(f"\n" + "="*100 + "\n")
        f.write(header + "\n")
        f.write(f"-"*100 + "\n")
        
        for m in range(1, max_m + 1):
            row = f"{m:<5} | "
            for model_name in all_results.keys():
                ppl = None
                for result in all_results[model_name]:
                    if result["m"] == m:
                        ppl = result["ppl"]
                        break
                if ppl is not None:
                    row += f"{ppl:<25.4f} | "
                else:
                    row += f"{'N/A':<25} | "
            f.write(row + "\n")
    print(f"\n✅ 结果已保存到 'expert_subset_eval_results.txt'")
    
    # 绘制专家子集推理的 PPL 曲线
    try:
        plt.figure(figsize=(8, 6))
        
        # 设置颜色
        colors = {"Qwen1.5-MoE-A2.7B": "red", "DeepSeek-V2-Lite": "purple", "Phi-3.5-MoE-instruct": "orange", "Mixtral-8x7B-v0.1": "yellow"}
        
        # 绘制每个模型的曲线
        for model_name, results in all_results.items():
            ms = [r['m'] for r in results]
            ppls = [r['ppl'] for r in results]
            plt.plot(ms, ppls, marker='o', linestyle='-', color=colors[model_name], label=model_name)
        
        plt.title("Expert Subset Inference")
        plt.xlabel("Use top-m experts for inference")
        plt.ylabel("Wikitext Perplexity")
        plt.grid(True)
        plt.legend()
        plt.savefig("expert_subset_inference_curve.png")
        print(f"\n✅ 结果图已保存为 'expert_subset_inference_curve.png'")
    except Exception as e:
        print(f"\n⚠️ 绘图失败: {e}")

if __name__ == "__main__":
    main()
