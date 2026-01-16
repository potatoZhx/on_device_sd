import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_utils import (
    apply_cache_prior_to_model, 
    prepare_gsm8k_data, 
    evaluate_gsm8k
)
# 🟢 导入 Wrapper 以控制开关
from Cache_Prior import CachePriorBlockWrapper

# --- 配置 ---
MODEL_ID = "/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B"
CACHE_RATIO = 0.5
TOP_J = 2         
BATCH_SIZE = 8

# 🟢 修改点：定义 Lambda 列表
LAMBDA_LIST = [0, 0.5]# [0.0, 0.2, 0.5, 0.8, 1.0]

def main():
    print(f"=== 🚀 GSM8K 8-Shot Evaluation (Batch Test) ===")
    
    # 🟢 开启"仅 Decode 模式"
    # 这会告诉底层 Wrapper：如果是 Prefill (seq > 1)，不要更新 Cache，不要计算 Bias
    CachePriorBlockWrapper.ONLY_CACHE_ON_DECODE = True
    print(f"⚙️  Config: ONLY_CACHE_ON_DECODE = {CachePriorBlockWrapper.ONLY_CACHE_ON_DECODE}")
    
    # 1. 加载
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda", 
        dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. 准备数据
    # 如果下载失败，请手动下载
    samples = prepare_gsm8k_data(tokenizer, num_shots=8)
    if not samples: return

    # 3. 结果存储
    results = []

    # 4. 循环测试
    for current_lambda in LAMBDA_LIST:
        print(f"\n" + "-"*40)
        print(f"🔄 Testing Lambda = {current_lambda} ...")
        print("-"*40)
        
        # 重新注入逻辑
        caches = apply_cache_prior_to_model(
            model, 
            cache_ratio=CACHE_RATIO, 
            lambda_val=current_lambda,
            top_j=TOP_J
        )
        
        # 评测 (建议先跑少量样本验证，例如 samples[:20])
        acc, miss_rate = evaluate_gsm8k(model, samples, tokenizer, caches, batch_size=BATCH_SIZE)
        
        results.append({
            "lambda": current_lambda,
            "accuracy": acc,
            "miss_rate": miss_rate
        })
        
        print(f"👉 Result: Acc={acc:.2%}, Miss Rate={miss_rate:.2%}")

    # 5. 输出汇总
    print("\n" + "="*60)
    print(f"📊 Final GSM8K Results")
    print(f"{'Lambda':<10} | {'Accuracy':<10} | {'Miss Rate':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['lambda']:<10.1f} | {res['accuracy']:<10.2%} | {res['miss_rate']:<15.2%}")
    print("="*60)

    # 6. 绘图
    try:
        lambdas = [r['lambda'] for r in results]
        accs = [r['accuracy'] * 100 for r in results]
        miss_rates = [r['miss_rate'] * 100 for r in results]

        plt.figure(figsize=(8, 6))
        plt.plot(miss_rates, accs, marker='o', linestyle='-', color='green', label='Cache-Prior (GSM8K)')
        
        for i, lam in enumerate(lambdas):
            plt.annotate(f"λ={lam}", (miss_rates[i], accs[i]), xytext=(5, 5), textcoords='offset points')

        plt.title(f"GSM8K Accuracy vs Miss Rate (Top-J={TOP_J})")
        plt.xlabel("Cache Miss Rate (%)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig("gsm8k_tradeoff_curve.png")
        print("\n✅ 结果图已保存为 'gsm8k_tradeoff_curve.png'")
    except Exception as e:
        print(f"\n⚠️ 绘图失败: {e}")

if __name__ == "__main__":
    main()