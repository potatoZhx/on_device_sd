import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_utils import (
    apply_cache_prior_to_model, 
    prepare_mmlu_data, 
    evaluate_mmlu
)

# --- 配置参数 ---
MODEL_ID = "/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B"
CACHE_RATIO = 0.5   # 缓存 50% 的专家 (30/60)
TOP_J = 2           # Qwen 推荐 Top-J=2
MMLU_SUBSET = "all" # 选一个子集测试，或者 'all'

LAMBDA_LIST = [0, 0.5]

def main():
    print(f"=== 🚀 MMLU 5-Shot Evaluation (Batch Test) ===")
    print(f"Testing Lambdas: {LAMBDA_LIST}")
    
    # 1. 加载模型 (只需加载一次)
    print("1. Loading Model to GPU VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda", 
        dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. 准备 MMLU 数据 (只需准备一次)
    # 如果下载失败，请手动下载 cais/mmlu 数据集到本地
    samples = prepare_mmlu_data(tokenizer, subset=MMLU_SUBSET, num_shots=5)
    
    if not samples:
        print("❌ 没有数据，退出。")
        return

    # 3. 结果存储
    results = []

    # 4. 循环测试不同的 Lambda
    for current_lambda in LAMBDA_LIST:
        print(f"\n" + "-"*40)
        print(f"🔄 Testing Lambda = {current_lambda} ...")
        print("-"*40)
        
        caches = apply_cache_prior_to_model(
            model, 
            cache_ratio=CACHE_RATIO, 
            lambda_val=current_lambda,
            top_j=TOP_J
        )
        
        # 5. 运行评测
        # 建议设置 batch_size=8 或 16 以加速
        acc, miss_rate = evaluate_mmlu(model, samples, tokenizer, caches, batch_size=8, use_mask=True)
        
        # 记录结果
        results.append({
            "lambda": current_lambda,
            "accuracy": acc,
            "miss_rate": miss_rate
        })
        
        print(f"👉 Result: Acc={acc:.2%}, Miss Rate={miss_rate:.2%}")

    # 6. 输出汇总表
    print("\n" + "="*60)
    print(f"📊 Final MMLU Results ({MMLU_SUBSET})")
    print(f"{'Lambda':<10} | {'Accuracy':<10} | {'Miss Rate':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['lambda']:<10.1f} | {res['accuracy']:<10.2%} | {res['miss_rate']:<15.2%}")
    print("="*60)

    with open("mmlu_eval_results.txt", "a") as f:
        f.write(f"MMLU Evaluation Results ({MMLU_SUBSET},date={__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"{'Lambda':<10} | {'Accuracy':<10} | {'Miss Rate':<15}\n")
        f.write("-" * 60 + "\n")
        for res in results:
            f.write(f"{res['lambda']:<10.1f} | {res['accuracy']:<10.2%} | {res['miss_rate']:<15.2%}\n")
    print("\n✅ 结果已保存到 'mmlu_eval_results.txt'")

    # 7. 简单的绘图 (保存到文件)
    try:
        lambdas = [r['lambda'] for r in results]
        accs = [r['accuracy'] * 100 for r in results]
        miss_rates = [r['miss_rate'] * 100 for r in results] # 转为百分比

        plt.figure(figsize=(8, 6))
        plt.plot(miss_rates, accs, marker='o', linestyle='-', color='blue', label='Cache-Prior (MMLU)')
        
        # 标注每个点
        for i, lam in enumerate(lambdas):
            plt.annotate(f"λ={lam}", (miss_rates[i], accs[i]), xytext=(5, 5), textcoords='offset points')

        plt.title(f"MMLU Accuracy vs Miss Rate (Top-J={TOP_J})")
        plt.xlabel("Cache Miss Rate (%)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig("mmlu_tradeoff_curve.png")
        print("\n✅ 结果图已保存为 'mmlu_tradeoff_curve.png'")
    except Exception as e:
        print(f"\n⚠️ 绘图失败: {e}")

if __name__ == "__main__":
    main()