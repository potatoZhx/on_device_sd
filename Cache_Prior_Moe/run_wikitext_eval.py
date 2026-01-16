# 没有设置P值控制选择专家的情况
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from moe_utils import (
    move_static_modules_to_gpu, 
    apply_cache_prior_to_model, 
    prepare_wikitext_data, 
    evaluate_perplexity
)

# --- 配置参数 ---
MODEL_ID = "/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B" # 你的本地模型路径
CACHE_RATIO = 0.5   # 缓存 50% 的专家 (30/60)
TOP_J = 2           # 原始分数最高的 J 个专家始终被缓存
SEQ_LEN = 1024      # WikiText 评测的上下文长度
LAMBDA_LIST = [x/10 for x in range(0,11)]

def main():
    print(f"=== 🚀 开始复现 WikiText 评测 (批量 Lambda 测试) ===")
    
    # 1. 加载模型 (只需加载一次)
    print("1. Loading Model to GPU VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda", 
        dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # move_static_modules_to_gpu(model, device="cuda")
    
    # 2. 准备数据 (只需准备一次)
    eval_batches = prepare_wikitext_data(tokenizer, seq_len=SEQ_LEN)
    
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
        
        # 运行评测
        ppl, miss_rate = evaluate_perplexity(model, eval_batches, caches)
        
        # 记录结果
        results.append({
            "lambda": current_lambda,
            "ppl": ppl,
            "miss_rate": miss_rate
        })
        
        print(f"👉 Result: PPL={ppl:.4f}, Miss Rate={miss_rate:.2%}")

    # 5. 输出汇总表
    print("\n" + "="*60)
    print(f"{'Lambda':<10} | {'PPL':<10} | {'Miss Rate':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['lambda']:<10.1f} | {res['ppl']:<10.4f} | {res['miss_rate']:<15.2%}")
    print("="*60)
    
    with open("wikitext_eval_results.txt", "a") as f:
        f.write(f"Wikitext Evaluation Results (date={__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"{'Lambda':<10} | {'PPL':<10} | {'Miss Rate':<15}\n")
        f.write("-" * 60 + "\n")
        for res in results:
            f.write(f"{res['lambda']:<10.1f} | {res['ppl']:<10.4f} | {res['miss_rate']:<15.2%}\n")
    print("\n✅ 结果已保存到 'wikitext_eval_results.txt'")

    # 6. 简单的绘图 (保存到文件)
    try:
        lambdas = [r['lambda'] for r in results]
        ppls = [r['ppl'] for r in results]
        miss_rates = [r['miss_rate'] * 100 for r in results] # 转为百分比

        plt.figure(figsize=(8, 6))
        plt.plot(miss_rates, ppls, marker='o', linestyle='-', color='purple', label='Cache-Prior')
        
        # 标注每个点
        for i, lam in enumerate(lambdas):
            plt.annotate(f"λ={lam}", (miss_rates[i], ppls[i]), xytext=(5, 5), textcoords='offset points')

        plt.title(f"PPL vs Miss Rate Trade-off (Top-J={TOP_J})")
        plt.xlabel("Cache Miss Rate (%)")
        plt.ylabel("Wikitext Perplexity")
        plt.grid(True)
        plt.legend()
        plt.savefig("wikitext_tradeoff_curve.png")
        print("\n✅ 结果图已保存为 'wikitext_tradeoff_curve.png'")
    except Exception as e:
        print(f"\n⚠️ 绘图失败: {e}")

if __name__ == "__main__":
    main()