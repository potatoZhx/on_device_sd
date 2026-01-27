import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_coverage_analysis(results, output_path):
    """
    绘制每一层的专家覆盖率折线图
    X轴：Layer Index
    Y轴：Average Coverage Ratio
    线条：不同 Step
    """
    plt.figure(figsize=(12, 6), dpi=150)
    
    # 颜色映射，支持多步
    cmap = plt.get_cmap("viridis")
    num_steps = len(results)
    
    for i, step_data in enumerate(results):
        step_idx = step_data["step"]
        layers = step_data["layers"]
        
        x = [l["layer_idx"] for l in layers]
        y = [l["avg_coverage_ratio"] for l in layers]
        
        color = cmap(i / num_steps)
        plt.plot(x, y, label=f"Step {step_idx}", color=color, alpha=0.8, linewidth=1.5)
        
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Average Expert Coverage", fontsize=12)
    plt.title("Expert Coverage per Layer across Steps", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"✅ 覆盖率图表已保存至: {output_path}")

def softmax(logits):
    logits = np.array(logits)
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum()

def calculate_acceptance_with_config(input_path, output_jsonl, config):
    """
    根据配置计算接受率。
    config['method'] = 'theoretical' : 使用 sum(min(p, q)) -> 理论期望
    config['method'] = 'empirical'   : 使用 min(1, p(x)/q(x)) -> 实际采样接受率
    """
    gamma = 10
    step_stats = {i: {"alphas": [], "match_count": 0} for i in range(gamma)}
    total_samples = 0
    
    method = config.get('method', 'theoretical')
    print(f"正在分析数据 [模式: {method}]: {input_path}")
    
    with open(output_jsonl, 'w', encoding='utf-8') as out_f, \
         open(input_path, 'r', encoding='utf-8') as in_f:
        
        for line in tqdm(in_f, desc="Processing Steps"):
            try:
                record = json.loads(line)
            except: continue
                
            total_samples += 1
            # 获取干预路径采样出的词序列
            # 注意：第 i 步产生的词是 int_out[i]
            int_out = record["intervention"]["output"]
            
            steps_int = record["intervention"]["steps"]
            steps_base = record["baseline"]["steps"]
            
            sample_alphas = []
            is_prefix_matched = True 
            
            for i in range(gamma):
                q_probs = softmax(steps_int[i]["full_logits"])
                p_probs = softmax(steps_base[i]["full_logits"])
                
                if method == 'theoretical':
                    # 公式：sum(min(p, q))
                    alpha_t = np.sum(np.minimum(p_probs, q_probs))
                else:
                    # 公式：min(1, p(x)/q(x))
                    # x 是干预模型在这一步实际选出的 token 索引
                    # 在你的数据中，int_out 包含了生成的序列
                    target_token_id = int_out[i]
                    
                    # 词表很大，直接取该词在两个分布中的概率
                    q_x = q_probs[target_token_id]
                    p_x = p_probs[target_token_id]
                    
                    # 避免除以 0
                    alpha_t = min(1.0, p_x / (q_x + 1e-10))
                
                sample_alphas.append(float(alpha_t))
                
                # 判断前缀匹配
                # if i > 0 and (int_out[i-1] != record["baseline"]["output"][i-1]):
                #    is_prefix_matched = True # False
                
                if is_prefix_matched:
                    step_stats[i]["match_count"] += 1
                    step_stats[i]["alphas"].append(alpha_t)
            
            out_f.write(json.dumps({
                "request_id": record["metadata"]["req_id"],
                "step_alphas": sample_alphas
            }) + "\n")

    # --- 打印汇总表 ---
    print("\n" + "="*85)
    print(f"{'Step':<6} | {'Mean α*':<10} | {'Std Dev':<10} | {'Match Ratio':<15} | {'Theo. Speedup'}")
    print("-" * 85)
    
    for i in range(gamma):
        matched_alphas = np.array(step_stats[i]["alphas"])
        
        if len(matched_alphas) > 0:
            mean_a = np.mean(matched_alphas)
            std_a = np.std(matched_alphas)
            # 理论加速比公式: (1 - alpha^(gamma+1)) / (1 - alpha)
            # 如果 alpha 非常接近 1，极限值为 gamma + 1
            if mean_a < 0.9999:
                speedup = (1 - mean_a**(gamma + 1)) / (1 - mean_a)
            else:
                speedup = gamma + 1
        else:
            mean_a, std_a, speedup = 0, 0, 0
            
        ratio = step_stats[i]["match_count"] / total_samples
        
        print(f"{i+1:<6} | {mean_a:<10.4f} | {std_a:<10.4f} | {ratio:<15.2%} | {speedup:<.2f}x")
    
    print("-" * 85)
    print(f"详细数据已存入: {output_jsonl}")
    print("="*85)
    
    # --- 保存汇总结果到新文件 ---
    summary_data = []
    for i in range(gamma):
        matched_alphas = np.array(step_stats[i]["alphas"])
        if len(matched_alphas) > 0:
            mean_a = float(np.mean(matched_alphas))
            std_a = float(np.std(matched_alphas))
            if mean_a < 0.9999:
                speedup = float((1 - mean_a**(gamma + 1)) / (1 - mean_a))
            else:
                speedup = float(gamma + 1)
        else:
            mean_a, std_a, speedup = 0.0, 0.0, 0.0
            
        ratio = float(step_stats[i]["match_count"] / total_samples)
        
        summary_data.append({
            "step": i + 1,
            "mean_alpha": mean_a,
            "std_dev": std_a,
            "match_ratio": ratio,
            "theo_speedup": speedup
        })
        
    summary_file = output_jsonl.replace(".jsonl", "_avg_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"平均结果已保存至: {summary_file}")

def calculate_expert_coverage(input_path, output_json):
    """
    计算专家覆盖率：对比 intervention 和 baseline 策略
    输出：每层、每步的平均专家覆盖率（Per-Sample Average）
    """
    print(f"正在计算专家覆盖率: {input_path}")
    
    # 存储结构: stats[step][layer_idx] = 累加器
    # 用于计算平均值
    stats = {} 
    
    total_samples = 0
    num_layers = 0
    num_steps = 0
    
    with open(input_path, 'r', encoding='utf-8') as in_f:
        for line in tqdm(in_f, desc="Processing Expert Coverage"):
            try:
                record = json.loads(line)
            except: continue
            
            total_samples += 1
            
            # 初始化层数和步数
            if num_steps == 0:
                num_steps = len(record["intervention"]["steps"])
                num_layers = len(record["intervention"]["steps"][0]["router_modified"]["ids"])
                
                # 初始化 stats 结构
                for step in range(num_steps):
                    stats[step] = {}
                    for l in range(num_layers):
                        stats[step][l] = {
                            "sum_base_count": 0,
                            "sum_int_count": 0,
                            "sum_intersection": 0,
                            "sum_coverage_ratio": 0.0,
                            "count": 0
                        }

            # 遍历步数
            for step_idx in range(num_steps):
                int_step_data = record["intervention"]["steps"][step_idx]
                base_step_data = record["baseline"]["steps"][step_idx]
                
                int_layer_ids = int_step_data["router_original"]["ids"]
                base_layer_ids = base_step_data["router_standard"]["ids"]
                
                for layer_idx in range(num_layers):
                    # 获取单样本、单步、单层的专家集合
                    # 辅助函数：展平并转为集合
                    def get_flat_set(items):
                        res = set()
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, list):
                                    # 处理嵌套列表 [[id1, id2...]]
                                    for sub in item:
                                        res.add(sub)
                                else:
                                    res.add(item)
                        else:
                            res.add(items)
                        return res

                    int_experts = get_flat_set(int_layer_ids[layer_idx])
                    base_experts = get_flat_set(base_layer_ids[layer_idx])
                    
                    # 计算单样本指标
                    intersection = int_experts.intersection(base_experts)
                    base_len = len(base_experts)
                    int_len = len(int_experts)
                    inter_len = len(intersection)
                    
                    ratio = inter_len / base_len if base_len > 0 else 0.0
                    
                    # 累加
                    stats[step_idx][layer_idx]["sum_base_count"] += base_len
                    stats[step_idx][layer_idx]["sum_int_count"] += int_len
                    stats[step_idx][layer_idx]["sum_intersection"] += inter_len
                    stats[step_idx][layer_idx]["sum_coverage_ratio"] += ratio
                    stats[step_idx][layer_idx]["count"] += 1

    # 汇总平均结果
    results = []
    
    for step in range(num_steps):
        step_res = {"step": step + 1, "layers": [], "step_avg_layer_coverage": 0.0}
        total_step_coverage = 0.0
        
        for layer in range(num_layers):
            s = stats[step][layer]
            cnt = s["count"]
            if cnt == 0: cnt = 1
            
            avg_cov = s["sum_coverage_ratio"] / cnt
            
            step_res["layers"].append({
                "layer_idx": layer,
                "avg_base_count": s["sum_base_count"] / cnt,
                "avg_int_original_count": s["sum_int_count"] / cnt,
                "avg_intersection": s["sum_intersection"] / cnt,
                "avg_coverage_ratio": avg_cov
            })
            total_step_coverage += avg_cov
            
        step_res["step_avg_layer_coverage"] = total_step_coverage / num_layers if num_layers > 0 else 0.0
        results.append(step_res)
        
    # 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"专家覆盖率分析已保存至: {output_json}")
    
    # 绘制图表
    try:
        plot_coverage_analysis(results, output_json.replace(".json", ".png"))
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
    
    # 打印前几层的简报
    print("\n=== Expert Coverage Summary (Step 1, First 5 Layers) [Average Per Sample] ===")
    first_step = results[0]
    print(f"{'Layer':<6} | {'Avg Base':<10} | {'Avg Int(Orig)':<14} | {'Avg Overlap':<12} | {'Avg Coverage'}")
    print("-" * 70)
    for l_data in first_step["layers"][:5]:
        print(f"{l_data['layer_idx']:<6} | {l_data['avg_base_count']:<10.2f} | {l_data['avg_int_original_count']:<14.2f} | {l_data['avg_intersection']:<12.2f} | {l_data['avg_coverage_ratio']:.2%}")
    print("=" * 70)

if __name__ == "__main__":
    # 请确保路径正确
    # DIR = "./data/mtbench_results_replace_last_one_with_topp"
    # 使用用户最近使用的目录
    DIR = "/data2/group_谈海生/lagin/data/Sd_Data/data/wiki_results_1_with_Qwen3-30B-A3B-Base"
    # 自动寻找最新的 jsonl 文件
    files = [f for f in os.listdir(DIR) if f.endswith('.jsonl') and 'summary' in f]
    if not files:
        print(f"❌ 目录 {DIR} 下没有找到 summary jsonl 文件")
        exit(1)
    
    # 按时间排序取最新的
    files.sort(reverse=True)
    INPUT_FILE = files[0]
    
    OUTPUT_ALPHA = os.path.join(DIR, f"step_empirical_rates_{INPUT_FILE.split('_')[-1]}")
    OUTPUT_COVERAGE = os.path.join(DIR, f"expert_coverage_{INPUT_FILE.split('_')[-1].replace('.jsonl', '.json')}")
    
    config = {'method': 'empirical'}
    
    input_path = os.path.join(DIR, INPUT_FILE)
    if os.path.exists(input_path):
        calculate_acceptance_with_config(input_path, OUTPUT_ALPHA, config)
        calculate_expert_coverage(input_path, OUTPUT_COVERAGE)
    else:
        print(f"❌ 找不到输入文件: {input_path}")


