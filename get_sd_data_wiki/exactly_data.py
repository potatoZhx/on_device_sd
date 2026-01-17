import json
import os
import numpy as np
from tqdm import tqdm

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
                if i > 0 and (int_out[i-1] != record["baseline"]["output"][i-1]):
                    is_prefix_matched = True # False
                
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
    print("* Mean α 仅在 Prefix Match 为 True 的样本中统计 (符合论文接受率定义)")
    print(f"详细数据已存入: {output_jsonl}")
    print("="*85)

if __name__ == "__main__":
    # 请确保路径正确
    DIR = "./data/results_replace_last_two_with_topp"
    INPUT_FILE = "experiment_summary_20260108_144751.jsonl"
    OUTPUT_ALPHA = "step_empirical_rates_0108_f.jsonl"
    # config = {'method': 'theoretical'}
    config = {'method': 'empirical'}
    if os.path.exists(os.path.join(DIR, INPUT_FILE)):
        calculate_acceptance_with_config(os.path.join(DIR, INPUT_FILE), os.path.join(DIR, OUTPUT_ALPHA), config)
    else:
        print(f"❌ 找不到输入文件: {os.path.join(DIR, INPUT_FILE)}")



'''0104 
theoretical
=====================================================================================
Step   | Mean α*    | Std Dev    | Match Ratio     | Theo. Speedup
-------------------------------------------------------------------------------------
1      | 0.5383     | 0.2286     | 100.00%         | 2.16x
2      | 0.5027     | 0.2443     | 47.19%          | 2.01x
3      | 0.4762     | 0.2441     | 24.75%          | 1.91x
4      | 0.4715     | 0.2352     | 12.87%          | 1.89x
5      | 0.5131     | 0.2532     | 8.25%           | 2.05x
6      | 0.5448     | 0.2336     | 3.96%           | 2.19x
7      | 0.5694     | 0.2539     | 3.30%           | 2.32x
8      | 0.6524     | 0.1464     | 2.64%           | 2.85x
9      | 0.5544     | 0.2841     | 2.64%           | 2.24x
10     | 0.6728     | 0.0969     | 1.65%           | 3.02x
-------------------------------------------------------------------------------------

empirical
=====================================================================================
Step   | Mean α*    | Std Dev    | Match Ratio     | Theo. Speedup
-------------------------------------------------------------------------------------
1      | 0.6096     | 0.3892     | 100.00%         | 2.55x
2      | 0.6031     | 0.4092     | 47.19%          | 2.51x
3      | 0.5512     | 0.4038     | 24.75%          | 2.23x
4      | 0.6553     | 0.3793     | 12.87%          | 2.87x
5      | 0.5895     | 0.4186     | 8.25%           | 2.43x
6      | 0.7530     | 0.3214     | 3.96%           | 3.87x
7      | 0.8437     | 0.3236     | 3.30%           | 5.41x
8      | 0.9807     | 0.0511     | 2.64%           | 10.00x
9      | 0.6383     | 0.4478     | 2.64%           | 2.75x
10     | 1.0000     | 0.0000     | 1.65%           | 11.00x
-------------------------------------------------------------------------------------
'''

'''
=====================================================================================
Step   | Mean α*    | Std Dev    | Match Ratio     | Theo. Speedup
-------------------------------------------------------------------------------------
1      | 0.4210     | 0.2132     | 100.00%         | 1.73x
2      | 0.5164     | 0.2163     | 37.62%          | 2.07x
3      | 0.4915     | 0.2162     | 17.49%          | 1.97x
4      | 0.5481     | 0.2189     | 6.27%           | 2.21x
5      | 0.5889     | 0.1772     | 3.96%           | 2.43x
6      | 0.5474     | 0.2238     | 2.64%           | 2.21x
7      | 0.3595     | 0.1413     | 1.32%           | 1.56x
8      | 0.3186     | 0.1925     | 0.66%           | 1.47x
9      | 0.4130     | 0.0000     | 0.33%           | 1.70x
10     | 0.8778     | 0.0000     | 0.33%           | 6.23x
-------------------------------------------------------------------------------------


=====================================================================================
Step   | Mean α*    | Std Dev    | Match Ratio     | Theo. Speedup
-------------------------------------------------------------------------------------
1      | 0.4220     | 0.3970     | 100.00%         | 1.73x
2      | 0.5667     | 0.4118     | 37.62%          | 2.30x
3      | 0.4319     | 0.3353     | 17.49%          | 1.76x
4      | 0.6544     | 0.3911     | 6.27%           | 2.87x
5      | 0.7131     | 0.3789     | 3.96%           | 3.40x
6      | 0.6930     | 0.3998     | 2.64%           | 3.20x
7      | 0.5036     | 0.4964     | 1.32%           | 2.01x
8      | 0.5002     | 0.4998     | 0.66%           | 2.00x
9      | 1.0000     | 0.0000     | 0.33%           | 11.00x
10     | 1.0000     | 0.0000     | 0.33%           | 11.00x
-------------------------------------------------------------------------------------
'''
