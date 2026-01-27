import json
import os
import numpy as np
import matplotlib.pyplot as plt

def process_fidelity_data(jsonl_path, prefix_match_only=True, max_steps=None):
    """
    对比干预路径 (router_original) 和 基准路径 (router_standard) 的专家路由保真度。
    
    Args:
        jsonl_path: 数据文件路径
        prefix_match_only: 是否仅对比输入前缀相同的步骤
        max_steps: 仅对比前 k 步 (None 表示不限制)
    """
    total_hard = 0
    total_soft = 0
    total_miss = 0
    
    sample_hard_percents = []
    sample_soft_percents = []
    sample_miss_percents = []

    # 构建日志信息
    mode_str = "Prefix Match Only" if prefix_match_only else "All Steps"
    if max_steps:
        mode_str += f" (First {max_steps} Steps)"
    else:
        mode_str += " (Full Sequence)"

    print(f"正在分析路由保真度 [{mode_str}]: {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            steps_int = record["intervention"]["steps"]
            steps_base = record["baseline"]["steps"]
            int_out = record["intervention"]["output"]
            base_out = record["baseline"]["output"]
            
            # --- 确定基础可用步数 ---
            # 首先受限于数据的实际长度
            available_steps = min(len(steps_int), len(steps_base))
            
            # 其次受限于 max_steps 参数 (如果设置了)
            if max_steps is not None:
                available_steps = min(available_steps, max_steps)

            # --- 确定最终对比步数 ---
            if prefix_match_only:
                # 寻找第一个生成分歧的点
                div_idx = 0
                # 注意：这里的 range 也要受限于 available_steps，避免无谓比较
                compare_len = min(len(int_out), len(base_out), available_steps)
                
                for i in range(compare_len):
                    if int_out[i] == base_out[i]:
                        div_idx = i + 1
                    else:
                        break
                
                # 有效步数 = 匹配的 token 数 + 1 (即发生分歧的那一步也算，除非已经到了 limit)
                # 取 min 确保不超过 available_steps
                num_steps_to_compare = min(div_idx + 1, available_steps)
            else:
                num_steps_to_compare = available_steps
            
            # 如果计算出的步数为 0 (例如 max_steps=0 或数据异常)，跳过
            if num_steps_to_compare <= 0:
                continue

            s_hard, s_soft, s_miss, s_total = 0, 0, 0, 0
            
            # --- 遍历有效步骤 ---
            for i in range(num_steps_to_compare):
                orig_layer_ids = steps_int[i]["router_original"]["ids"] 
                std_layer_ids = steps_base[i]["router_standard"]["ids"] 
                
                num_layers = len(orig_layer_ids)
                for l in range(num_layers):
                    gold = std_layer_ids[l] 
                    orig = orig_layer_ids[l] 
                    
                    # 辅助函数：将可能的嵌套列表展平为集合
                    def get_expert_set(data):
                        s = set()
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, list):
                                    for sub in item:
                                        s.add(sub)
                                else:
                                    s.add(item)
                        else:
                            s.add(data)
                        return s

                    gold_set = get_expert_set(gold)
                    orig_set = get_expert_set(orig)
                    
                    # Hard Match: 列表结构完全一致 (顺序、内容)
                    # Soft Match: 激活的专家集合一致
                    if gold == orig:
                        s_hard += 1
                        total_hard += 1
                    elif gold_set == orig_set:
                        s_soft += 1
                        total_soft += 1
                    else:
                        s_miss += 1
                        total_miss += 1
                    s_total += 1
            
            if s_total > 0:
                sample_hard_percents.append(s_hard / s_total * 100)
                sample_soft_percents.append(s_soft / s_total * 100)
                sample_miss_percents.append(s_miss / s_total * 100)

    total_count = total_hard + total_soft + total_miss
    if total_count == 0: 
        print("未找到有效数据。")
        return None, None

    stats = {
        "hard": (total_hard / total_count) * 100,
        "soft": (total_soft / total_count) * 100,
        "miss": (total_miss / total_count) * 100,
        "accuracy": ((total_hard + total_soft) / total_count) * 100,
        "prefix_match_only": prefix_match_only,
        "max_steps": max_steps # 记录到 stats 以便绘图标题使用
    }
    
    return stats, (sample_hard_percents, sample_soft_percents, sample_miss_percents)

def plot_fidelity_figure(stats, distributions, save_path="routing_fidelity_plot.png"):
    hard_dist, soft_dist, miss_dist = distributions
    fig = plt.figure(figsize=(10, 6), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.4)

    # --- 上半部分: 总体百分比堆叠图 ---
    ax0 = fig.add_subplot(gs[0])
    ax0.barh([0], [stats['hard']], color='#4a9b73', label='Hard Matches')
    ax0.barh([0], [stats['soft']], left=[stats['hard']], color='#a4f4ac', label='Soft Matches')
    ax0.barh([0], [stats['miss']], left=[stats['hard'] + stats['soft']], color='#ff7a6b', label='Mismatches')
    
    for val, start, color in zip([stats['hard'], stats['soft'], stats['miss']], 
                                  [0, stats['hard'], stats['hard'] + stats['soft']], 
                                  ['white', 'black', 'white']):
        if val > 5:
            ax0.text(start + val/2, 0, f"{val:.1f}%", va='center', ha='center', fontweight='bold', color=color)
    
    # 动态生成标题后缀
    title_suffix = "(Shared Prefix Only)" if stats['prefix_match_only'] else "(Full Sequence)"
    if stats['max_steps']:
        title_suffix += f" - First {stats['max_steps']} Steps"
    
    ax0.set_title(f"{stats['accuracy']:.1f}% Routing Fidelity {title_suffix}", fontsize=14, pad=10)
    ax0.set_xlim(0, 100)
    ax0.set_axis_off()

    # --- 下半部分: 密度分布图 ---
    ax1 = fig.add_subplot(gs[1])
    ax1.hist(hard_dist, bins=100, range=(0, 100), alpha=0.8, color='#4a9b73', label='Hard Matches', density=True, edgecolor='black', lw=0.5)
    ax1.hist(soft_dist, bins=100, range=(0, 100), alpha=0.6, color='#a4f4ac', label='Soft Matches', density=True, edgecolor='black', lw=0.5)
    ax1.hist(miss_dist, bins=100, range=(0, 100), alpha=0.8, color='#ff7a6b', label='Mismatches', density=True, edgecolor='black', lw=0.5)

    ax1.set_xlabel("Fidelity Percentage (%)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 绘图完成，保存在: {save_path}")

if __name__ == "__main__":
    # === 配置参数 ===
    INPUT_FILE = "experiment_summary_20260117_221546.jsonl"
    DIR = "./data/mtbench_results_3_with_Qwen3-30B-A3B-Base/" 
    
    PREFIX_MATCH_ONLY = False  # 是否仅对比前缀一致的步骤
    
    # [修改点 1] 这里改为列表，支持同时跑多个配置 (None 表示所有步)
    K_STEPS_LIST = [1, 2, 3] 
    # =================
    
    # 使用 os.path.join 拼接路径更安全
    full_input_path = os.path.join(DIR, INPUT_FILE)

    if os.path.exists(full_input_path):
        # 确保输出目录存在
        os.makedirs(DIR, exist_ok=True)

        # [修改点 2] 遍历列表中的每一个 K 值
        for k in K_STEPS_LIST:
            print(f"🔄 正在处理 K_STEPS = {k} ...")
            
            # 运行分析
            stats, dists = process_fidelity_data(
                full_input_path, 
                prefix_match_only=PREFIX_MATCH_ONLY,
                max_steps=k  # 这里传入当前循环的 k
            )
            
            if stats:
                # 构造文件名: fidelity_prefix_{True/False}_k_{Steps/All}.png
                prefix_tag = "prefix_on" if PREFIX_MATCH_ONLY else "prefix_off"
                k_tag = f"k_{k}" if k is not None else "k_all"
                filename = f"fidelity_{prefix_tag}_{k_tag}.png"
                
                save_path = os.path.join(DIR, filename)
                
                plot_fidelity_figure(stats, dists, save_path=save_path)
                print(f"✅ 图表已保存: {filename}")
            else:
                print(f"⚠️ K={k} 时未生成统计数据")

        print("🎉 所有任务处理完成！")
            
    else:
        print(f"❌ 找不到输入文件: {full_input_path}")