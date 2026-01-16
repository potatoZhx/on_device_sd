import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 模型配置
MODELS = [
    {"id": "/data2/group_谈海生/lagin/models/Qwen1.5-MoE-A2.7B", "name": "Qwen1.5-MoE-A2.7B", "color": "red", "marker": "o"},
    {"id": "/data2/group_谈海生/lagin/models/DeepSeek-V2-Lite", "name": "DeepSeek-V2-Lite", "color": "purple", "marker": "o"},
    {"id": "/data2/group_谈海生/lagin/models/Phi-3.5-MoE-instruct", "name": "Phi-3.5-MoE-instruct", "color": "orange", "marker": "o"},
    {"id": "/data2/group_谈海生/lagin/models/Mixtral-8x7B-v0.1", "name": "Mixtral-8x7B-v0.1", "color": "yellow", "marker": "o"},
]

def load_results(results_file):
    """
    从JSON文件加载结果
    
    Args:
        results_file: JSON结果文件路径
    
    Returns:
        all_results: 加载的结果字典
    """
    if not os.path.exists(results_file):
        print(f"❌ 结果文件 {results_file} 不存在")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    print(f"✅ 成功加载结果文件: {results_file}")
    return all_results

def plot_expert_resampling(all_results, output_file="expert_resampling_subset_curve.png"):
    """
    绘制专家重采样曲线
    
    Args:
        all_results: 结果字典
        output_file: 输出图片文件名
    """
    plt.figure(figsize=(8, 6))
    
    for model_config in MODELS:
        model_name = model_config["name"]
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        ms = [r['m'] for r in results]
        ppls = [r['ppl'] for r in results]
        
        # 绘制曲线
        plt.plot(ms, ppls, marker=model_config["marker"], linestyle='-', 
                 color=model_config["color"], label=model_config["name"])
    
    # 设置图表属性
    plt.title("Expert Resampling")
    plt.xlabel("Randomly replace expert with rank k")
    plt.ylabel("Wikitext Perplexity")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(loc='upper right')
    
    # 添加虚线表示基线困惑度
    plt.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5)
    
    # 调整y轴范围
    plt.ylim(4, 14)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ 专家重采样曲线已保存为 '{output_file}'")
    plt.close()

def plot_combined_analysis(remove_results, replace_results, output_file="combined_expert_sensitivity_analysis.png"):
    """
    绘制组合分析图表
    
    Args:
        remove_results: remove模式的结果字典
        replace_results: replace模式的结果字典
        output_file: 输出图片文件名
    """
    plt.figure(figsize=(12, 5))
    
    # 专家删除子图 (remove模式)
    plt.subplot(1, 2, 1)
    for model_config in MODELS:
        model_name = model_config["name"]
        if model_name not in remove_results:
            continue
        
        results = remove_results[model_name]
        ms = [r['m'] for r in results]
        ppls = [r['ppl'] for r in results]
        
        plt.plot(ms, ppls, marker=model_config["marker"], linestyle='-', 
                 color=model_config["color"], label=model_config["name"])
    
    plt.title("Expert Pruning")
    plt.xlabel("Prune expert with rank ≥ k")
    plt.ylabel("Wikitext Perplexity")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(4, 14)
    plt.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5)
    
    # 专家重采样子图 (replace模式)
    plt.subplot(1, 2, 2)
    for model_config in MODELS:
        model_name = model_config["name"]
        if model_name not in replace_results:
            continue
        
        results = replace_results[model_name]
        ms = [r['m'] for r in results]
        ppls = [r['ppl'] for r in results]
        
        plt.plot(ms, ppls, marker=model_config["marker"], linestyle='-', 
                 color=model_config["color"], label=model_config["name"])
    
    plt.title("Expert Resampling")
    plt.xlabel("Randomly replace expert with rank k")
    plt.ylabel("Wikitext Perplexity")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 将replace模式的y轴范围设置为与remove模式一致
    plt.ylim(4, 14)
    plt.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    # 添加整体标题
    plt.suptitle("专家敏感性分析", fontsize=14, y=1.02)
    
    # 保存组合图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ 组合分析图表已保存为 '{output_file}'")
    plt.close()

def main():
    import argparse
    remove_results_file = "../expert_remove_results.json"
    replace_results_file = "../expert_replace_results.json"
    parser = argparse.ArgumentParser(description="绘制专家子集推理分析图表")
    parser.add_argument("--plot-combined", action="store_true", default=True,
                      help="绘制组合分析图表")
    
    args = parser.parse_args()
    
    # 加载两种模式的结果
    remove_results = load_results(remove_results_file)
    replace_results = load_results(replace_results_file)
    
    if remove_results is None and replace_results is None:
        print("❌ 没有加载到任何结果文件")
        return
    
    # 绘制组合图表
    if args.plot_combined and remove_results and replace_results:
        plot_combined_analysis(remove_results, replace_results)
    
    print("\n=== 🎨 图表绘制完成 ===")

if __name__ == "__main__":
    main()