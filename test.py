import json
import os
import math

def is_nan(value):
    """检查一个值是否为 NaN"""
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, list):
        return any(is_nan(v) for v in value)
    if isinstance(value, dict):
        return any(is_nan(v) for v in value.values())
    return False

def check_nan_in_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"🔍 正在扫描文件: {file_path} ...")
    
    nan_lines = []
    total_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            total_lines += 1
            
            # 1. 快速字符串检测 (不区分大小写)
            # 这种方法极快，能捕捉到直接写入文件的 "NaN" 或 "nan"
            lower_line = line.lower()
            if "nan" in lower_line:
                try:
                    data = json.loads(line)
                    req_id = data.get("metadata", {}).get("req_id", "未知 ID")
                    
                    # 2. 定位具体的键
                    affected_keys = []
                    # 针对你的数据结构进行重点检查
                    if "intervention" in data:
                        for i, step in enumerate(data["intervention"]["steps"]):
                            if any(is_nan(v) for v in step.get("full_logits", [])):
                                affected_keys.append(f"intervention.steps[{i}].full_logits")
                            if any(is_nan(v) for v in step.get("final_embedding", [])):
                                affected_keys.append(f"intervention.steps[{i}].final_embedding")
                    
                    if "baseline" in data:
                        for i, step in enumerate(data["baseline"]["steps"]):
                            if any(is_nan(v) for v in step.get("full_logits", [])):
                                affected_keys.append(f"baseline.steps[{i}].full_logits")

                    nan_lines.append({
                        "line": line_idx,
                        "req_id": req_id,
                        "keys": affected_keys
                    })
                except json.JSONDecodeError:
                    nan_lines.append({
                        "line": line_idx, 
                        "req_id": "解析失败", 
                        "keys": ["非法 JSON 格式"]
                    })

    # --- 输出报告 ---
    print("\n" + "="*50)
    if nan_lines:
        print(f"⚠️ 发现 {len(nan_lines)} 行数据包含 NaN (总行数: {total_lines})")
        print("-" * 50)
        for item in nan_lines[:10]: # 只列出前10个
            print(f"行号: {item['line']} | Request ID: {item['req_id']}")
            print(f"  影响位置: {', '.join(item['keys'][:3])}{'...' if len(item['keys']) > 3 else ''}")
        if len(nan_lines) > 10:
            print(f"... 还有 {len(nan_lines) - 10} 处错误未列出。")
    else:
        print(f"✅ 完美！文件清爽，未发现 NaN 值 (总行数: {total_lines})。")
    print("="*50)

if __name__ == "__main__":
    # 修改为你的文件路径
    TARGET_FILE = "./get_sd_data/data/results_20260104_164906/experiment_summary_20260104_164906.jsonl"
    check_nan_in_jsonl(TARGET_FILE)